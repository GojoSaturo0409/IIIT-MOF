#!/usr/bin/env python3

import os
import argparse
import math
from pathlib import Path
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import random
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import pandas as pd
import json
import traceback
import io
from datetime import timedelta



def set_seed(seed: int):
    
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


def ensure_cif_ext(name: str) -> str:
    if not isinstance(name, str):
        name = str(name)
    return name if name.lower().endswith('.cif') else name + '.cif'


def voxel_path_to_cif_name(p: Path) -> str:
    name = p.name
    lower = name.lower()
    for s in ['_vox.npz', '_vox.pt', '_vox.npy', '_vox', '.npy', '.pt', '.npz']:
        if lower.endswith(s):
            base = name[:len(name)-len(s)]
            return ensure_cif_ext(base)
    return ensure_cif_ext(p.stem)


def load_targets_csv(csv_path: str, filename_col: str = 'filename', target_col: str = 'wc_mmolg') -> dict:
    df = pd.read_csv(csv_path, low_memory=False)
    if filename_col not in df.columns or target_col not in df.columns:
        raise RuntimeError(f"Targets CSV must contain columns: {filename_col} and {target_col}")
    mapping = {}
    for _, row in df.iterrows():
        name = ensure_cif_ext(str(row[filename_col]))
        try:
            val = float(row[target_col])
        except Exception:
            val = float('nan')
        mapping[name] = val
    return mapping


def find_missing_cifs(voxel_paths, cif_dir: Path):
    missing = []
    present = []
    for p in voxel_paths:
        cif_name = voxel_path_to_cif_name(Path(p))
        if not (Path(cif_dir) / cif_name).exists():
            missing.append(cif_name)
        else:
            present.append(cif_name)
    missing = sorted(list(set(missing)))
    present = sorted(list(set(present)))
    return present, missing


def _clean_state_dict(state_dict):
    
    if state_dict is None:
        return None
    if not isinstance(state_dict, dict):
        return state_dict
    new_state = {}
    for k, v in state_dict.items():
        newk = k[len('module.'):] if k.startswith('module.') else k
        new_state[newk] = v
    return new_state


def _safe_barrier(distributed, device):
   
    if not distributed:
        return
    try:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        dist.barrier()
    except Exception as e:
        print(f"[Warning] Barrier failed: {e}")


class VoxelDataset(Dataset):
    def __init__(self, files, targets=None):
        self.files = list(files)
        self.targets = targets or {}

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        f = self.files[i]
        suf = f.suffix.lower()
        try:
            if suf == ".pt":
                obj = torch.load(str(f), weights_only=False)
                if isinstance(obj, dict) and 'vox' in obj:
                    v = obj['vox']
                    vox = v.numpy() if torch.is_tensor(v) else np.array(v)
                elif torch.is_tensor(obj):
                    vox = obj.numpy()
                else:
                    raise RuntimeError(f"{f} is a .pt but missing 'vox' key or not a tensor")
            elif suf == ".npz":
                d = np.load(f, allow_pickle=True)
                if 'vox' in d:
                    vox = d['vox']
                else:
                    keys = [k for k in d.files]
                    vox = d[keys[0]]
            elif suf == ".npy":
                vox = np.load(f)
            else:
                raise RuntimeError(f"Unsupported file {f}")
        except Exception as e:
            raise RuntimeError(f"Failed to load voxel {f}: {e}")

        vox = np.array(vox, dtype=np.float32)
        if vox.ndim == 3:
            vox = vox[np.newaxis, ...]
        if vox.ndim != 4:
            raise RuntimeError(f"Voxel array must be shape (C,G,G,G) or (G,G,G). Got {vox.shape}")

        cif_name = voxel_path_to_cif_name(Path(f))
        target = float(self.targets.get(cif_name, float('nan')))
        return torch.from_numpy(vox), torch.tensor(target, dtype=torch.float32), f.name



def patchify_batch(x, patch):
    B, C, G, _, _ = x.shape
    assert G % patch == 0, "G must be divisible by patch"
    n = G // patch
    x = x.view(B, C, n, patch, n, patch, n, patch)
    x = x.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous()
    patches = x.view(B, n*n*n, C * (patch ** 3))
    return patches


def random_masking(N, ratio, device):
    N_mask = int(N * ratio)
    perm = torch.randperm(N, device=device)
    mask_indices = perm[:N_mask]
    keep_indices = perm[N_mask:]
    mask_indices, _ = torch.sort(mask_indices)
    keep_indices, _ = torch.sort(keep_indices)
    return mask_indices.long(), keep_indices.long()


def mae_loss_on_masked(patches, pred, mask_indices):
    if not isinstance(mask_indices, torch.Tensor):
        mask_indices = torch.tensor(mask_indices, device=patches.device, dtype=torch.long)
    target = patches.index_select(1, mask_indices)
    pred_m = pred.index_select(1, mask_indices)
    loss = ((pred_m - target) ** 2).mean()
    return loss



class PatchEmbed(nn.Module):
    def __init__(self, patch_dim, embed_dim):
        super().__init__()
        self.proj = nn.Linear(patch_dim, embed_dim)
    def forward(self, x): return self.proj(x)

class Transformer(nn.Module):
    def __init__(self, embed_dim, depth, heads):
        super().__init__()
        layer = nn.TransformerEncoderLayer(embed_dim, heads, 4*embed_dim, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, depth)
    def forward(self, x): return self.enc(x)

class MAE3D(nn.Module):
    def __init__(self, patch_dim, enc_embed, enc_depth, enc_heads,
                 dec_embed, dec_depth, dec_heads, mask_ratio):
        super().__init__()
        self.enc_embed = enc_embed
        self.patch_embed = PatchEmbed(patch_dim, enc_embed)
        self.encoder = Transformer(enc_embed, enc_depth, enc_heads)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dec_embed))
        self.enc_to_dec = nn.Linear(enc_embed, dec_embed)
        self.pos_embed_enc = None
        self.pos_embed_dec = None
        self.decoder = Transformer(dec_embed, dec_depth, dec_heads)
        self.dec_to_patch = nn.Linear(dec_embed, patch_dim)
        self.mask_ratio = mask_ratio
        self._init_weights()

    def _init_weights(self):
        with torch.no_grad():
            self.mask_token.normal_(mean=0., std=0.02)
            nn.init.xavier_uniform_(self.patch_embed.proj.weight)
            nn.init.xavier_uniform_(self.enc_to_dec.weight)
            nn.init.xavier_uniform_(self.dec_to_patch.weight)
            if self.patch_embed.proj.bias is not None:
                nn.init.zeros_(self.patch_embed.proj.bias)
            if self.enc_to_dec.bias is not None:
                nn.init.zeros_(self.enc_to_dec.bias)
            if self.dec_to_patch.bias is not None:
                nn.init.zeros_(self.dec_to_patch.bias)

    def init_pos_embeds(self, N, device):
        if (self.pos_embed_enc is None) or (self.pos_embed_enc.shape[1] != N) or (self.pos_embed_enc.device != device):
            p = torch.zeros(1, N, self.enc_embed, device=device)
            self.pos_embed_enc = nn.Parameter(p)
            with torch.no_grad():
                self.pos_embed_enc.normal_(mean=0., std=0.02)
        if (self.pos_embed_dec is None) or (self.pos_embed_dec.shape[1] != N) or (self.pos_embed_dec.device != device):
            p = torch.zeros(1, N, self.enc_to_dec.out_features, device=device)
            self.pos_embed_dec = nn.Parameter(p)
            with torch.no_grad():
                self.pos_embed_dec.normal_(mean=0., std=0.02)

    def encode(self, patches):
        x = self.patch_embed(patches)
        if self.pos_embed_enc is not None:
            x = x + self.pos_embed_enc[:, :x.shape[1], :].type_as(x)
        enc_out = self.encoder(x)
        pooled = enc_out.mean(dim=1)
        return pooled

    def forward(self, patches, mask_indices, keep_indices):
        B, N, D = patches.shape
        device = patches.device
        if (self.pos_embed_enc is None) or (self.pos_embed_enc.shape[1] != N) or (self.pos_embed_enc.device != device):
            self.init_pos_embeds(N, device)

        x_vis = patches.index_select(1, keep_indices)
        x_vis = self.patch_embed(x_vis)
        pos_vis = self.pos_embed_enc[0].index_select(0, keep_indices).unsqueeze(0).type_as(x_vis)
        x_vis = x_vis + pos_vis
        enc_out = self.encoder(x_vis)
        dec_vis = self.enc_to_dec(enc_out)
        dec_tokens = self.mask_token.expand(B, N, -1).clone().type_as(dec_vis)
        dec_tokens.scatter_(1,
                             keep_indices.view(1, -1, 1).expand(B, keep_indices.shape[0], dec_vis.shape[2]),
                             dec_vis)
        dec_tokens = dec_tokens + self.pos_embed_dec.type_as(dec_tokens)
        dec_out = self.decoder(dec_tokens)
        pred_patches = self.dec_to_patch(dec_out)
        return pred_patches


def evaluate_and_save_csv(model, loader, device, args, reg_head=None, distributed=False, output_csv=None):
   
    model_eval = (model.module if (distributed and hasattr(model, 'module')) else model)
    model_eval.eval()
    if reg_head is not None:
        reg_eval = (reg_head.module if (distributed and hasattr(reg_head, 'module')) else reg_head)
        reg_eval.eval()
    else:
        reg_eval = None

    local_weighted_loss = 0.0
    local_count = 0
    loss_fn = nn.L1Loss(reduction='mean') if args.reg_loss == 'l1' else nn.MSELoss(reduction='mean')

    
    all_filenames = []
    all_predictions = []
    all_targets = []
    all_valid_flags = []

    with torch.no_grad():
        for vox_batch, target_batch, names in loader:
            if not isinstance(vox_batch, torch.Tensor):
                vox_batch = torch.stack(vox_batch, dim=0)
            vox_batch = vox_batch.to(device, non_blocking=True)
            target_batch = target_batch.to(device, non_blocking=True).float()
            patches = patchify_batch(vox_batch, args.patch)

            if reg_eval is not None:
                feats = model_eval.encode(patches)
                preds = reg_eval(feats).view(-1)
                
                if args.normalize_target and hasattr(args, '_target_mean'):
                    targets_norm = (target_batch - args._target_mean) / args._target_std
                else:
                    targets_norm = target_batch
                
                valid_mask = ~torch.isnan(targets_norm)
                n_valid = int(valid_mask.sum().item())
                
                if n_valid > 0:
                    loss = loss_fn(preds[valid_mask], targets_norm[valid_mask])
                    local_weighted_loss += float(loss.item()) * n_valid
                    local_count += n_valid
                
                
                if args.normalize_target and hasattr(args, '_target_mean'):
                    preds_denorm = preds * args._target_std + args._target_mean
                else:
                    preds_denorm = preds
                
               
                preds_cpu = preds_denorm.cpu().numpy()
                targets_cpu = target_batch.cpu().numpy()
                valid_cpu = valid_mask.cpu().numpy()
                
               
                all_filenames.extend(names)
                all_predictions.extend(preds_cpu.tolist())
                all_targets.extend(targets_cpu.tolist())
                all_valid_flags.extend(valid_cpu.tolist())
            else:
                mask_idx, keep_idx = random_masking(patches.shape[1], args.mask_ratio, device)
                pred = model_eval(patches, mask_idx, keep_idx)
                n_masked = int(mask_idx.numel())
                loss = mae_loss_on_masked(patches, pred, mask_idx)
                local_weighted_loss += float(loss.item()) * n_masked
                local_count += n_masked

    if distributed:
        tensor = torch.tensor([local_weighted_loss, local_count], device=device, dtype=torch.float64)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        total_weighted_loss = float(tensor[0].item())
        total_count = int(tensor[1].item())
    else:
        total_weighted_loss = local_weighted_loss
        total_count = local_count

    eval_loss = float('nan') if total_count == 0 else (total_weighted_loss / total_count)
    
    
    is_main = (not distributed) or (dist.get_rank() == 0)
    if output_csv is not None and is_main and len(all_filenames) > 0:
        df = pd.DataFrame({
            'filename': all_filenames,
            'prediction': all_predictions,
            'target': all_targets,
            'is_valid': all_valid_flags
        })
        df.to_csv(output_csv, index=False)
        print(f"[CSV] Saved {len(df)} predictions to {output_csv}")
    
    return eval_loss


def evaluate(model, loader, device, args, reg_head=None, distributed=False):
    
    return evaluate_and_save_csv(model, loader, device, args, reg_head, distributed, output_csv=None)



def load_checkpoint_if_requested(args, model, opt, sched, scaler, reg_head=None, device=None, distributed=False, is_main=True):
   
    start_epoch = 1
    best_loss_local = float("inf")
    patience_local = 0
    bad_batches_local = []

    if args.resume is None:
        return start_epoch, best_loss_local, patience_local, bad_batches_local

    resume_path = Path(args.resume)
    if not resume_path.exists():
        if is_main:
            print(f"Warning: resume checkpoint {resume_path} not found. Starting fresh.")
        return start_epoch, best_loss_local, patience_local, bad_batches_local

    if is_main:
        print(f"Loading checkpoint from: {resume_path}")
    resume_ckpt = torch.load(str(resume_path), map_location=(device if device is not None else "cpu"), weights_only=False)

    resume_dir = resume_path.parent
    best_candidate = resume_dir / "mae_best.pt"
    if resume_path.name == "mae_best.pt":
        best_candidate = resume_path

    reset_patience_requested = getattr(args, "reset_patience", False) and getattr(args, "finetune", False)

    def _get_model_state_from_ckpt(ckpt):
        return ckpt.get('model_state') or ckpt.get('state_dict') or ckpt.get('model')

    if reset_patience_requested:
        if best_candidate.exists():
            if is_main:
                print(f"reset-patience: loading weights from best: {best_candidate}")
            load_ckpt = torch.load(str(best_candidate), map_location=(device if device is not None else "cpu"), weights_only=False)
        else:
            if is_main:
                print(f"reset-patience: {best_candidate} not found, using resume weights")
            load_ckpt = resume_ckpt

        model_state = _get_model_state_from_ckpt(load_ckpt)
        reg_head_state = load_ckpt.get('reg_head_state', None)

        if model_state is not None:
            try:
                if distributed and hasattr(model, 'module'):
                    try:
                        model.module.load_state_dict(model_state)
                    except Exception:
                        model.module.load_state_dict(_clean_state_dict(model_state))
                else:
                    try:
                        model.load_state_dict(model_state)
                    except Exception:
                        model.load_state_dict(_clean_state_dict(model_state))
            except Exception as e:
                if is_main:
                    print("Warning: failed to load model_state:", e)

        if reg_head is not None and reg_head_state is not None:
            try:
                if distributed and hasattr(reg_head, 'module'):
                    try:
                        reg_head.module.load_state_dict(reg_head_state)
                    except Exception:
                        reg_head.module.load_state_dict(_clean_state_dict(reg_head_state))
                else:
                    try:
                        reg_head.load_state_dict(reg_head_state)
                    except Exception:
                        reg_head.load_state_dict(_clean_state_dict(reg_head_state))
            except Exception as e:
                if is_main:
                    print("Warning: failed to load reg_head_state:", e)

        start_epoch = int(resume_ckpt.get('epoch', 0)) + 1
        bad_batches_local = resume_ckpt.get('bad_batches', [])

        if is_main:
            print(f"Resuming at epoch {start_epoch} with fresh early-stopping (best_loss=inf, patience=0)")
        return start_epoch, best_loss_local, patience_local, bad_batches_local

   
    if is_main:
        print(f"Normal resume from: {resume_path}")

    ckpt = resume_ckpt
    model_state = _get_model_state_from_ckpt(ckpt)
    reg_head_state = ckpt.get('reg_head_state', None)

    if model_state is not None:
        try:
            if distributed and hasattr(model, 'module'):
                try:
                    model.module.load_state_dict(model_state)
                except Exception:
                    model.module.load_state_dict(_clean_state_dict(model_state))
            else:
                try:
                    model.load_state_dict(model_state)
                except Exception:
                    model.load_state_dict(_clean_state_dict(model_state))
        except Exception as e:
            if is_main:
                print("Warning: failed to load model_state:", e)

    if reg_head is not None and reg_head_state is not None:
        try:
            if distributed and hasattr(reg_head, 'module'):
                try:
                    reg_head.module.load_state_dict(reg_head_state)
                except Exception:
                    reg_head.module.load_state_dict(_clean_state_dict(reg_head_state))
            else:
                try:
                    reg_head.load_state_dict(reg_head_state)
                except Exception:
                    reg_head.load_state_dict(_clean_state_dict(reg_head_state))
        except Exception as e:
            if is_main:
                print("Warning: failed to load reg_head_state:", e)

    if 'opt_state' in ckpt and opt is not None:
        try:
            opt.load_state_dict(ckpt['opt_state'])
        except Exception as e:
            if is_main:
                print("Warning: failed to load optimizer state:", e)

    if 'sched_state' in ckpt and sched is not None and ckpt.get('sched_state') is not None:
        try:
            sched.load_state_dict(ckpt['sched_state'])
        except Exception as e:
            if is_main:
                print("Warning: failed to load scheduler state:", e)

    if 'scaler_state' in ckpt and scaler is not None and ckpt.get('scaler_state') is not None:
        try:
            scaler.load_state_dict(ckpt['scaler_state'])
        except Exception as e:
            if is_main:
                print("Warning: failed to load scaler state:", e)

    start_epoch = int(ckpt.get('epoch', 0)) + 1
    best_loss_local = float(ckpt.get('best_loss', float("inf")))
    patience_local = int(ckpt.get('patience_cnt', 0))
    bad_batches_local = ckpt.get('bad_batches', [])

    if is_main:
        print(f"Resumed at epoch {start_epoch}, best_loss={best_loss_local:.6f}, patience={patience_local}")
    return start_epoch, best_loss_local, patience_local, bad_batches_local



def train(args):
    
    distributed = False
    device = None
    is_main = True
    local_rank = 0
    
    if args.distributed:
        distributed = True
        timeout_seconds = int(getattr(args, "dist_timeout_seconds", 3600))
        if is_main:
            print(f"[DDP] Setting timeout to {timeout_seconds}s ({timeout_seconds//60} minutes)")
        
        dist.init_process_group(
            backend="nccl",
            timeout=timedelta(seconds=timeout_seconds)
        )
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        is_main = (dist.get_rank() == 0)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if is_main:
        print(f"[Device] Using {device}")

    set_seed(args.seed)

  
    vox_dir = Path(args.vox_dir)
    if not vox_dir.exists():
        raise RuntimeError(f"vox-dir {vox_dir} does not exist")

    raw_files = sorted(
        list(vox_dir.glob("*_vox.npz")) +
        list(vox_dir.glob("*_vox.pt")) +
        list(vox_dir.glob("*_vox.npy")) +
        list(vox_dir.glob("*.npy")) +
        list(vox_dir.glob("*.pt")) +
        list(vox_dir.glob("*.npz"))
    )
    if len(raw_files) == 0:
        raise RuntimeError(f"No voxel files found in {vox_dir}")

    if is_main:
        print(f"[Dataset] Found {len(raw_files)} voxel files")

    
    targets = None
    if args.targets_csv is not None:
        if is_main:
            print(f"[Dataset] Loading targets from {args.targets_csv}")
        targets = load_targets_csv(args.targets_csv, filename_col=args.filename_col, target_col=args.target_col)

    
    rng = np.random.RandomState(args.seed)
    file_list = np.array(raw_files)
    idxs = np.arange(len(file_list))
    rng.shuffle(idxs)
    n = len(idxs)
    n_train = int(math.floor(0.8 * n))
    n_val = int(math.floor(0.1 * n))
    n_test = n - n_train - n_val
    
    train_idx = idxs[:n_train]
    val_idx = idxs[n_train:n_train + n_val]
    test_idx = idxs[n_train + n_val:]

    train_files = [file_list[i] for i in train_idx]
    val_files = [file_list[i] for i in val_idx]
    test_files = [file_list[i] for i in test_idx]

    if is_main:
        print(f"[Split] train={len(train_files)}, val={len(val_files)}, test={len(test_files)}")

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

   
    train_ds = VoxelDataset(train_files, targets=targets)
    val_ds = VoxelDataset(val_files, targets=targets) if len(val_files) > 0 else None
    test_ds = VoxelDataset(test_files, targets=targets) if len(test_files) > 0 else None

    train_dl_kwargs = dict(
        batch_size=args.batch_size,
        shuffle=(not distributed),
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    eval_dl_kwargs = dict(
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=max(1, args.num_workers),
        pin_memory=True,
        drop_last=False
    )
    if args.num_workers > 0:
        train_dl_kwargs['persistent_workers'] = True
        eval_dl_kwargs['persistent_workers'] = True

    train_sampler = DistributedSampler(train_ds) if distributed else None
    val_sampler = DistributedSampler(val_ds, shuffle=False) if (distributed and val_ds is not None) else None
    test_sampler = DistributedSampler(test_ds, shuffle=False) if (distributed and test_ds is not None) else None

    if train_sampler is not None:
        train_dl_kwargs['sampler'] = train_sampler
        train_dl_kwargs.pop('shuffle', None)
    if val_sampler is not None:
        eval_dl_kwargs['sampler'] = val_sampler

    eval_dl_kwargs_test = dict(eval_dl_kwargs)
    if test_sampler is not None:
        eval_dl_kwargs_test['sampler'] = test_sampler

    train_loader = DataLoader(train_ds, **train_dl_kwargs)
    val_loader = DataLoader(val_ds, **eval_dl_kwargs) if val_ds is not None else None
    test_loader = DataLoader(test_ds, **eval_dl_kwargs_test) if test_ds is not None else None

   
    sample_vox, sample_t, sample_name = train_ds[0]
    C, G, _, _ = sample_vox.shape
    assert G % args.patch == 0
    N = (G // args.patch) ** 3
    patch_dim = C * (args.patch ** 3)
    if is_main:
        print(f"[Model] Grid {G}^3, channels {C}, patch_dim {patch_dim}, num_patches {N}")


    model = MAE3D(
        patch_dim=patch_dim,
        enc_embed=args.enc_embed,
        enc_depth=args.enc_depth,
        enc_heads=args.enc_heads,
        dec_embed=args.dec_embed,
        dec_depth=args.dec_depth,
        dec_heads=args.dec_heads,
        mask_ratio=args.mask_ratio
    ).to(device)

    model.init_pos_embeds(N, device)

    reg_head = None
    if args.finetune:
        reg_head = nn.Sequential(
            nn.Linear(model.enc_embed, args.ft_hidden),
            nn.ReLU(),
            nn.Linear(args.ft_hidden, 1)
        ).to(device)

 
    if distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
        if reg_head is not None:
            reg_head = DDP(reg_head, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

   
    if args.freeze_encoder_epochs > 0 and args.finetune:
        if is_main:
            print(f"[Training] Freezing encoder for first {args.freeze_encoder_epochs} epochs")
        m = model.module if (distributed and hasattr(model, 'module')) else model
        for p_ in m.encoder.parameters():
            p_.requires_grad = False

  
    if args.finetune and reg_head is not None:
        if is_main:
            print("[Optimizer] Fine-tuning with separate LRs for encoder and head")
        
        m = model.module if (distributed and hasattr(model, 'module')) else model
        rh = reg_head.module if (distributed and hasattr(reg_head, 'module')) else reg_head

        encoder_params = [p for p in m.encoder.parameters() if p.requires_grad]
        head_params = [p for p in rh.parameters() if p.requires_grad]
        other_model_params = [p for name, p in m.named_parameters() if not name.startswith('encoder.') and p.requires_grad]

        lr_enc = args.lr * 0.1

        param_groups = [
            {'params': head_params, 'lr': args.lr},
            {'params': encoder_params, 'lr': lr_enc},
            {'params': other_model_params, 'lr': lr_enc}
        ]
        opt = torch.optim.AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)
    else:
        params = list(model.parameters()) + (list(reg_head.parameters()) if reg_head is not None else [])
        params = [p for p in params if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

  
    scaler = torch.amp.GradScaler("cuda" if device.type == "cuda" else "cpu", enabled=(device.type == "cuda"))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)


    target_mean = 0.0
    target_std = 1.0
    if args.finetune and args.normalize_target and (targets is not None):
        all_vals = [v for v in targets.values() if not math.isnan(v)]
        if len(all_vals) > 0:
            arr = np.array(all_vals, dtype=float)
            target_mean = float(arr.mean())
            target_std = float(arr.std() if arr.std() > 0 else 1.0)
            args._target_mean = target_mean
            args._target_std = target_std
            if is_main:
                print(f"[Targets] Normalization: mean={target_mean:.4g}, std={target_std:.4g}")

    loss_fn = nn.L1Loss() if args.reg_loss == 'l1' else nn.MSELoss()

   
    os.makedirs(args.out_dir, exist_ok=True)
    start_epoch, best_loss, patience_cnt, bad_batches = load_checkpoint_if_requested(
        args, model, opt, sched, scaler, reg_head=reg_head, device=device, 
        distributed=distributed, is_main=is_main
    )

    if args.epochs_per_run and args.epochs_per_run > 0:
        end_epoch = min(args.epochs, start_epoch + args.epochs_per_run - 1)
    else:
        end_epoch = args.epochs

    if is_main:
        print(f"[Training] Epochs: {start_epoch} -> {end_epoch}")

    try:
   
        for epoch in range(start_epoch, end_epoch + 1):
            
            if args.freeze_encoder_epochs > 0 and epoch == args.freeze_encoder_epochs + 1 and args.finetune:
                if is_main:
                    print(f"[Epoch {epoch}] Unfreezing encoder")
                m = model.module if (distributed and hasattr(model, 'module')) else model
                for p_ in m.encoder.parameters():
                    p_.requires_grad = True
                
                rh = reg_head.module if (distributed and hasattr(reg_head, 'module')) else reg_head
                encoder_params = [p for p in m.encoder.parameters() if p.requires_grad]
                head_params = [p for p in rh.parameters() if p.requires_grad]
                other_model_params = [p for name, p in m.named_parameters() if not name.startswith('encoder.') and p.requires_grad]
                lr_enc = args.lr * 0.1
                param_groups = [
                    {'params': head_params, 'lr': args.lr},
                    {'params': encoder_params, 'lr': lr_enc},
                    {'params': other_model_params, 'lr': lr_enc}
                ]
                opt = torch.optim.AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)
                sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

            model.train()
            if reg_head is not None:
                reg_head.train()

            running_loss = 0.0
            steps = 0

            if distributed and train_sampler is not None:
                train_sampler.set_epoch(epoch)

            if is_main:
                pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{end_epoch}", ncols=100)
            else:
                pbar = train_loader

            for batch_idx, batch in enumerate(pbar):
                vox_batch, target_batch, names = batch
                if not isinstance(vox_batch, torch.Tensor):
                    vox_batch = torch.stack(vox_batch, dim=0)
                vox_batch = vox_batch.to(device, non_blocking=True)
                target_batch = target_batch.to(device, non_blocking=True).float()

                patches = patchify_batch(vox_batch, args.patch)

                opt.zero_grad()
                
               
                with torch.amp.autocast("cuda" if device.type == "cuda" else "cpu", enabled=(device.type == "cuda")):
                    if args.finetune:
                        feats = (model.module.encode(patches) if (distributed and hasattr(model, 'module')) else model.encode(patches))
                        preds = (reg_head.module(feats).view(-1) if (distributed and hasattr(reg_head, 'module')) else (reg_head(feats).view(-1) if reg_head is not None else None))
                        
                        if args.normalize_target:
                            targets_norm = (target_batch - args._target_mean) / args._target_std
                        else:
                            targets_norm = target_batch
                        
                        valid_mask = ~torch.isnan(targets_norm)
                        if valid_mask.sum() == 0:
                            raise RuntimeError("All targets are NaN; pre-filter dataset to avoid empty batches.")
                        loss = loss_fn(preds[valid_mask], targets_norm[valid_mask])
                    else:
                        mask_idx, keep_idx = random_masking(patches.shape[1], args.mask_ratio, device)
                        m = (model.module if (distributed and hasattr(model, 'module')) else model)
                        pred = m(patches, mask_idx, keep_idx)
                        loss = mae_loss_on_masked(patches, pred, mask_idx)

                scaler.scale(loss).backward()

                if args.max_grad_norm > 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    if reg_head is not None:
                        torch.nn.utils.clip_grad_norm_(reg_head.parameters(), args.max_grad_norm)

                scaler.step(opt)
                scaler.update()

                running_loss += float(loss.item())
                steps += 1
                if is_main:
                    pbar.set_postfix({'loss': running_loss / steps})

            epoch_loss = running_loss / max(1, steps)
            if is_main:
                print(f"Epoch {epoch} train loss: {epoch_loss:.6f}")

         
            val_loss = float('nan')
            if val_loader is not None:
                _safe_barrier(distributed, device)
                val_loss = evaluate(
                    (model.module if distributed and hasattr(model, 'module') else model),
                    val_loader, device, args,
                    reg_head=(reg_head.module if distributed and reg_head is not None and hasattr(reg_head, 'module') else reg_head),
                    distributed=distributed
                )
                _safe_barrier(distributed, device)
                if is_main:
                    print(f"Epoch {epoch} val loss: {val_loss:.6f}")

            try:
                sched.step()
            except Exception:
                pass

        
            deciding_loss = val_loss if (not math.isnan(val_loss)) else epoch_loss
            if is_main:
                ckpt_path = os.path.join(args.out_dir, f"mae_epoch{epoch}.pt")
                save_dict = {
                    'epoch': epoch,
                    'model_state': (model.module.state_dict() if distributed and hasattr(model, 'module') else model.state_dict()),
                    'opt_state': opt.state_dict(),
                    'sched_state': (sched.state_dict() if sched is not None else None),
                    'scaler_state': (scaler.state_dict() if scaler is not None else None),
                    'bad_batches': bad_batches,
                    'best_loss': best_loss,
                    'patience_cnt': patience_cnt
                }
                if reg_head is not None:
                    save_dict['reg_head_state'] = (reg_head.module.state_dict() if distributed and hasattr(reg_head, 'module') else reg_head.state_dict())
                torch.save(save_dict, ckpt_path)

            if deciding_loss < best_loss:
                best_loss = deciding_loss
                patience_cnt = 0
                if is_main:
                    best_path = os.path.join(args.out_dir, "mae_best.pt")
                    best_save = {
                        'epoch': epoch,
                        'model_state': (model.module.state_dict() if distributed and hasattr(model, 'module') else model.state_dict()),
                        'reg_head_state': (reg_head.module.state_dict() if distributed and reg_head is not None and hasattr(reg_head, 'module') else (reg_head.state_dict() if reg_head is not None else None)),
                        'opt_state': opt.state_dict(),
                        'sched_state': (sched.state_dict() if sched is not None else None),
                        'scaler_state': (scaler.state_dict() if scaler is not None else None),
                        'best_loss': best_loss,
                        'patience_cnt': patience_cnt,
                        'bad_batches': bad_batches
                    }
                    torch.save(best_save, best_path)
                    print(f"Saved best model to {best_path} (loss: {best_loss:.6f})")
            else:
                patience_cnt += 1
                if patience_cnt >= args.patience:
                    if is_main:
                        print(f"Early stopping: patience {args.patience} reached.")
                    break

       
        if test_loader is not None:
            _safe_barrier(distributed, device)
            
            best_path = Path(args.out_dir) / "mae_best.pt"
            fallback_path = Path(args.out_dir) / f"mae_epoch{end_epoch}.pt"

            if best_path.exists():
                ck = torch.load(str(best_path), map_location='cpu', weights_only=False)
                model_state = ck.get('model_state') or ck.get('state_dict')
                if is_main:
                    print(f"Loading best model for test: {best_path}")
            elif fallback_path.exists():
                ck = torch.load(str(fallback_path), map_location='cpu', weights_only=False)
                model_state = ck.get('model_state') or ck.get('state_dict')
                if is_main:
                    print(f"Loading fallback model for test: {fallback_path}")
            else:
                model_state = None
                if is_main:
                    print("No checkpoint found; using current model weights for test.")

            if model_state is not None:
                m = model.module if (distributed and hasattr(model, 'module')) else model
                try:
                    m.load_state_dict(model_state)
                except Exception:
                    m.load_state_dict(_clean_state_dict(model_state))

            if distributed and test_sampler is not None:
                try:
                    test_sampler.set_epoch(0)
                except Exception:
                    pass

            _safe_barrier(distributed, device)

            # Save predictions to CSV
            test_csv_path = os.path.join(args.out_dir, 'test_predictions.csv')
            test_loss = evaluate_and_save_csv(
                (model.module if distributed and hasattr(model, 'module') else model),
                test_loader, device, args,
                reg_head=(reg_head.module if distributed and reg_head is not None and hasattr(reg_head, 'module') else reg_head),
                distributed=distributed,
                output_csv=test_csv_path
            )

            _safe_barrier(distributed, device)

            if is_main:
                print(f"Final test loss: {test_loss:.6f}")
                print(f"Test predictions saved to: {test_csv_path}")

    except Exception as e:
        if is_main:
            print(f"[ERROR] Training failed: {e}")
            traceback.print_exc()
        raise
    finally:
        
        if distributed:
            try:
                _safe_barrier(distributed, device)
            except Exception:
                pass
            try:
                dist.destroy_process_group()
                if is_main:
                    print("[DDP] Process group destroyed")
            except Exception as e:
                print(f"[Warning] Failed to destroy process group: {e}")

    if is_main:
        print(f"[Done] Best loss: {best_loss:.6f}")



if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--vox-dir", required=True, help="folder with voxel files")
    p.add_argument("--out-dir", default="mae_runs", help="output directory for checkpoints")
    p.add_argument("--patch", type=int, required=True, help="patch size")
    p.add_argument("--epochs", type=int, default=200, help="total epochs to train")
    p.add_argument("--batch", dest='batch_size', type=int, default=8, help="batch size per process")
    p.add_argument("--num-workers", type=int, default=4, help="dataloader workers")
    p.add_argument("--lr", type=float, default=3e-4, help="learning rate")
    p.add_argument("--weight-decay", type=float, default=0.05, help="weight decay")
    p.add_argument("--mask-ratio", type=float, default=0.75, help="masking ratio")
    p.add_argument("--enc-embed", type=int, default=512, help="encoder embedding dim")
    p.add_argument("--enc-depth", type=int, default=8, help="encoder depth")
    p.add_argument("--enc-heads", type=int, default=8, help="encoder heads")
    p.add_argument("--dec-embed", type=int, default=256, help="decoder embedding dim")
    p.add_argument("--dec-depth", type=int, default=4, help="decoder depth")
    p.add_argument("--dec-heads", type=int, default=8, help="decoder heads")
    p.add_argument("--seed", type=int, default=42, help="random seed")
    

    p.add_argument("--finetune", action="store_true", help="enable fine-tuning mode")
    p.add_argument("--targets-csv", type=str, default=None, help="targets CSV file")
    p.add_argument("--filename-col", type=str, default='filename')
    p.add_argument("--target-col", type=str, default='wc_mmolg')
    p.add_argument("--normalize-target", action='store_true', help="normalize targets")
    p.add_argument("--reg-loss", choices=['l1','mse'], default='l1', help="regression loss")
    p.add_argument("--ft-hidden", type=int, default=256, help="regression head hidden dim")
    p.add_argument("--max-grad-norm", type=float, default=1.0, help="gradient clipping norm")
    p.add_argument("--freeze-encoder-epochs", type=int, default=5, help="freeze encoder epochs")
    

    p.add_argument("--match-cif-dir", type=str, default=None)
    p.add_argument("--fail-on-missing", action='store_true')
    

    p.add_argument("--patience", type=int, default=20, help="early stopping patience")
    p.add_argument("--distributed", action="store_true", help="enable DDP")
    p.add_argument("--dist-timeout-seconds", type=int, default=3600, help="NCCL timeout (secs)")
    p.add_argument("--reset-patience", action="store_true", help="reset early-stopping when resuming for fine-tune")
    

    p.add_argument("--resume", type=str, default=None, help="checkpoint to resume from")
    p.add_argument("--epochs-per-run", type=int, default=0, help="stop after this many epochs (0=no limit)")

    args = p.parse_args()
    args.batch_size = int(args.batch_size)
    args.patch = int(args.patch)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    train(args)


