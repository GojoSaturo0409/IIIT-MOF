"""
CNN baseline for voxel regression.
"""

import os
import math
import argparse
import random
import traceback
from pathlib import Path
from datetime import timedelta

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm


# ─────────────────────────────────────────────
# Hardware setup
# ─────────────────────────────────────────────

def configure_gpu():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


# ─────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─────────────────────────────────────────────
# Filename helpers
# ─────────────────────────────────────────────

def ensure_cif_ext(name: str) -> str:
    if not isinstance(name, str):
        name = str(name)
    return name if name.lower().endswith(".cif") else name + ".cif"


def voxel_path_to_cif_name(p: Path) -> str:
    name = p.name
    lower = name.lower()
    for s in ["_vox.npz", "_vox.pt", "_vox.npy", "_vox", ".npy", ".pt", ".npz"]:
        if lower.endswith(s):
            return ensure_cif_ext(name[: len(name) - len(s)])
    return ensure_cif_ext(p.stem)


# ─────────────────────────────────────────────
# CSV helpers
# ─────────────────────────────────────────────

def load_targets_csv(csv_path: str, filename_col: str, target_col: str) -> dict:
    df = pd.read_csv(csv_path, low_memory=False)
    if filename_col not in df.columns or target_col not in df.columns:
        raise RuntimeError(f"CSV must have columns '{filename_col}' and '{target_col}'")
    mapping = {}
    for _, row in df.iterrows():
        if pd.isna(row[filename_col]):
            continue
        key = ensure_cif_ext(str(row[filename_col]))
        try:
            val = float(row[target_col])
        except Exception:
            val = float("nan")
        mapping[key] = val
    return mapping


def load_name_set(csv_path: str, filename_col: str) -> set:
    df = pd.read_csv(csv_path, low_memory=False)
    if filename_col not in df.columns:
        raise RuntimeError(f"CSV must have column '{filename_col}'")
    names = set()
    for _, row in df.iterrows():
        if pd.isna(row[filename_col]):
            continue
        names.add(ensure_cif_ext(str(row[filename_col])))
    return names


# ─────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────

def unwrap_model(m):
    """Return the base model, removing DDP and compile wrappers if present."""
    if hasattr(m, "module"):
        m = m.module
    if hasattr(m, "_orig_mod"):
        m = m._orig_mod
    return m


def clean_state_dict(state_dict):
    """
    Remove common wrapper prefixes so checkpoints can be loaded across:
    - plain models
    - DDP-wrapped models
    - torch.compile() wrapped models
    """
    cleaned = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            k = k[len("_orig_mod.") :]
        if k.startswith("module."):
            k = k[len("module.") :]
        cleaned[k] = v
    return cleaned


# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────

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
                obj = torch.load(str(f), map_location="cpu", weights_only=False)
                if isinstance(obj, dict) and "vox" in obj:
                    v = obj["vox"]
                    vox = v.numpy() if torch.is_tensor(v) else np.array(v)
                elif torch.is_tensor(obj):
                    vox = obj.numpy()
                else:
                    raise RuntimeError(f"{f}: .pt missing 'vox' key or not a tensor")
            elif suf == ".npz":
                d = np.load(f, allow_pickle=True)
                vox = d["vox"] if "vox" in d else d[d.files[0]]
            elif suf == ".npy":
                vox = np.load(f)
            else:
                raise RuntimeError(f"Unsupported file type: {f}")
        except Exception as e:
            raise RuntimeError(f"Failed loading {f}: {e}")

        vox = np.array(vox, dtype=np.float32)
        if vox.ndim == 3:
            vox = vox[np.newaxis]
        if vox.ndim != 4:
            raise RuntimeError(f"Expected (C,G,G,G), got {vox.shape} for {f}")

        cif_name = voxel_path_to_cif_name(f)
        target = float(self.targets.get(cif_name, float("nan")))
        return torch.from_numpy(vox), torch.tensor(target, dtype=torch.float32), f.name


# ─────────────────────────────────────────────
# GPU Prefetcher
# ─────────────────────────────────────────────

class CUDAPrefetcher:
    def __init__(self, loader, device: torch.device):
        self.loader = loader
        self.device = device
        self.stream = torch.cuda.Stream(device=device) if device.type == "cuda" else None

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        self._iter = iter(self.loader)
        self._preload()
        return self

    def _preload(self):
        try:
            self._next = next(self._iter)
        except StopIteration:
            self._next = None
            return
        if self.stream is None:
            return
        with torch.cuda.stream(self.stream):
            vox, tgt, names = self._next
            self._next = (
                vox.to(self.device, non_blocking=True),
                tgt.to(self.device, non_blocking=True),
                names,
            )

    def __next__(self):
        if self.stream is not None:
            torch.cuda.current_stream(self.device).wait_stream(self.stream)
        batch = self._next
        if batch is None:
            raise StopIteration
        self._preload()
        return batch


# ─────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────

class SimpleVoxelCNN(nn.Module):
    def __init__(self, in_channels: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, 16, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm3d(16), nn.ReLU(inplace=True),

            nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(32), nn.ReLU(inplace=True),

            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(64), nn.ReLU(inplace=True),

            nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(128), nn.ReLU(inplace=True),
        )
        self.gap = nn.AdaptiveAvgPool3d(1)
        self.head = nn.Linear(128, 1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.head(self.gap(self.net(x)).flatten(1)).view(-1)


# ─────────────────────────────────────────────
# Distributed helpers
# ─────────────────────────────────────────────

def _safe_barrier(distributed, device):
    if not distributed:
        return
    try:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        dist.barrier()
    except Exception as e:
        print(f"[Warning] Barrier failed: {e}")


def _gather_objects(local_obj, distributed):
    if not distributed:
        return [local_obj]
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, local_obj)
    return gathered


# ─────────────────────────────────────────────
# Atomic checkpoint save / prune
# ─────────────────────────────────────────────

def _atomic_save(obj: dict, dst: Path):
    tmp = dst.with_suffix(".tmp")
    torch.save(obj, str(tmp))
    tmp.rename(dst)


def _prune_epoch_checkpoints(out_dir: Path, current_epoch: int, keep_last_n: int):
    if keep_last_n <= 0:
        return
    epoch_ckpts = sorted(out_dir.glob("cnn_epoch*.pt"), key=lambda p: p.stat().st_mtime)
    to_delete = epoch_ckpts[: max(0, len(epoch_ckpts) - keep_last_n)]
    for p in to_delete:
        try:
            p.unlink()
        except OSError:
            pass


# ─────────────────────────────────────────────
# Sync BN running stats across ranks manually
# (replaces DDP broadcast_buffers which we disabled)
# ─────────────────────────────────────────────

def _sync_bn_buffers(model, distributed):
    """All-reduce BatchNorm running_mean / running_var across ranks."""
    if not distributed:
        return
    world_size = dist.get_world_size()
    buffers = []
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            buffers.append(module.running_mean)
            buffers.append(module.running_var)
    if not buffers:
        return
    flat = torch.cat([b.flatten() for b in buffers])
    dist.all_reduce(flat, op=dist.ReduceOp.SUM)
    flat.div_(world_size)
    offset = 0
    for b in buffers:
        n = b.numel()
        b.copy_(flat[offset: offset + n].view_as(b))
        offset += n


# ─────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────

def evaluate(model, loader, device, args, distributed=False, output_csv=None):
    model_eval = model.module if (distributed and hasattr(model, "module")) else model
    model_eval.eval()

    # Sync BN stats before eval so all ranks see the same running stats
    _sync_bn_buffers(model_eval, distributed)

    loss_fn = nn.L1Loss(reduction="mean") if args.reg_loss == "l1" else nn.MSELoss(reduction="mean")
    local_weighted_loss = 0.0
    local_count = 0
    local_rows = []
    amp_enabled = device.type == "cuda"

    if amp_enabled:
        loader = CUDAPrefetcher(loader, device)

    with torch.no_grad():
        for vox, target, names in loader:
            if not amp_enabled:
                vox = vox.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
            target = target.float()
            vox = vox.to(memory_format=torch.channels_last_3d)

            with torch.amp.autocast("cuda" if amp_enabled else "cpu", enabled=amp_enabled):
                preds = model_eval(vox)

            if args.normalize_target and hasattr(args, "_target_mean"):
                targets_norm = (target - args._target_mean) / args._target_std
            else:
                targets_norm = target

            valid = ~torch.isnan(targets_norm)
            n_valid = int(valid.sum().item())
            if n_valid > 0:
                local_weighted_loss += float(loss_fn(preds[valid], targets_norm[valid]).item()) * n_valid
                local_count += n_valid

            if output_csv is not None:
                preds_out = (
                    preds * args._target_std + args._target_mean
                    if (args.normalize_target and hasattr(args, "_target_mean"))
                    else preds
                )
                for n, p, t, v in zip(
                    names,
                    preds_out.detach().cpu().tolist(),
                    target.detach().cpu().tolist(),
                    valid.cpu().tolist(),
                ):
                    local_rows.append(
                        {
                            "filename": n,
                            "prediction": float(p),
                            "target": float(t),
                            "is_valid": bool(v),
                        }
                    )

    # All ranks must participate in this all_reduce
    if distributed:
        dist.barrier()
        t = torch.tensor([local_weighted_loss, local_count], device=device, dtype=torch.float64)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        local_weighted_loss, local_count = float(t[0]), int(t[1])

    eval_loss = float("nan") if local_count == 0 else local_weighted_loss / local_count

    is_main = (not distributed) or (dist.get_rank() == 0)
    if output_csv is not None:
        gathered = _gather_objects(local_rows, distributed)
        if is_main:
            rows = [r for chunk in gathered if chunk for r in chunk]
            if rows:
                df = pd.DataFrame(rows).sort_values("filename").reset_index(drop=True)
                df.to_csv(output_csv, index=False)
                print(f"[CSV] Saved {len(df)} predictions → {output_csv}")

    return eval_loss


# ─────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────

def train(args):
    configure_gpu()

    # ── DDP setup ─────────────────────────────
    distributed = False
    local_rank = 0
    is_main = True

    if args.distributed:
        distributed = True
        dist.init_process_group(
            backend="nccl",
            timeout=timedelta(seconds=args.dist_timeout_seconds),
        )
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        is_main = dist.get_rank() == 0
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if is_main:
        print(f"[Device] {device}  |  distributed={distributed}")

    set_seed(args.seed)

    # ── Targets ───────────────────────────────
    targets = None
    if args.targets_csv:
        targets = load_targets_csv(args.targets_csv, args.filename_col, args.target_col)

    # ── Build split from CSV files ─────────────
    split_dir = Path(args.split_dir)
    train_csv_path = split_dir / "train.csv"
    val_csv_path = split_dir / "val.csv"
    test_csv_path = split_dir / "test.csv"

    for p in (train_csv_path, val_csv_path, test_csv_path):
        if not p.exists():
            raise RuntimeError(f"Split file not found: {p}")

    train_names = load_name_set(str(train_csv_path), args.filename_col)
    val_names = load_name_set(str(val_csv_path), args.filename_col)
    test_names = load_name_set(str(test_csv_path), args.filename_col)

    if is_main:
        print(f"[Split] CSV names — train={len(train_names)}  val={len(val_names)}  test={len(test_names)}")

    # ── Discover voxel files ───────────────────
    vox_dir = Path(args.vox_dir)
    if not vox_dir.exists():
        raise RuntimeError(f"--vox-dir {vox_dir} does not exist")

    raw_files = sorted(
        list(vox_dir.glob("*_vox.npz")) + list(vox_dir.glob("*_vox.pt")) +
        list(vox_dir.glob("*_vox.npy")) + list(vox_dir.glob("*.npy")) +
        list(vox_dir.glob("*.pt")) + list(vox_dir.glob("*.npz"))
    )
    if not raw_files:
        raise RuntimeError(f"No voxel files found in {vox_dir}")

    if is_main:
        print(f"[Dataset] Found {len(raw_files)} voxel files on disk")

    train_files, val_files, test_files, skipped = [], [], [], []
    for f in raw_files:
        key = voxel_path_to_cif_name(f)
        if key in train_names:
            train_files.append(f)
        elif key in val_names:
            val_files.append(f)
        elif key in test_names:
            test_files.append(f)
        else:
            skipped.append(f)

    if is_main:
        print(
            f"[Split] Matched — train={len(train_files)}  val={len(val_files)}  "
            f"test={len(test_files)}  skipped={len(skipped)}"
        )
        if not train_files:
            raise RuntimeError("No voxel files matched the train CSV. Check --filename-col and file naming.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds = VoxelDataset(train_files, targets=targets)
    val_ds = VoxelDataset(val_files, targets=targets) if val_files else None
    test_ds = VoxelDataset(test_files, targets=targets) if test_files else None

    # ── DataLoaders ────────────────────────────
    num_workers = args.num_workers
    dl_common = dict(
        pin_memory=(device.type == "cuda"),
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
        prefetch_factor=(4 if num_workers > 0 else None),
    )

    tr_sampler = DistributedSampler(train_ds) if distributed else None
    val_sampler = DistributedSampler(val_ds, shuffle=False) if (distributed and val_ds) else None
    test_sampler = DistributedSampler(test_ds, shuffle=False) if (distributed and test_ds) else None

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=tr_sampler,
        shuffle=(tr_sampler is None),
        drop_last=True,
        **dl_common,
    )
    val_loader = (
        DataLoader(val_ds, batch_size=args.batch_size, sampler=val_sampler, shuffle=False, drop_last=False, **dl_common)
        if val_ds
        else None
    )
    test_loader = (
        DataLoader(test_ds, batch_size=args.batch_size, sampler=test_sampler, shuffle=False, drop_last=False, **dl_common)
        if test_ds
        else None
    )

    # ── Model ──────────────────────────────────
    sample_vox, _, _ = train_ds[0]
    in_channels = sample_vox.shape[0]

    model = SimpleVoxelCNN(in_channels=in_channels)
    model = model.to(memory_format=torch.channels_last_3d).to(device)

    if is_main:
        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"[Model] SimpleVoxelCNN | params: {n_params:.2f} M")

    # NOTE: torch.compile intentionally removed — causes asymmetric NCCL
    # collectives with DDP + channels_last_3d (see cudagraph warnings in logs)

    if distributed:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
            broadcast_buffers=False,
            static_graph=False,
        )

    # ── Target normalisation (train split only) ─
    if args.normalize_target and targets is not None:
        train_vals = [
            targets[voxel_path_to_cif_name(f)]
            for f in train_files
            if not math.isnan(targets.get(voxel_path_to_cif_name(f), float("nan")))
        ]
        if not train_vals:
            raise RuntimeError("No valid training targets found for normalisation.")
        arr = np.array(train_vals, dtype=float)
        args._target_mean = float(arr.mean())
        args._target_std = float(arr.std()) if arr.std() > 0 else 1.0
        if is_main:
            print(f"[Targets] train mean={args._target_mean:.6g}  std={args._target_std:.6g}")

    # ── Optimizer: fused AdamW ─────────────────
    try:
        opt = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            fused=(device.type == "cuda"),
        )
        if is_main:
            print("[Optimizer] AdamW fused=True")
    except TypeError:
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        if is_main:
            print("[Optimizer] AdamW (fused unavailable)")

    # ── Scheduler ─────────────────────────────
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    # ── AMP ───────────────────────────────────
    amp_enabled = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda" if amp_enabled else "cpu", enabled=amp_enabled)

    loss_fn = nn.L1Loss() if args.reg_loss == "l1" else nn.MSELoss()

    # ── Resume ─────────────────────────────────
    start_epoch = 1
    best_loss = float("inf")
    patience_cnt = 0

    resume_path = None
    if args.resume and Path(args.resume).exists():
        resume_path = Path(args.resume)
    elif args.auto_resume:
        candidate = out_dir / "cnn_last.pt"
        if candidate.exists():
            resume_path = candidate
            if is_main:
                print(f"[Resume] Auto-detected checkpoint: {resume_path}")

    if resume_path is not None:
        if is_main:
            print(f"[Resume] Loading: {resume_path}")
        ck = torch.load(str(resume_path), map_location=device, weights_only=False)
        m = unwrap_model(model)
        m.load_state_dict(clean_state_dict(ck["model_state"]))
        opt.load_state_dict(ck["opt_state"])
        if ck.get("sched_state"):
            try:
                sched.load_state_dict(ck["sched_state"])
            except Exception:
                pass
        if ck.get("scaler_state"):
            try:
                scaler.load_state_dict(ck["scaler_state"])
            except Exception:
                pass
        start_epoch = int(ck.get("epoch", 0)) + 1
        best_loss = float(ck.get("best_loss", float("inf")))
        patience_cnt = int(ck.get("patience_cnt", 0))

        if ck.get("target_mean") is not None and args.normalize_target:
            args._target_mean = float(ck["target_mean"])
            args._target_std = float(ck["target_std"])
            if is_main:
                print(
                    f"[Resume] Restored normalisation — mean={args._target_mean:.6g}  "
                    f"std={args._target_std:.6g}"
                )

        if is_main:
            print(f"[Resume] epoch={start_epoch}  best_loss={best_loss:.6f}  patience={patience_cnt}")

    end_epoch = (
        args.epochs if not args.epochs_per_run
        else min(args.epochs, start_epoch + args.epochs_per_run - 1)
    )
    if is_main:
        print(f"[Training] epochs {start_epoch} → {end_epoch}")

    # ── Helper: build checkpoint dict ─────────
    def _make_checkpoint(epoch: int) -> dict:
        m_state = clean_state_dict(unwrap_model(model).state_dict())
        ck = {
            "epoch": epoch,
            "model_state": m_state,
            "opt_state": opt.state_dict(),
            "sched_state": sched.state_dict(),
            "scaler_state": scaler.state_dict(),
            "best_loss": best_loss,
            "patience_cnt": patience_cnt,
            "target_mean": getattr(args, "_target_mean", None),
            "target_std": getattr(args, "_target_std", None),
        }
        return ck

    # ── Training loop ─────────────────────────
    try:
        for epoch in range(start_epoch, end_epoch + 1):
            model.train()
            if distributed and tr_sampler is not None:
                tr_sampler.set_epoch(epoch)

            run_loss, steps = 0.0, 0

            iterable = CUDAPrefetcher(train_loader, device) if amp_enabled else train_loader
            pbar = tqdm(iterable, desc=f"Epoch {epoch}/{end_epoch}", ncols=100) if is_main else iterable

            for vox, target, _ in pbar:
                if not amp_enabled:
                    vox = vox.to(device, non_blocking=True)
                    target = target.to(device, non_blocking=True)
                target = target.float()
                vox = vox.to(memory_format=torch.channels_last_3d)

                opt.zero_grad(set_to_none=True)

                with torch.amp.autocast("cuda" if amp_enabled else "cpu", enabled=amp_enabled):
                    preds = model(vox)

                    if args.normalize_target and hasattr(args, "_target_mean"):
                        targets_norm = (target - args._target_mean) / args._target_std
                    else:
                        targets_norm = target

                    valid = ~torch.isnan(targets_norm)
                    if valid.sum() == 0:
                        continue

                    loss = loss_fn(preds[valid], targets_norm[valid])

                scaler.scale(loss).backward()

                if args.max_grad_norm > 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                scaler.step(opt)
                scaler.update()

                run_loss += float(loss.item())
                steps += 1
                if is_main:
                    pbar.set_postfix({"loss": f"{run_loss / steps:.5f}"})

            epoch_loss = run_loss / max(steps, 1)
            if is_main:
                print(f"Epoch {epoch} train loss: {epoch_loss:.6f}")

            # ── Validation ────────────────────
            val_loss = float("nan")
            if val_loader is not None:
                _safe_barrier(distributed, device)
                val_loss = evaluate(model, val_loader, device, args, distributed)
                _safe_barrier(distributed, device)
                if is_main:
                    print(f"Epoch {epoch} val loss:   {val_loss:.6f}")

            try:
                sched.step()
            except Exception:
                pass

            deciding = val_loss if not math.isnan(val_loss) else epoch_loss

            if deciding < best_loss:
                best_loss = deciding
                patience_cnt = 0
                improved = True
            else:
                patience_cnt += 1
                improved = False

            # ── Save checkpoints (main rank only) ─
            if is_main:
                ck = _make_checkpoint(epoch)

                epoch_path = out_dir / f"cnn_epoch{epoch}.pt"
                _atomic_save(ck, epoch_path)
                _prune_epoch_checkpoints(out_dir, epoch, args.keep_last_n)

                _atomic_save(ck, out_dir / "cnn_last.pt")

                if improved:
                    _atomic_save(ck, out_dir / "cnn_best.pt")
                    print(f"  → New best: {best_loss:.6f}  [cnn_best.pt updated]")

            if patience_cnt >= args.patience:
                if is_main:
                    print(f"Early stopping at epoch {epoch}  (patience={args.patience}).")
                break

        # ── Test ───────────────────────────────
        if test_loader is not None:
            _safe_barrier(distributed, device)

            best_ckpt = out_dir / "cnn_best.pt"
            last_ckpt = out_dir / "cnn_last.pt"
            fallback_ckpt = out_dir / f"cnn_epoch{end_epoch}.pt"

            loaded_ck = None
            for candidate, label in [
                (best_ckpt, "best"),
                (last_ckpt, "last"),
                (fallback_ckpt, f"epoch{end_epoch}"),
            ]:
                if candidate.exists():
                    loaded_ck = torch.load(str(candidate), map_location="cpu", weights_only=False)
                    if is_main:
                        print(f"[Test] Loaded {label} model: {candidate}")
                    break

            if loaded_ck is None and is_main:
                print("[Test] No checkpoint found; using current in-memory model.")

            if loaded_ck is not None:
                m = unwrap_model(model)
                try:
                    m.load_state_dict(clean_state_dict(loaded_ck["model_state"]))
                    if loaded_ck.get("target_mean") is not None and args.normalize_target:
                        args._target_mean = float(loaded_ck["target_mean"])
                        args._target_std = float(loaded_ck["target_std"])
                except Exception as e:
                    if is_main:
                        print(f"[Warning] Could not restore model state for test: {e}")

            _safe_barrier(distributed, device)

            out_csv = str(out_dir / "test_predictions.csv")
            test_loss = evaluate(model, test_loader, device, args, distributed=distributed, output_csv=out_csv)
            _safe_barrier(distributed, device)

            if is_main:
                print(f"Final test loss: {test_loss:.6f}")
                print(f"Test predictions saved to: {out_csv}")

    except Exception as e:
        if is_main:
            print(f"[ERROR] {e}")
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
            except Exception as ex:
                print(f"[Warning] Failed to destroy process group: {ex}")

    if is_main:
        print(f"[Done] Best loss: {best_loss:.6f}")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Paths
    p.add_argument("--vox-dir", required=True)
    p.add_argument("--out-dir", default="cnn_runs")
    p.add_argument("--split-dir", required=True)
    p.add_argument("--targets-csv", default=None)
    p.add_argument("--filename-col", default="filename")
    p.add_argument("--target-col", default="wc_mmolg")

    # Training
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--epochs-per-run", type=int, default=0)
    p.add_argument("--batch", dest="batch_size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--reg-loss", choices=["l1", "mse"], default="l1")
    p.add_argument("--normalize-target", action="store_true")
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)

    # Checkpointing
    p.add_argument("--resume", default=None)
    p.add_argument("--auto-resume", action="store_true")
    p.add_argument("--keep-last-n", type=int, default=3)

    # DDP
    p.add_argument("--distributed", action="store_true")
    p.add_argument("--dist-timeout-seconds", type=int, default=14400)  # 4 hours

    args = p.parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    train(args) 

