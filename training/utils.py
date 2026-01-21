import os
import torch
import torch.distributed as dist
import random
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import timedelta

def setup_distributed(args):
    """Initializes DDP if requested."""
    if args.distributed:
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            args.rank = int(os.environ["RANK"])
            args.world_size = int(os.environ["WORLD_SIZE"])
            args.gpu = int(os.environ["LOCAL_RANK"])
        else:
            print('Not using distributed mode')
            args.distributed = False
            return False, torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        torch.cuda.set_device(args.gpu)
        dist_backend = 'nccl'
        print(f'| distributed init (rank {args.rank}): {dist_backend}', flush=True)
        dist.init_process_group(
            backend=dist_backend, 
            init_method='env://',
            world_size=args.world_size, 
            rank=args.rank,
            timeout=timedelta(seconds=args.dist_timeout_seconds)
        )
        dist.barrier()
        return True, torch.device(f'cuda:{args.gpu}')
    else:
        return False, torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0

def ensure_cif_ext(name: str) -> str:
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

def load_targets_csv(csv_path: str, filename_col: str, target_col: str) -> dict:
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

def save_checkpoint(args, epoch, model, opt, sched, scaler, best_loss, patience_cnt, bad_batches, reg_head=None, is_best=False):
    if not is_main_process():
        return
    
    model_state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
    reg_state = None
    if reg_head is not None:
        reg_state = reg_head.module.state_dict() if hasattr(reg_head, 'module') else reg_head.state_dict()

    save_dict = {
        'epoch': epoch,
        'model_state': model_state,
        'reg_head_state': reg_state,
        'opt_state': opt.state_dict(),
        'sched_state': (sched.state_dict() if sched else None),
        'scaler_state': (scaler.state_dict() if scaler else None),
        'bad_batches': bad_batches,
        'best_loss': best_loss,
        'patience_cnt': patience_cnt
    }
    
    fname = "mae_best.pt" if is_best else f"mae_epoch{epoch}.pt"
    torch.save(save_dict, os.path.join(args.out_dir, fname))
    if is_best:
        print(f"Saved best model to {fname} (loss: {best_loss:.6f})")

def _clean_state_dict(state_dict):
    if state_dict is None: return None
    new_state = {}
    for k, v in state_dict.items():
        newk = k[len('module.'):] if k.startswith('module.') else k
        new_state[newk] = v
    return new_state
