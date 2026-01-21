import math
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from utils import voxel_path_to_cif_name

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
                    raise RuntimeError(f"{f} is .pt but missing 'vox' key")
            elif suf == ".npz":
                d = np.load(f, allow_pickle=True)
                vox = d['vox'] if 'vox' in d else d[list(d.files)[0]]
            elif suf == ".npy":
                vox = np.load(f)
            else:
                raise RuntimeError(f"Unsupported file {f}")
        except Exception as e:
            raise RuntimeError(f"Failed to load voxel {f}: {e}")

        vox = np.array(vox, dtype=np.float32)
        if vox.ndim == 3: vox = vox[np.newaxis, ...]
        
        cif_name = voxel_path_to_cif_name(Path(f))
        target = float(self.targets.get(cif_name, float('nan')))
        return torch.from_numpy(vox), torch.tensor(target, dtype=torch.float32), f.name

def create_dataloaders(args, targets=None, is_main=True):
    vox_dir = Path(args.vox_dir)
    extensions = ["*_vox.npz", "*_vox.pt", "*_vox.npy", "*.npy", "*.pt", "*.npz"]
    raw_files = []
    for ext in extensions:
        raw_files.extend(list(vox_dir.glob(ext)))
    
    # Deduplicate and sort
    raw_files = sorted(list(set(raw_files)))

    if len(raw_files) == 0:
        raise RuntimeError(f"No voxel files found in {vox_dir}")
    if is_main:
        print(f"[Dataset] Found {len(raw_files)} voxel files")

    # Split
    rng = np.random.RandomState(args.seed)
    idxs = np.arange(len(raw_files))
    rng.shuffle(idxs)
    
    n = len(idxs)
    n_train = int(math.floor(0.8 * n))
    n_val = int(math.floor(0.1 * n))
    
    train_files = [raw_files[i] for i in idxs[:n_train]]
    val_files = [raw_files[i] for i in idxs[n_train:n_train + n_val]]
    test_files = [raw_files[i] for i in idxs[n_train + n_val:]]

    train_ds = VoxelDataset(train_files, targets)
    val_ds = VoxelDataset(val_files, targets) if val_files else None
    test_ds = VoxelDataset(test_files, targets) if test_files else None

    # Loaders
    train_sampler = DistributedSampler(train_ds) if args.distributed else None
    val_sampler = DistributedSampler(val_ds, shuffle=False) if (args.distributed and val_ds) else None
    test_sampler = DistributedSampler(test_ds, shuffle=False) if (args.distributed and test_ds) else None

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, 
        shuffle=(train_sampler is None), sampler=train_sampler,
        num_workers=args.num_workers, pin_memory=True, drop_last=True
    )
    
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, 
        sampler=val_sampler, num_workers=max(1, args.num_workers), 
        pin_memory=True
    ) if val_ds else None

    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, 
        sampler=test_sampler, num_workers=max(1, args.num_workers), 
        pin_memory=True
    ) if test_ds else None

    return train_loader, val_loader, test_loader, train_sampler
