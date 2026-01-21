# mof_voxelizer/io_utils.py
import json
import numpy as np
from pathlib import Path
from typing import List

def find_cif_files(directory: Path) -> List[Path]:
    if not directory.exists():
        return []
    return sorted([f for f in directory.iterdir() if f.is_file() and f.suffix.lower() == ".cif"])

def save_results(out_dir: Path, stem: str, vox: np.ndarray, channels: List[str], meta: dict, save_torch: bool = False):
    out_npz = out_dir / f"{stem}_vox.npz"
    out_meta = out_dir / f"{stem}_meta.json"
    
    # Save NPZ
    np.savez_compressed(str(out_npz), vox=vox, channels=np.array(channels, dtype=object))
    
    # Save Meta
    with open(str(out_meta), "w") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True)

    # Optional Torch Save
    if save_torch:
        try:
            import torch
            torch.save({"vox": torch.from_numpy(vox), "meta": meta}, str(out_dir / f"{stem}_vox.pt"))
        except ImportError:
            pass
