import numpy as np
import pandas as pd
import shutil
import subprocess
import sys
import tempfile
import logging
from pathlib import Path
from typing import Dict, List, Optional

def find_cif(filename: str, cif_root: Path, repeat_subdir: str = "repeat_cifs") -> Optional[Path]:
    p = Path(filename)
    if p.exists() and p.suffix == '.cif': return p
    
    # Strip voxel suffixes if present
    stem = p.name.split('_vox')[0].replace('.cif', '')
    
    search_paths = [cif_root / repeat_subdir, cif_root]
    for root in search_paths:
        if not root.exists(): continue
        # Exact match
        found = list(root.rglob(f"{stem}.cif"))
        if found: return found[0]
        # Partial match
        for f in root.rglob("*.cif"):
            if stem in f.name: return f
    return None

def run_voxelizer(voxel_script: Path, cif_dir: Path, out_dir: Path, grid: int):
    cmd = [sys.executable, str(voxel_script), 
           "--cif-dir", str(cif_dir), 
           "--out-dir", str(out_dir), 
           "--grid", str(grid), 
           "--normalize", "per_channel_max", 
           "--include-charge"]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=3600)
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Voxelizer failed: {e.stderr}")
        return False

def extract_features_from_npz(npz_path: Path) -> Dict[str, float]:
    try:
        data = np.load(npz_path, allow_pickle=True)
        vox = data['vox']
        channels = [str(c) for c in data['channels']]
        ch_map = {c: i for i, c in enumerate(channels)}
        
        G = vox.shape[1]
        feats = {
            'global_mean': float(np.mean(vox)),
            'global_std': float(np.std(vox)),
            'sparsity': float(np.mean(vox == 0))
        }

        # Per-channel stats
        for ch, idx in ch_map.items():
            ch_data = vox[idx]
            occ = np.sum(ch_data)
            feats[f'{ch}_occupancy'] = float(occ)
            
            # Center of mass distance from center
            if occ > 0:
                coords = np.indices(ch_data.shape)
                com = np.array([np.sum(coords[i] * ch_data) for i in range(3)]) / occ
                dist = np.linalg.norm(com - G/2) / (G * np.sqrt(3)/2)
                feats[f'{ch}_com_dist'] = float(dist)

        # Metal-Organic overlap
        if 'metal' in ch_map and 'organic' in ch_map:
            m = vox[ch_map['metal']] > 0
            o = vox[ch_map['organic']] > 0
            feats['metal_organic_overlap'] = float(np.sum(m & o) / (np.sum(m | o) + 1e-9))
            
        return feats
    except Exception as e:
        logging.warning(f"Error extracting features {npz_path}: {e}")
        return {}
