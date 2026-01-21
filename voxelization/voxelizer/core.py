# mof_voxelizer/core.py
import numpy as np
from typing import List, Tuple, Optional
from pymatgen.core import Structure

from .constants import DEFAULT_ELEM_CHANNELS, METAL_Z_CUTOFF
from .chemistry import build_supercell, get_atom_info, extract_site_charge
from .grid import trilinear_splat, add_local_gaussian, estimate_sigma_vox, apply_global_smoothing, normalize_voxels

def voxelize_structure(
    struct: Structure,
    grid: int = 64,
    elem_channels: Optional[List[str]] = None,
    Lmin: float = 35.0,
    default_sigma_vox: float = 0.75,
    normalize: str = "per_channel_max",
    use_trilinear: bool = True,
    apply_gaussian: bool = True,
    include_charge: bool = True,
    map_mode: str = "fractional",
    per_atom_gauss: bool = False,
) -> Tuple[np.ndarray, List[str], float]:
    
    elem_channels = elem_channels or DEFAULT_ELEM_CHANNELS
    s = build_supercell(struct, Lmin)

    # 1. Define Channels
    channels = ["total", "metal", "organic"] + list(elem_channels)
    if include_charge:
        channels.append("charge")
    ch_to_idx = {ch: i for i, ch in enumerate(channels)}
    
    # 2. Determine Box Size & Coordinates
    if map_mode == "fractional":
        lattice_lengths = np.array(s.lattice.abc)
        box_size_ang = float(max(lattice_lengths.max(), float(Lmin)))
        # Map fractional [0,1] to voxel [0, G-1]
        coords_src = np.array([site.frac_coords for site in s.sites])
        coords_uvw = coords_src % 1.0
    elif map_mode == "cartesian":
        coords = np.array([site.coords for site in s.sites])
        mins, maxs = coords.min(axis=0), coords.max(axis=0)
        box_size_ang = max(float((maxs - mins).max()), float(Lmin))
        # Center coordinates
        center = (mins + maxs) / 2.0
        coords_centered = coords - center
        coords_uvw = (coords_centered + box_size_ang / 2.0) / box_size_ang
        coords_uvw = np.clip(coords_uvw, 0.0, 1.0 - 1e-12)
    else:
        raise ValueError(f"Unknown map_mode: {map_mode}")

    vox_coords = coords_uvw * (grid - 1.0)
    vox = np.zeros((len(channels), grid, grid, grid), dtype=np.float64)

    # 3. Splatting Loop
    for i, site in enumerate(s.sites):
        x, y, z = vox_coords[i]
        sym, Z, occ = get_atom_info(site)
        
        # Determine contributions
        contributions = [("total", 1.0)]
        contributions.append(("metal" if Z >= METAL_Z_CUTOFF else "organic", 1.0))
        
        sym_cap = str(sym).capitalize()
        if sym_cap in elem_channels:
            contributions.append((sym_cap, 1.0))

        # Apply structural channels
        for name, weight in contributions:
            val = weight * occ
            idx = ch_to_idx[name]
            
            if per_atom_gauss:
                sigma = estimate_sigma_vox(sym_cap, grid, box_size_ang)
                add_local_gaussian(vox, x, y, z, val, sigma, idx)
            elif use_trilinear:
                trilinear_splat(vox, x, y, z, val, idx)
            else:
                # Nearest Neighbor fallback
                ix, iy, iz = int(round(x)), int(round(y)), int(round(z))
                if 0 <= ix < grid and 0 <= iy < grid and 0 <= iz < grid:
                    vox[idx, ix, iy, iz] += val

        # Apply Charge
        if include_charge:
            ch_val = extract_site_charge(site) * occ
            idx_c = ch_to_idx["charge"]
            if per_atom_gauss:
                add_local_gaussian(vox, x, y, z, ch_val, default_sigma_vox, idx_c)
            elif use_trilinear:
                trilinear_splat(vox, x, y, z, ch_val, idx_c)

    # 4. Post-processing
    if apply_gaussian and not per_atom_gauss:
        apply_global_smoothing(vox, default_sigma_vox)

    vox = vox.astype(np.float32)
    vox = normalize_voxels(vox, normalize)

    return vox, channels, float(box_size_ang)
