# mof_voxelizer/grid.py
import math
import numpy as np
from scipy.ndimage import gaussian_filter
from pymatgen.core.periodic_table import Element

def estimate_sigma_vox(symbol: str, grid: int, box_size_ang: float, base_sigma_ang: float = 0.6) -> float:
    """Calculates dynamic sigma based on atomic radius."""
    try:
        el = Element(symbol)
        rad = el.covalent_radius or el.atomic_radius or el.vdw_radius or base_sigma_ang
        sigma_ang = max(0.3, float(rad) * 0.6)
    except Exception:
        sigma_ang = base_sigma_ang
        
    sigma_vox = (sigma_ang / float(box_size_ang)) * float(grid)
    return float(max(0.25, min(sigma_vox, grid / 4.0)))

def trilinear_splat(vox: np.ndarray, x_f: float, y_f: float, z_f: float, val: float, ch_idx: int):
    """Splats value into 8 neighbors based on distance."""
    G = vox.shape[1]
    gx, gy, gz = int(np.floor(x_f)), int(np.floor(y_f)), int(np.floor(z_f))
    dx, dy, dz = x_f - gx, y_f - gy, z_f - gz
    
    for ox in (0, 1):
        wx = (1 - dx) if ox == 0 else dx
        ix = gx + ox
        if not (0 <= ix < G): continue
        
        for oy in (0, 1):
            wy = (1 - dy) if oy == 0 else dy
            iy = gy + oy
            if not (0 <= iy < G): continue
            
            for oz in (0, 1):
                wz = (1 - dz) if oz == 0 else dz
                iz = gz + oz
                if not (0 <= iz < G): continue
                
                vox[ch_idx, ix, iy, iz] += val * wx * wy * wz

def add_local_gaussian(vox: np.ndarray, x_f: float, y_f: float, z_f: float, amp: float, sigma: float, ch_idx: int):
    """Adds a Gaussian kernel centered at (x,y,z) directly to the grid."""
    if sigma <= 0.0: return
    G = vox.shape[1]
    r = max(1, int(math.ceil(3.0 * sigma)))
    ix_c, iy_c, iz_c = int(round(x_f)), int(round(y_f)), int(round(z_f))
    s2 = sigma ** 2
    
    # Bounding box clamping
    x0, x1 = max(0, ix_c - r), min(G - 1, ix_c + r)
    y0, y1 = max(0, iy_c - r), min(G - 1, iy_c + r)
    z0, z1 = max(0, iz_c - r), min(G - 1, iz_c + r)

    for ix in range(x0, x1 + 1):
        dx2 = (ix - x_f) ** 2
        for iy in range(y0, y1 + 1):
            dy2 = (iy - y_f) ** 2
            for iz in range(z0, z1 + 1):
                dz2 = (iz - z_f) ** 2
                val = amp * math.exp(-0.5 * (dx2 + dy2 + dz2) / s2)
                vox[ch_idx, ix, iy, iz] += val

def apply_global_smoothing(vox: np.ndarray, sigma: float):
    for ci in range(vox.shape[0]):
        vox[ci] = gaussian_filter(vox[ci].astype(np.float64), sigma=sigma, mode="constant", cval=0.0)

def normalize_voxels(vox: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none": return vox
    
    if mode == "per_channel_max":
        # Vectorized normalization
        maxs = vox.max(axis=(1, 2, 3), keepdims=True)
        maxs[maxs == 0] = 1.0 # Prevent div/0
        return vox / maxs
        
    elif mode == "global_max":
        gm = vox.max()
        return vox / gm if gm > 0 else vox
        
    elif mode == "sum_normalize":
        sums = vox.sum(axis=(1, 2, 3), keepdims=True)
        sums[sums == 0] = 1.0
        return vox / sums
        
    raise ValueError(f"Unknown normalize mode: {mode}")
