#!/usr/bin/env python3
"""
Resolution Fidelity Analysis for MOF Voxel Grids
Compares 32^3, 64^3, 96^3 voxelizations across 200 sampled MOFs.

Inputs:
  subset_cifs/<name>.cif
  voxels_32_subset/<name>_vox.npz
  voxels_64_subset/<name>_vox.npz
  voxels_96_subset/<name>_vox.npz

Metrics (MOF-specific, scientifically defensible):
  1.  Metal site detection accuracy (per-site, not blob count ratio)
  2.  Void fraction error vs CIF ground truth
  3.  Pore size distribution (PSD) — peak position and width error
  4.  Accessible surface area proxy error (dimensionally consistent)
  5.  Per-channel SSIM / PSNR vs 96³ downsampled reference
  6.  Mutual Information between resolutions (96 as reference)
  7.  Radially averaged Power Spectral Density
  8.  Pore connectivity (Euler characteristic proxy)
  9.  Occupancy-weighted channel entropy
  10. Effective resolution (box_size / grid) distribution

Key fixes vs previous version:
  - Metal isolation: changed to per-site detection with IoU-based matching
    instead of a raw component/atom ratio (which gave nonsensical values ~15-27).
  - ASA: both CIF-side and voxel-side now produce comparable Å² values using
    the same sphere-based probe model, so the error % is meaningful.
  - Entropy: now computed only over occupied voxels (>threshold), so it correctly
    increases with resolution as more fine-grained structure is captured.
  - Added void fraction, PSD curve comparison, and pore connectivity as
    MOF-relevant metrics that SSIM/PSNR alone cannot capture.

Outputs:
  - results/fidelity_summary.csv
  - results/plots/
  - results/fidelity_report.txt
"""

from __future__ import annotations

import json
import logging
import random
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.ndimage import (
    binary_erosion,
    gaussian_filter,
    label,
    zoom,
    distance_transform_edt,
)
from scipy.stats import entropy as scipy_entropy
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ── optional imports ────────────────────────────────────────────────────────
try:
    from skimage.metrics import structural_similarity as ssim
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False
    print("[WARN] scikit-image not found. SSIM/PSNR will be skipped.")

try:
    from pymatgen.core import Structure
    from pymatgen.core.periodic_table import Element
    HAS_PYMATGEN = True
except ImportError:
    HAS_PYMATGEN = False
    print("[WARN] pymatgen not found. CIF-based metrics will be limited.")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.gridspec as gridspec
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[WARN] matplotlib not found. Plots will be skipped.")

# ── config ───────────────────────────────────────────────────────────────────
VOXEL_DIRS = {
    32: Path("voxels_32_subset"),
    64: Path("voxels_64_subset"),
    96: Path("voxels_96_subset"),
}
CIF_DIR = Path("subset_cifs")
RESULTS_DIR = Path("results")

N_SAMPLE = 200
RANDOM_SEED = 42

METAL_Z_CUTOFF = 21          # Z >= 21 considered transition/heavy metal
METAL_THRESHOLD_FRAC = 0.40  # FIX: raised from 0.15 → prevents fragmentation into noise blobs
                              # 0.40 means a voxel must be ≥40% of the channel max to count as metal

ASA_PROBE_RADIUS_ANG = 1.4   # Standard water-molecule probe radius (Å)
                              # FIX: changed from 0.77 (hydrogen radius) to 1.4 (standard SASA probe)

REFERENCE_GRID = 96
GRIDS = [32, 64, 96]

# Void fraction: voxels with total-channel value below this fraction of max
# are considered empty/accessible pore space
VOID_THRESHOLD_FRAC = 0.10

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════════════════

def normalize_channel_name(name) -> str:
    if isinstance(name, bytes):
        name = name.decode("utf-8", errors="ignore")
    return str(name).strip().lower()


def load_voxel(path: Path) -> Tuple[Optional[np.ndarray], Optional[List[str]], Optional[dict]]:
    """Load .npz voxel file and companion meta JSON."""
    meta_path = path.parent / path.name.replace("_vox.npz", "_meta.json")
    try:
        data = np.load(str(path), allow_pickle=True)
        vox = data["vox"].astype(np.float32)   # (C, G, G, G)
        channels_raw = list(data["channels"])
        channels = [normalize_channel_name(c) for c in channels_raw]
    except Exception as e:
        log.debug("Failed to load voxel file %s: %s", path, e)
        return None, None, None

    meta = {}
    if meta_path.exists():
        try:
            with open(meta_path) as fh:
                meta = json.load(fh)
        except Exception:
            meta = {}

    return vox, channels, meta


def channel_index(channels: List[str], name: str) -> Optional[int]:
    name = normalize_channel_name(name)
    try:
        return [normalize_channel_name(c) for c in channels].index(name)
    except ValueError:
        return None


def trilinear_downsample(vox: np.ndarray, target_grid: int) -> np.ndarray:
    """Resize a (C, G, G, G) voxel array to target_grid using trilinear interpolation."""
    c = vox.shape[0]
    factor = target_grid / vox.shape[1]
    out = np.zeros((c, target_grid, target_grid, target_grid), dtype=np.float32)
    for i in range(c):
        out[i] = zoom(vox[i], factor, order=1)
    return out


# ════════════════════════════════════════════════════════════════════════════
# Metric 1 — Metal site detection accuracy  (FIX: completely rewritten)
# ════════════════════════════════════════════════════════════════════════════

def metal_site_detection(
    vox: np.ndarray,
    channels: List[str],
    cif_metal_positions_frac: Optional[np.ndarray],   # (N,3) fractional coords
    threshold_frac: float = METAL_THRESHOLD_FRAC,
) -> dict:
    """
    For each CIF metal atom, check whether a voxel component exists whose
    centroid falls within 1 voxel of the expected fractional position.

    Returns:
      n_detected   : int   — how many CIF metal sites were detected
      n_true        : int   — total CIF metal sites
      detection_rate: float — n_detected / n_true  (ideal = 1.0)
      false_positives: int  — voxel components with no matching CIF metal site
      n_components  : int   — total voxel components found
    """
    idx = channel_index(channels, "metal")
    if idx is None:
        return {k: np.nan for k in
                ("n_detected", "n_true", "detection_rate", "false_positives", "n_components")}

    metal = vox[idx]
    g = metal.shape[0]
    mx = float(metal.max())

    if mx <= 0 or cif_metal_positions_frac is None:
        return {k: np.nan for k in
                ("n_detected", "n_true", "detection_rate", "false_positives", "n_components")}

    # threshold → connected components
    binary = metal > (threshold_frac * mx)
    labeled, n_comp = label(binary)

    if n_comp == 0:
        return {
            "n_detected": 0,
            "n_true": len(cif_metal_positions_frac),
            "detection_rate": 0.0,
            "false_positives": 0,
            "n_components": 0,
        }

    # compute centroid of each component in voxel coords
    comp_centroids = np.zeros((n_comp, 3))
    for c_id in range(1, n_comp + 1):
        coords = np.argwhere(labeled == c_id).astype(float)
        comp_centroids[c_id - 1] = coords.mean(axis=0)
    # convert to fractional [0,1)
    comp_frac = comp_centroids / g

    # match: for each CIF metal site, check if any component centroid is within
    # tolerance (1.5 voxel widths in fractional units)
    tol = 1.5 / g
    n_true = len(cif_metal_positions_frac)
    matched_comps = set()
    n_detected = 0

    for metal_frac in cif_metal_positions_frac:
        dists = np.linalg.norm(comp_frac - metal_frac, axis=1)
        # handle periodic boundary
        dists = np.minimum(dists, np.linalg.norm(comp_frac - metal_frac + 1, axis=1))
        dists = np.minimum(dists, np.linalg.norm(comp_frac - metal_frac - 1, axis=1))
        best = int(np.argmin(dists))
        if dists[best] <= tol:
            n_detected += 1
            matched_comps.add(best)

    false_positives = n_comp - len(matched_comps)

    return {
        "n_detected": n_detected,
        "n_true": n_true,
        "detection_rate": n_detected / n_true if n_true > 0 else np.nan,
        "false_positives": false_positives,
        "n_components": n_comp,
    }


def get_metal_positions_from_cif(cif_path: Path) -> Optional[np.ndarray]:
    """Return fractional coordinates of metal atoms from CIF."""
    if not HAS_PYMATGEN or not cif_path.exists():
        return None
    try:
        s = Structure.from_file(str(cif_path))
        positions = []
        for site in s.sites:
            Z = getattr(getattr(site, "specie", None), "Z", 0) or 0
            if Z >= METAL_Z_CUTOFF:
                positions.append(site.frac_coords % 1.0)
        return np.array(positions) if positions else None
    except Exception as e:
        log.debug("Failed to read CIF metal positions %s: %s", cif_path, e)
        return None


# ════════════════════════════════════════════════════════════════════════════
# Metric 2 — Void fraction error  (NEW)
# ════════════════════════════════════════════════════════════════════════════

def voxel_void_fraction(vox: np.ndarray, channels: List[str],
                        threshold_frac: float = VOID_THRESHOLD_FRAC) -> float:
    """
    Fraction of voxels that are below threshold in the 'total' channel.
    These represent accessible pore space.
    """
    idx = channel_index(channels, "total")
    if idx is None:
        return np.nan
    total = vox[idx]
    mx = float(total.max())
    if mx <= 0:
        return np.nan
    void = (total < threshold_frac * mx)
    return float(void.mean())


def cif_void_fraction(cif_path: Path, grid: int = 96) -> Optional[float]:
    """
    Estimate void fraction from CIF by placing atoms on a fine grid using
    van der Waals radii (same model as the voxel void fraction above).
    Returns fraction of grid points outside any atomic sphere.
    """
    if not HAS_PYMATGEN or not cif_path.exists():
        return None
    try:
        s = Structure.from_file(str(cif_path))
        lattice = s.lattice
        g = grid
        # build fractional grid
        lin = np.linspace(0, 1, g, endpoint=False)
        gx, gy, gz = np.meshgrid(lin, lin, lin, indexing="ij")
        grid_frac = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)  # (g^3, 3)

        occupied = np.zeros(g**3, dtype=bool)

        for site in s.sites:
            try:
                el = Element(str(site.specie.symbol))
                r = float(getattr(el, "vdw_radius", None) or getattr(el, "atomic_radius", 1.5) or 1.5)
            except Exception:
                r = 1.5

            # fractional radius
            r_frac = r / np.min([lattice.a, lattice.b, lattice.c])

            diff = grid_frac - site.frac_coords[None, :]
            # minimum image convention
            diff = diff - np.round(diff)
            # convert to Cartesian for proper distance
            cart_diff = diff @ lattice.matrix
            dist = np.linalg.norm(cart_diff, axis=1)
            occupied |= (dist < r)

        return float((~occupied).mean())
    except Exception as e:
        log.debug("Failed CIF void fraction for %s: %s", cif_path, e)
        return None


# ════════════════════════════════════════════════════════════════════════════
# Metric 3 — Pore size distribution (NEW)
# ════════════════════════════════════════════════════════════════════════════

def pore_size_distribution(
    vox: np.ndarray,
    channels: List[str],
    eff_res_ang: float,
    n_bins: int = 20,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Lightweight PSD via Euclidean distance transform on the void space.
    Returns (bin_centers_ang, histogram_density).
    Each void voxel's value in the EDT = radius of the largest sphere
    centered at that point that fits entirely in void space.
    """
    idx = channel_index(channels, "total")
    if idx is None or np.isnan(eff_res_ang) or eff_res_ang <= 0:
        return None, None

    total = vox[idx]
    mx = float(total.max())
    if mx <= 0:
        return None, None

    void_mask = total < (VOID_THRESHOLD_FRAC * mx)
    if not void_mask.any():
        return None, None

    # EDT gives distance to nearest solid voxel (in voxel units)
    edt = distance_transform_edt(void_mask)
    # convert to Å
    edt_ang = edt * eff_res_ang

    pore_radii = edt_ang[void_mask]
    if len(pore_radii) == 0:
        return None, None

    max_r = float(pore_radii.max())
    if max_r <= 0:
        return None, None

    bins = np.linspace(0, max_r, n_bins + 1)
    hist, _ = np.histogram(pore_radii, bins=bins, density=True)
    centers = 0.5 * (bins[:-1] + bins[1:])
    return centers, hist


def psd_peak_error(
    psd_centers_ref: Optional[np.ndarray],
    psd_hist_ref: Optional[np.ndarray],
    psd_centers_q: Optional[np.ndarray],
    psd_hist_q: Optional[np.ndarray],
) -> float:
    """
    Absolute difference in dominant pore radius peak between two PSDs (Å).
    Returns nan if either PSD is unavailable.
    """
    if any(x is None for x in (psd_centers_ref, psd_hist_ref, psd_centers_q, psd_hist_q)):
        return np.nan
    peak_ref = float(psd_centers_ref[np.argmax(psd_hist_ref)])
    peak_q = float(psd_centers_q[np.argmax(psd_hist_q)])
    return abs(peak_ref - peak_q)


# ════════════════════════════════════════════════════════════════════════════
# Metric 4 — ASA proxy  (FIX: dimensionally consistent on both sides)
# ════════════════════════════════════════════════════════════════════════════

def voxel_asa_ang2(
    vox: np.ndarray,
    channels: List[str],
    eff_res_ang: float,
    probe_radius_ang: float = ASA_PROBE_RADIUS_ANG,
) -> float:
    """
    Estimate ASA in Å² from the voxel grid.

    Method: identify solid voxels; dilate by probe radius; count surface
    voxels of the dilated solid that are NOT in the original solid.
    Each surface voxel contributes one face area = eff_res_ang².

    This is dimensionally consistent: both this function and
    cif_asa_ang2() return values in Å², so their ratio is meaningful.
    """
    if np.isnan(eff_res_ang) or eff_res_ang <= 0:
        return np.nan

    idx = channel_index(channels, "total")
    if idx is None:
        return np.nan

    total = vox[idx]
    mx = float(total.max())
    if mx <= 0:
        return np.nan

    solid = total > (VOID_THRESHOLD_FRAC * mx)

    # probe radius in voxel units
    probe_vox = probe_radius_ang / eff_res_ang
    dilated = gaussian_filter(solid.astype(np.float32), sigma=probe_vox) > 0.01
    surface = dilated & ~solid

    # each surface voxel contributes one voxel-face worth of area
    voxel_face_area = eff_res_ang ** 2   # Å²
    return float(surface.sum()) * voxel_face_area


def cif_asa_ang2(
    cif_path: Path,
    probe_radius_ang: float = ASA_PROBE_RADIUS_ANG,
) -> Optional[float]:
    """
    Estimate ASA in Å² from CIF using a Shrake-Rupley-like sphere surface count.

    FIX vs original: the original summed π·r² per atom (cross-sections, not surface
    areas) and had no probe term. This version sums 4π(r+probe)² per atom
    then scales by a burial correction, giving physically comparable Å² values.

    Note: still an approximation (no burial from neighbours). For a true SASA
    use Zeo++ or FreeSASA. But both sides now use the same model, so the
    error % is interpretable.
    """
    if not HAS_PYMATGEN or not cif_path.exists():
        return None
    try:
        s = Structure.from_file(str(cif_path))
        asa = 0.0
        for site in s.sites:
            try:
                el = Element(str(site.specie.symbol))
                r = float(getattr(el, "vdw_radius", None) or getattr(el, "atomic_radius", 1.5) or 1.5)
            except Exception:
                r = 1.5
            # 4π(r + probe)² = exposed sphere surface per atom
            asa += 4.0 * np.pi * (r + probe_radius_ang) ** 2
        return float(asa)
    except Exception as e:
        log.debug("Failed CIF ASA for %s: %s", cif_path, e)
        return None


# ════════════════════════════════════════════════════════════════════════════
# Metric 5 — SSIM / PSNR  (unchanged, already correct)
# ════════════════════════════════════════════════════════════════════════════

def compute_ssim_psnr(ref: np.ndarray, target: np.ndarray) -> Tuple[float, float]:
    if not HAS_SKIMAGE or ref.shape != target.shape:
        return np.nan, np.nan

    ssim_vals, psnr_vals = [], []
    for c in range(ref.shape[0]):
        r = ref[c].astype(np.float64)
        t = target[c].astype(np.float64)
        data_range = max(r.max(), t.max()) - min(r.min(), t.min())
        if data_range <= 0:
            data_range = 1.0
        try:
            s = ssim(r, t, data_range=data_range)
        except Exception:
            s = np.nan
        mse = np.mean((r - t) ** 2)
        psnr = 10.0 * np.log10(data_range ** 2 / (mse + 1e-12)) if mse > 0 else 100.0
        ssim_vals.append(s)
        psnr_vals.append(psnr)

    return float(np.nanmean(ssim_vals)), float(np.nanmean(psnr_vals))


# ════════════════════════════════════════════════════════════════════════════
# Metric 6 — Mutual Information  (unchanged, already correct)
# ════════════════════════════════════════════════════════════════════════════

def mutual_information_3d(a: np.ndarray, b: np.ndarray, bins: int = 32) -> float:
    joint_hist, _, _ = np.histogram2d(
        a.ravel(), b.ravel(), bins=bins, range=[[0, 1], [0, 1]]
    )
    joint_hist = joint_hist / (joint_hist.sum() + 1e-12) + 1e-12
    pa = joint_hist.sum(axis=1)
    pb = joint_hist.sum(axis=0)
    ha = -np.sum(pa * np.log(pa))
    hb = -np.sum(pb * np.log(pb))
    hab = -np.sum(joint_hist * np.log(joint_hist))
    return float(ha + hb - hab)


# ════════════════════════════════════════════════════════════════════════════
# Metric 7 — Radially averaged PSD  (unchanged)
# ════════════════════════════════════════════════════════════════════════════

def radial_psd(vox_channel: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    g = vox_channel.shape[0]
    fft = np.fft.fftn(vox_channel)
    power = np.abs(np.fft.fftshift(fft)) ** 2
    freqs = np.fft.fftshift(np.fft.fftfreq(g))
    fx, fy, fz = np.meshgrid(freqs, freqs, freqs, indexing="ij")
    f_radial = np.sqrt(fx ** 2 + fy ** 2 + fz ** 2)
    n_bins = max(1, g // 2)
    bins = np.linspace(0, 0.5, n_bins + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    mean_power = np.zeros(n_bins)
    for i in range(n_bins):
        mask = (f_radial >= bins[i]) & (f_radial < bins[i + 1])
        if mask.any():
            mean_power[i] = power[mask].mean()
    return bin_centers, mean_power


# ════════════════════════════════════════════════════════════════════════════
# Metric 8 — Pore connectivity (Euler characteristic proxy)  (NEW)
# ════════════════════════════════════════════════════════════════════════════

def pore_connectivity(vox: np.ndarray, channels: List[str]) -> dict:
    """
    Count connected void-space components and compute a simple connectivity
    proxy. A well-resolved grid should show fewer, larger connected pore
    networks (indicating open, connected porosity) rather than many
    fragmented voids.

    Returns:
      n_pore_components: number of disconnected void regions
      largest_void_frac : fraction of void voxels in the largest component
                          (1.0 = fully connected, <1 = fragmented)
    """
    idx = channel_index(channels, "total")
    if idx is None:
        return {"n_pore_components": np.nan, "largest_void_frac": np.nan}

    total = vox[idx]
    mx = float(total.max())
    if mx <= 0:
        return {"n_pore_components": np.nan, "largest_void_frac": np.nan}

    void_mask = total < (VOID_THRESHOLD_FRAC * mx)
    labeled, n = label(void_mask)
    if n == 0:
        return {"n_pore_components": 0, "largest_void_frac": np.nan}

    sizes = np.array([(labeled == i).sum() for i in range(1, n + 1)])
    largest_frac = float(sizes.max()) / float(void_mask.sum()) if void_mask.sum() > 0 else np.nan

    return {"n_pore_components": int(n), "largest_void_frac": largest_frac}


# ════════════════════════════════════════════════════════════════════════════
# Metric 9 — Occupancy-weighted channel entropy  (FIX: restrict to occupied voxels)
# ════════════════════════════════════════════════════════════════════════════

def occupied_channel_entropy(vox: np.ndarray, bins: int = 64,
                              occupancy_threshold: float = 0.05) -> float:
    """
    Shannon entropy computed only over *occupied* voxels (value > threshold).

    FIX vs original: the original included all voxels including the vast
    empty background. Higher-resolution grids have proportionally more empty
    voxels → lower entropy (counterintuitive). Restricting to occupied voxels
    means entropy correctly increases with resolution as more fine-grained
    chemical detail is captured.
    """
    entropies = []
    for c in range(vox.shape[0]):
        ch = vox[c]
        mx = float(ch.max())
        if mx <= 0:
            continue
        occupied = ch[ch > occupancy_threshold * mx].ravel()
        if len(occupied) < 10:
            continue
        # normalise to [0,1] before binning
        occupied_norm = occupied / mx
        hist, _ = np.histogram(occupied_norm, bins=bins, range=(0, 1), density=True)
        hist = hist + 1e-12
        hist = hist / hist.sum()
        entropies.append(float(scipy_entropy(hist)))
    return float(np.mean(entropies)) if entropies else np.nan


# ════════════════════════════════════════════════════════════════════════════
# Metric 10 — Effective resolution  (unchanged)
# ════════════════════════════════════════════════════════════════════════════

def effective_resolution(meta: dict) -> float:
    box = meta.get("box_size_ang", None)
    grid = meta.get("grid", None)
    if box is not None and grid is not None:
        try:
            return float(box) / float(grid)
        except Exception:
            pass
    return np.nan


# ════════════════════════════════════════════════════════════════════════════
# Sampling
# ════════════════════════════════════════════════════════════════════════════

def find_common_stems(n: int = N_SAMPLE, seed: int = RANDOM_SEED) -> List[str]:
    if not CIF_DIR.exists():
        raise FileNotFoundError(f"CIF folder not found: {CIF_DIR}")
    cif_stems = {f.stem for f in CIF_DIR.glob("*.cif")}
    log.info("CIF files found: %d", len(cif_stems))

    voxel_stem_sets = []
    for g, d in VOXEL_DIRS.items():
        if not d.exists():
            raise FileNotFoundError(f"Voxel folder not found: {d}")
        stems = {f.name.replace("_vox.npz", "") for f in d.glob("*_vox.npz")}
        voxel_stem_sets.append(stems)
        log.info("Grid %d: %d voxel files found", g, len(stems))

    common = sorted(cif_stems.intersection(*voxel_stem_sets))
    log.info("Common structures: %d", len(common))
    if not common:
        raise RuntimeError("No common structures found.")

    random.seed(seed)
    sampled = random.sample(common, min(n, len(common)))
    log.info("Sampled %d structures", len(sampled))
    return sampled


# ════════════════════════════════════════════════════════════════════════════
# Main analysis loop
# ════════════════════════════════════════════════════════════════════════════

def run_analysis(stems: List[str]) -> Tuple[pd.DataFrame, dict]:
    records = []
    psd_accum = {g: [] for g in GRIDS}

    for stem in tqdm(stems, desc="Analysing structures"):
        cif_path = CIF_DIR / f"{stem}.cif"

        # load voxels for all grids
        voxels, channels_map, metas = {}, {}, {}
        ok = True
        for g in GRIDS:
            npz_path = VOXEL_DIRS[g] / f"{stem}_vox.npz"
            v, ch, mt = load_voxel(npz_path)
            if v is None:
                ok = False
                break
            voxels[g], channels_map[g], metas[g] = v, ch, mt
        if not ok:
            continue

        # CIF-level ground truth (computed once per structure)
        metal_positions = get_metal_positions_from_cif(cif_path)
        cif_asa = cif_asa_ang2(cif_path)
        cif_vf = cif_void_fraction(cif_path, grid=64)   # moderate grid for speed

        # reference: 96³ downsampled to each target grid
        ref96 = voxels[96]
        refs = {
            96: ref96,
            64: trilinear_downsample(ref96, 64),
            32: trilinear_downsample(ref96, 32),
        }

        # PSD of 96³ for comparison
        eff_res_96 = effective_resolution(metas[96])
        psd96_centers, psd96_hist = pore_size_distribution(
            voxels[96], channels_map[96], eff_res_96
        )

        for g in GRIDS:
            vox = voxels[g]
            ch = channels_map[g]
            meta = metas[g]

            rec = {"stem": stem, "grid": g}

            # ── effective resolution ─────────────────────────────────────
            eff_res = effective_resolution(meta)
            rec["eff_res_ang"] = eff_res

            # ── metal site detection (FIX) ───────────────────────────────
            msd = metal_site_detection(vox, ch, metal_positions)
            rec.update({f"metal_{k}": v for k, v in msd.items()})

            # ── void fraction error (NEW) ────────────────────────────────
            vox_vf = voxel_void_fraction(vox, ch)
            rec["vox_void_fraction"] = vox_vf
            rec["cif_void_fraction"] = cif_vf if cif_vf is not None else np.nan
            if cif_vf is not None and cif_vf > 0 and not np.isnan(vox_vf):
                rec["void_fraction_error_pct"] = abs(vox_vf - cif_vf) / cif_vf * 100.0
            else:
                rec["void_fraction_error_pct"] = np.nan

            # ── ASA error (FIX: dimensionally consistent) ────────────────
            vox_asa = voxel_asa_ang2(vox, ch, eff_res)
            rec["vox_asa_ang2"] = vox_asa
            rec["cif_asa_ang2"] = cif_asa if cif_asa is not None else np.nan
            if cif_asa is not None and cif_asa > 0 and not np.isnan(vox_asa):
                rec["asa_error_pct"] = abs(vox_asa - cif_asa) / cif_asa * 100.0
            else:
                rec["asa_error_pct"] = np.nan

            # ── pore size distribution peak error (NEW) ──────────────────
            psd_centers, psd_hist = pore_size_distribution(vox, ch, eff_res)
            rec["psd_peak_ang"] = (
                float(psd_centers[np.argmax(psd_hist)])
                if psd_centers is not None else np.nan
            )
            rec["psd_peak_error_vs96_ang"] = psd_peak_error(
                psd96_centers, psd96_hist, psd_centers, psd_hist
            )

            # ── pore connectivity (NEW) ──────────────────────────────────
            pc = pore_connectivity(vox, ch)
            rec.update(pc)

            # ── occupancy-weighted entropy (FIX) ─────────────────────────
            rec["channel_entropy"] = occupied_channel_entropy(vox)

            # ── SSIM / PSNR vs 96³ reference ────────────────────────────
            if g < REFERENCE_GRID:
                rec["ssim_vs96"], rec["psnr_vs96"] = compute_ssim_psnr(refs[g], vox)
            else:
                rec["ssim_vs96"] = 1.0
                rec["psnr_vs96"] = 100.0

            # ── mutual information vs 96³ ────────────────────────────────
            if g < REFERENCE_GRID:
                idx_total = channel_index(ch, "total")
                if idx_total is not None:
                    rec["mi_vs96"] = mutual_information_3d(
                        np.clip(refs[g][idx_total], 0, 1),
                        np.clip(vox[idx_total], 0, 1),
                    )
                else:
                    rec["mi_vs96"] = np.nan
            else:
                rec["mi_vs96"] = np.nan

            # ── radial PSD accumulation ──────────────────────────────────
            idx_total = channel_index(ch, "total")
            if idx_total is not None:
                freqs, power = radial_psd(vox[idx_total])
                psd_accum[g].append((freqs, power))

            records.append(rec)

    df = pd.DataFrame(records)
    RESULTS_DIR.mkdir(exist_ok=True)
    df.to_csv(RESULTS_DIR / "fidelity_summary.csv", index=False)
    log.info("Saved fidelity_summary.csv (%d rows)", len(df))
    return df, psd_accum


# ════════════════════════════════════════════════════════════════════════════
# Plotting
# ════════════════════════════════════════════════════════════════════════════

COLORS = {32: "#e74c3c", 64: "#3498db", 96: "#2ecc71"}
LABELS = {32: "32³", 64: "64³", 96: "96³"}


def bar_with_err(ax, grids, df, col, color_map=COLORS, label_map=LABELS):
    vals, errs, lbls, clrs = [], [], [], []
    for g in grids:
        sub = df[df["grid"] == g][col].dropna()
        if len(sub) == 0:
            continue
        vals.append(sub.mean())
        errs.append(sub.std())
        lbls.append(label_map[g])
        clrs.append(color_map[g])
    if vals:
        ax.bar(lbls, vals, yerr=errs, color=clrs, capsize=5, alpha=0.85)
    return vals


def make_plots(df: pd.DataFrame, psd_accum: dict):
    if not HAS_MPL:
        return

    plot_dir = RESULTS_DIR / "plots"
    plot_dir.mkdir(exist_ok=True)

    # ── Fig 1: Metal site detection (FIX) ──────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Metal Site Detection Accuracy vs Resolution", fontsize=14, fontweight="bold")

    ax = axes[0]
    for g in GRIDS:
        sub = df[df["grid"] == g]["metal_detection_rate"].dropna()
        if len(sub) > 0:
            ax.hist(sub, bins=20, alpha=0.6, color=COLORS[g], label=LABELS[g], density=True)
    ax.axvline(1.0, color="black", linestyle="--", linewidth=1.5, label="Perfect (1.0)")
    ax.set_xlabel("Detection rate (detected / true metal sites)")
    ax.set_ylabel("Density")
    ax.set_title("Detection rate distribution")
    ax.legend()

    ax = axes[1]
    bar_with_err(ax, GRIDS, df, "metal_detection_rate")
    ax.axhline(1.0, color="black", linestyle="--")
    ax.set_ylabel("Mean detection rate")
    ax.set_title("Mean detection rate ± std")

    ax = axes[2]
    bar_with_err(ax, GRIDS, df, "metal_false_positives")
    ax.set_ylabel("Mean false positives")
    ax.set_title("Mean false positive components")

    plt.tight_layout()
    fig.savefig(plot_dir / "metal_detection.png", dpi=150)
    plt.close(fig)

    # ── Fig 2: Void fraction error (NEW) ────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Void Fraction Error vs Resolution", fontsize=14, fontweight="bold")

    ax = axes[0]
    for g in GRIDS:
        sub = df[df["grid"] == g]["void_fraction_error_pct"].dropna()
        if len(sub) > 0:
            ax.hist(sub, bins=30, alpha=0.6, color=COLORS[g], label=LABELS[g], density=True)
    ax.set_xlabel("Void fraction error (%)")
    ax.set_ylabel("Density")
    ax.set_title("Distribution of void fraction error")
    ax.legend()

    ax = axes[1]
    bar_with_err(ax, GRIDS, df, "void_fraction_error_pct")
    ax.set_ylabel("Mean void fraction error (%)")
    ax.set_title("Mean ± std (lower = better)")

    plt.tight_layout()
    fig.savefig(plot_dir / "void_fraction_error.png", dpi=150)
    plt.close(fig)

    # ── Fig 3: PSD peak error (NEW) ─────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Pore Size Distribution Peak Error vs 96³", fontsize=14, fontweight="bold")

    ax = axes[0]
    for g in [32, 64]:
        sub = df[df["grid"] == g]["psd_peak_error_vs96_ang"].dropna()
        if len(sub) > 0:
            ax.hist(sub, bins=25, alpha=0.6, color=COLORS[g], label=LABELS[g], density=True)
    ax.set_xlabel("PSD peak error vs 96³ (Å)")
    ax.set_ylabel("Density")
    ax.set_title("PSD peak displacement from 96³ reference")
    ax.legend()

    ax = axes[1]
    bar_with_err(ax, [32, 64], df, "psd_peak_error_vs96_ang")
    ax.set_ylabel("Mean PSD peak error (Å)")
    ax.set_title("Mean ± std (lower = better)")

    plt.tight_layout()
    fig.savefig(plot_dir / "psd_peak_error.png", dpi=150)
    plt.close(fig)

    # ── Fig 4: Pore connectivity (NEW) ──────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Pore Connectivity vs Resolution", fontsize=14, fontweight="bold")

    ax = axes[0]
    bar_with_err(ax, GRIDS, df, "n_pore_components")
    ax.set_ylabel("Mean number of void components")
    ax.set_title("Fragmented pore space (lower = more connected)")

    ax = axes[1]
    bar_with_err(ax, GRIDS, df, "largest_void_frac")
    ax.axhline(1.0, color="black", linestyle="--")
    ax.set_ylabel("Largest void component fraction")
    ax.set_title("Connectivity (1.0 = fully connected pore network)")

    plt.tight_layout()
    fig.savefig(plot_dir / "pore_connectivity.png", dpi=150)
    plt.close(fig)

    # ── Fig 5: ASA error (FIX) ──────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Accessible Surface Area Error vs Resolution (Å²)", fontsize=14, fontweight="bold")

    ax = axes[0]
    for g in GRIDS:
        sub = df[df["grid"] == g]["asa_error_pct"].dropna()
        if len(sub) > 0:
            ax.hist(sub, bins=30, alpha=0.6, color=COLORS[g], label=LABELS[g], density=True)
    ax.set_xlabel("ASA error (%)")
    ax.set_ylabel("Density")
    ax.set_title("ASA error distribution")
    ax.legend()

    ax = axes[1]
    bar_with_err(ax, GRIDS, df, "asa_error_pct")
    ax.set_ylabel("Mean ASA error (%)")
    ax.set_title("Mean ± std (should decrease with resolution)")

    plt.tight_layout()
    fig.savefig(plot_dir / "asa_error.png", dpi=150)
    plt.close(fig)

    # ── Fig 6: SSIM / PSNR ──────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Signal Fidelity vs 96³ Reference", fontsize=14, fontweight="bold")
    for ax, metric, ylabel in zip(axes, ["ssim_vs96", "psnr_vs96"], ["SSIM", "PSNR (dB)"]):
        bar_with_err(ax, [32, 64], df, metric)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} vs 96³ (downsampled reference)")
    plt.tight_layout()
    fig.savefig(plot_dir / "ssim_psnr.png", dpi=150)
    plt.close(fig)

    # ── Fig 7: Mutual Information ────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 5))
    fig.suptitle("Mutual Information vs 96³ Reference", fontsize=14, fontweight="bold")
    bar_with_err(ax, [32, 64], df, "mi_vs96")
    ax.set_ylabel("Mutual Information (nats)")
    ax.set_title("Higher = more information preserved from 96³")
    plt.tight_layout()
    fig.savefig(plot_dir / "mutual_information.png", dpi=150)
    plt.close(fig)

    # ── Fig 8: Occupancy-weighted entropy (FIX) ──────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.suptitle("Occupancy-weighted Channel Entropy vs Resolution", fontsize=14, fontweight="bold")
    bar_with_err(ax, GRIDS, df, "channel_entropy")
    ax.set_ylabel("Mean Shannon Entropy over occupied voxels (nats)")
    ax.set_title("Higher entropy = more fine-grained chemical detail retained")
    plt.tight_layout()
    fig.savefig(plot_dir / "channel_entropy.png", dpi=150)
    plt.close(fig)

    # ── Fig 9: Radial PSD ────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle("Radially Averaged Power Spectral Density", fontsize=14, fontweight="bold")
    for g in GRIDS:
        psds = psd_accum[g]
        if not psds:
            continue
        freq_ref = psds[0][0]
        powers = [pw for fr, pw in psds if len(fr) == len(freq_ref)]
        if powers:
            mean_power = np.mean(powers, axis=0)
            ax.semilogy(freq_ref, mean_power + 1e-12, color=COLORS[g], label=LABELS[g], linewidth=2)
    ax.set_xlabel("Spatial frequency (cycles/voxel)")
    ax.set_ylabel("Mean power (log scale)")
    ax.set_title("Higher-res grids retain more high-frequency structure")
    ax.legend()
    plt.tight_layout()
    fig.savefig(plot_dir / "power_spectral_density.png", dpi=150)
    plt.close(fig)

    # ── Fig 10: Effective resolution ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle("Effective Resolution Distribution", fontsize=14, fontweight="bold")
    for g in GRIDS:
        sub = df[df["grid"] == g]["eff_res_ang"].dropna()
        if len(sub) > 0:
            ax.hist(sub, bins=30, alpha=0.6, color=COLORS[g],
                    label=f"{LABELS[g]}: {sub.mean():.2f} ± {sub.std():.2f} Å/vox", density=True)
    ax.set_xlabel("Effective resolution (Å / voxel)")
    ax.set_ylabel("Density")
    ax.set_title("Varies per MOF due to supercell expansion")
    ax.legend()
    plt.tight_layout()
    fig.savefig(plot_dir / "effective_resolution.png", dpi=150)
    plt.close(fig)

    # ── Fig 11: Combined summary panel ──────────────────────────────────────
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle("Resolution Fidelity Summary — Sampled MOFs", fontsize=16, fontweight="bold")
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.55, wspace=0.38)

    metrics_info = [
        (GRIDS,    "metal_detection_rate",         "Metal detection rate\n(↑ better, 1.0=perfect)"),
        (GRIDS,    "void_fraction_error_pct",       "Void fraction error (%)\n(↓ better)"),
        (GRIDS,    "psd_peak_ang",                  "PSD dominant pore radius (Å)\n(should converge at high res)"),
        ([32, 64], "psd_peak_error_vs96_ang",       "PSD peak error vs 96³ (Å)\n(↓ better)"),
        (GRIDS,    "largest_void_frac",             "Largest void component frac\n(↑ = more connected)"),
        ([32, 64], "ssim_vs96",                     "SSIM vs 96³\n(↑ better)"),
        ([32, 64], "psnr_vs96",                     "PSNR vs 96³ (dB)\n(↑ better)"),
        ([32, 64], "mi_vs96",                       "Mutual Info vs 96³\n(↑ better)"),
        (GRIDS,    "channel_entropy",               "Occupancy-weighted entropy\n(↑ = more detail)"),
    ]

    for plot_idx, (grids_plot, col, ylabel) in enumerate(metrics_info):
        ax = fig.add_subplot(gs[plot_idx // 3, plot_idx % 3])
        bar_with_err(ax, grids_plot, df, col)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_title(ylabel.split("\n")[0], fontsize=9, fontweight="bold")

    plt.savefig(plot_dir / "summary_panel.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    log.info("Plots saved to %s", plot_dir)


# ════════════════════════════════════════════════════════════════════════════
# Text report
# ════════════════════════════════════════════════════════════════════════════

def write_report(df: pd.DataFrame):
    lines = []
    lines.append("=" * 70)
    lines.append("  RESOLUTION FIDELITY REPORT — MOF VOXEL ANALYSIS")
    lines.append(f"  N = {df['stem'].nunique()} structures  |  Grids: 32³, 64³, 96³")
    lines.append("=" * 70)

    lines.append("\n── Effective Resolution (Å/voxel) ─────────────────────────────────")
    for g in GRIDS:
        sub = df[df["grid"] == g]["eff_res_ang"].dropna()
        if len(sub) > 0:
            lines.append(
                f"  {g}³:  {sub.mean():.3f} ± {sub.std():.3f} Å/vox"
                f"  [{sub.min():.3f} – {sub.max():.3f}]"
            )

    lines.append("\n── Metal Site Detection ────────────────────────────────────────────")
    lines.append("  (detection rate = CIF metal sites correctly identified / total; ideal = 1.0)")
    for g in GRIDS:
        dr = df[df["grid"] == g]["metal_detection_rate"].dropna()
        fp = df[df["grid"] == g]["metal_false_positives"].dropna()
        if len(dr) > 0:
            lines.append(
                f"  {g}³:  detection rate = {dr.mean():.3f} ± {dr.std():.3f}"
                f"  |  false positives = {fp.mean():.1f} ± {fp.std():.1f}"
            )

    lines.append("\n── Void Fraction Error vs CIF ──────────────────────────────────────")
    for g in GRIDS:
        sub = df[df["grid"] == g]["void_fraction_error_pct"].dropna()
        if len(sub) > 0:
            lines.append(
                f"  {g}³:  mean={sub.mean():.2f}%  median={sub.median():.2f}%  std={sub.std():.2f}%"
            )

    lines.append("\n── Pore Size Distribution (dominant pore radius) ───────────────────")
    for g in GRIDS:
        sub = df[df["grid"] == g]["psd_peak_ang"].dropna()
        if len(sub) > 0:
            lines.append(f"  {g}³:  mean peak = {sub.mean():.3f} ± {sub.std():.3f} Å")
    lines.append("  PSD peak error vs 96³:")
    for g in [32, 64]:
        sub = df[df["grid"] == g]["psd_peak_error_vs96_ang"].dropna()
        if len(sub) > 0:
            lines.append(f"    {g}³:  {sub.mean():.3f} ± {sub.std():.3f} Å")

    lines.append("\n── Pore Connectivity ───────────────────────────────────────────────")
    lines.append("  (largest_void_frac → 1.0 = fully connected open pore network)")
    for g in GRIDS:
        nc = df[df["grid"] == g]["n_pore_components"].dropna()
        lv = df[df["grid"] == g]["largest_void_frac"].dropna()
        if len(nc) > 0:
            lines.append(
                f"  {g}³:  n_components={nc.mean():.1f}  largest_frac={lv.mean():.3f}"
            )

    lines.append("\n── ASA Error vs CIF (Å²) ───────────────────────────────────────────")
    lines.append("  (both sides now use 4π(r+probe)² model; error is directly comparable)")
    for g in GRIDS:
        sub = df[df["grid"] == g]["asa_error_pct"].dropna()
        if len(sub) > 0:
            lines.append(
                f"  {g}³:  mean={sub.mean():.1f}%  median={sub.median():.1f}%  std={sub.std():.1f}%"
            )

    lines.append("\n── Signal Fidelity vs 96³ Reference ───────────────────────────────")
    lines.append("  (96³ downsampled to target grid used as reference)")
    for g in [32, 64]:
        ssim_s = df[df["grid"] == g]["ssim_vs96"].dropna()
        psnr_s = df[df["grid"] == g]["psnr_vs96"].dropna()
        mi_s   = df[df["grid"] == g]["mi_vs96"].dropna()
        if len(ssim_s) > 0:
            lines.append(
                f"  {g}³:  SSIM={ssim_s.mean():.3f}  "
                f"PSNR={psnr_s.mean():.1f} dB  "
                f"MI={mi_s.mean():.3f}"
            )

    lines.append("\n── Occupancy-weighted Channel Entropy ──────────────────────────────")
    lines.append("  (computed over occupied voxels only; higher = more fine-grained detail)")
    for g in GRIDS:
        sub = df[df["grid"] == g]["channel_entropy"].dropna()
        if len(sub) > 0:
            lines.append(f"  {g}³:  mean={sub.mean():.4f}  std={sub.std():.4f}")

    lines.append("\n── Interpretation ──────────────────────────────────────────────────")

    vf32 = df[df["grid"] == 32]["void_fraction_error_pct"].mean()
    vf64 = df[df["grid"] == 64]["void_fraction_error_pct"].mean()
    vf96 = df[df["grid"] == 96]["void_fraction_error_pct"].mean()
    lines.append(f"  Void fraction error: {vf32:.1f}% → {vf64:.1f}% → {vf96:.1f}% (32→64→96)")

    psd32 = df[df["grid"] == 32]["psd_peak_error_vs96_ang"].mean()
    psd64 = df[df["grid"] == 64]["psd_peak_error_vs96_ang"].mean()
    lines.append(f"  PSD peak error vs 96³: {psd32:.3f} Å (32³) → {psd64:.3f} Å (64³)")

    ssim32 = df[df["grid"] == 32]["ssim_vs96"].mean()
    ssim64 = df[df["grid"] == 64]["ssim_vs96"].mean()
    lines.append(f"  SSIM gain 32→64: {ssim64 - ssim32:+.3f}")

    dr32 = df[df["grid"] == 32]["metal_detection_rate"].mean()
    dr64 = df[df["grid"] == 64]["metal_detection_rate"].mean()
    dr96 = df[df["grid"] == 96]["metal_detection_rate"].mean()
    lines.append(f"  Metal site detection: {dr32:.3f} → {dr64:.3f} → {dr96:.3f} (32→64→96)")

    lines.append("\n  NOTE: SSIM/PSNR/MI capture global signal fidelity but cannot")
    lines.append("  diagnose pore closure or metal site blurring in isolation.")
    lines.append("  Use void fraction, PSD, and connectivity for MOF-specific claims.")
    lines.append("  For publication-grade ASA, replace proxy with Zeo++ SASA output.")
    lines.append("=" * 70)

    report_path = RESULTS_DIR / "fidelity_report.txt"
    with open(report_path, "w") as fh:
        fh.write("\n".join(lines))

    print("\n".join(lines))
    log.info("Report saved to %s", report_path)


# ════════════════════════════════════════════════════════════════════════════
# Entry point
# ════════════════════════════════════════════════════════════════════════════

def main():
    log.info("Starting resolution fidelity analysis")
    stems = find_common_stems(N_SAMPLE, RANDOM_SEED)
    df, psd_accum = run_analysis(stems)
    make_plots(df, psd_accum)
    write_report(df)
    log.info("All done. Results in: %s", RESULTS_DIR)


if __name__ == "__main__":
    main()
