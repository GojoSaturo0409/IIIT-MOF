#!/usr/bin/env python3
"""
voxelize_cifs_publishable_fixed.py

Robust, publication-ready voxelizer for CIF -> multi-channel 3D voxel grids.

Key fixes vs original:
 - lattice-aware fractional mapping (default) with optional cartesian mode
 - robust handling of species/Specie/Element atomic number lookups
 - trilinear splatting for charge channel (consistent with other channels)
 - optional per-atom gaussian splatting (enabled with --per-atom-gauss)
 - case-insensitive CIF scanning
 - save `channels` inside .npz alongside `vox`
 - save pymatgen package version in metadata
 - idempotent logging setup (avoids double handlers)
 - added CLI flags for map-mode and per-atom gaussian, with performance notes
 - added small unit-test helper `run_sanity_test()` when run with `--test`

Usage example:
  python voxelize_cifs_publishable_fixed.py --cif-dir repeat_cifs --out-dir voxels --grid 64 --Lmin 35.0

Dependencies:
  pip install numpy scipy pymatgen tqdm

"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
import platform
import math
from typing import List, Tuple, Optional

import numpy as np
from pymatgen.core import Structure
import pymatgen
from pymatgen.core.periodic_table import Element
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

# -------------------------- Defaults & constants -------------------------
DEFAULT_ELEM_CHANNELS = ["C", "O", "N", "H"]
METAL_Z_CUTOFF = 21

# -------------------------- Helpers -------------------------------------

def setup_logger(verbose: bool = False):
    """Idempotent logger setup."""
    level = logging.DEBUG if verbose else logging.INFO
    logger = logging.getLogger()
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(level)


def build_supercell(struct: Structure, Lmin: float) -> Structure:
    """Expand the structure so each lattice vector length >= Lmin."""
    a, b, c = struct.lattice.abc
    na = max(1, int(np.ceil(Lmin / a)))
    nb = max(1, int(np.ceil(Lmin / b)))
    nc = max(1, int(np.ceil(Lmin / c)))
    if (na, nb, nc) != (1, 1, 1):
        logging.debug("Expanding supercell: %s -> (%d,%d,%d)", struct.formula, na, nb, nc)
        return struct * (na, nb, nc)
    return struct


def extract_site_charge(site) -> float:
    """Robustly extract a partial charge for a site if present; otherwise 0.0."""
    props = getattr(site, "properties", {}) or {}
    keys = (
        "partial_charge",
        "partial_charges",
        "charge",
        "q",
        "_atom_site_partial_charge",
        "_atom_site_charges",
    )
    for k in keys:
        if k in props and props[k] is not None:
            try:
                return float(props[k])
            except Exception:
                try:
                    return float(props[k][0])
                except Exception:
                    pass
    # try species properties
    try:
        sp = site.specie
        if hasattr(sp, "properties") and sp.properties:
            for k in ("partial_charge", "charge", "q"):
                if k in sp.properties:
                    try:
                        return float(sp.properties[k])
                    except Exception:
                        pass
    except Exception:
        pass
    return 0.0


def get_species_and_occupancy(site) -> List[Tuple[object, float]]:
    """Return list of tuples (Specie/Element, occupancy) for the site.
    Handles both ordered sites (site.specie) and disordered (site.species.items()).
    """
    try:
        sp = site.specie
        return [(sp, 1.0)]
    except Exception:
        try:
            items = list(site.species.items())
            if len(items) == 0:
                return []
            return items
        except Exception:
            return []


def _atomic_number_from_specie(sp) -> int:
    """Robust atomic number lookup from a pymatgen Specie/Element-like object."""
    # common attributes: Z, atomic_number
    Z = getattr(sp, "Z", None)
    if Z is None:
        Z = getattr(sp, "atomic_number", None)
    try:
        if Z is None and hasattr(sp, "__str__"):
            # fallback: try Element(sp.symbol) if possible
            sym = getattr(sp, "symbol", None) or str(sp)
            try:
                el = Element(sym)
                Z = getattr(el, "Z", getattr(el, "atomic_number", None))
            except Exception:
                Z = None
    except Exception:
        Z = None
    return int(Z) if Z is not None else 0


def _symbol_from_specie(sp) -> str:
    # prefer symbol attribute, else str()
    return getattr(sp, "symbol", None) or str(sp)


def get_symbol_Z_and_occupancy(site) -> Tuple[str, int, float]:
    """Return (symbol, Z, occupancy) choosing the highest-occupancy specie when disordered."""
    spp = get_species_and_occupancy(site)
    if len(spp) == 0:
        # fallback: try site.specie
        try:
            sp = site.specie
            sym = _symbol_from_specie(sp)
            Z = _atomic_number_from_specie(sp)
            return sym, Z, 1.0
        except Exception:
            return "X", 0, 1.0
    spp_sorted = sorted(spp, key=lambda x: x[1], reverse=True)
    sp, occ = spp_sorted[0]
    return _symbol_from_specie(sp), _atomic_number_from_specie(sp), float(occ)


def estimate_sigma_vox_from_element(symbol: str, grid: int, box_size_ang: float, base_sigma_ang: float = 0.6) -> float:
    """Estimate per-atom gaussian sigma in voxels using element radius.
    sigma_vox = (sigma_ang / box_size_ang) * grid
    """
    try:
        el = Element(symbol)
        rad = None
        if getattr(el, "covalent_radius", None) is not None:
            rad = el.covalent_radius
        elif getattr(el, "atomic_radius", None) is not None:
            rad = el.atomic_radius
        elif getattr(el, "vdw_radius", None) is not None:
            rad = el.vdw_radius
        if rad is None:
            rad = base_sigma_ang
        sigma_ang = max(0.3, float(rad) * 0.6)
    except Exception:
        sigma_ang = base_sigma_ang
    sigma_vox = (sigma_ang / float(box_size_ang)) * float(grid)
    sigma_vox = float(max(0.25, min(sigma_vox, grid / 4.0)))
    return sigma_vox


def trilinear_splat(vox: np.ndarray, x_f: float, y_f: float, z_f: float, val: float, ch_idx: int):
    """Add `val` to the eight neighboring voxels using trilinear interpolation weights."""
    G = vox.shape[1]
    gx = int(np.floor(x_f))
    gy = int(np.floor(y_f))
    gz = int(np.floor(z_f))
    dx = x_f - gx
    dy = y_f - gy
    dz = z_f - gz
    for ox in (0, 1):
        wx = (1 - dx) if ox == 0 else dx
        ix = gx + ox
        if ix < 0 or ix >= G:
            continue
        for oy in (0, 1):
            wy = (1 - dy) if oy == 0 else dy
            iy = gy + oy
            if iy < 0 or iy >= G:
                continue
            for oz in (0, 1):
                wz = (1 - dz) if oz == 0 else dz
                iz = gz + oz
                if iz < 0 or iz >= G:
                    continue
                w = wx * wy * wz
                if w > 0:
                    vox[ch_idx, ix, iy, iz] += val * w


def normalize_voxels(vox: np.ndarray, mode: str = "per_channel_max") -> np.ndarray:
    """Normalize voxels in-place according to chosen mode."""
    assert vox.ndim == 4
    if mode == "none":
        return vox
    if mode == "per_channel_max":
        for ci in range(vox.shape[0]):
            m = vox[ci].max()
            if m > 0:
                vox[ci] = vox[ci] / float(m)
    elif mode == "global_max":
        gm = vox.max()
        if gm > 0:
            vox[:] = vox / float(gm)
    elif mode == "sum_normalize":
        for ci in range(vox.shape[0]):
            s = vox[ci].sum()
            if s > 0:
                vox[ci] = vox[ci] / float(s)
    else:
        raise ValueError("Unknown normalize mode: %s" % mode)
    return vox


def sanity_checks(vox: np.ndarray, channels: List[str]):
    if np.isnan(vox).any():
        raise ValueError("NaNs found in voxel grid")
    if vox.ndim != 4:
        raise ValueError("Voxel array must be 4D (C,G,G,G)")
    empties = [c for i, c in enumerate(channels) if vox[i].max() == 0]
    if empties:
        logging.debug("Empty channels (may be expected): %s", empties)


# ------------------------- Core voxelization -----------------------------

def _add_local_gaussian(vox: np.ndarray, x_f: float, y_f: float, z_f: float, amplitude: float, sigma: float, ch_idx: int):
    """Add a small Gaussian kernel centered at (x_f,y_f,z_f) in voxel coordinates.
    This does a brute-force local kernel addition over a cube of +/- ceil(3*sigma) voxels.
    """
    if sigma <= 0.0:
        return
    G = vox.shape[1]
    r = max(1, int(math.ceil(3.0 * sigma)))
    ix_c = int(round(x_f))
    iy_c = int(round(y_f))
    iz_c = int(round(z_f))
    x0 = max(0, ix_c - r)
    x1 = min(G - 1, ix_c + r)
    y0 = max(0, iy_c - r)
    y1 = min(G - 1, iy_c + r)
    z0 = max(0, iz_c - r)
    z1 = min(G - 1, iz_c + r)
    # compute squared sigma
    s2 = (sigma ** 2)
    for ix in range(x0, x1 + 1):
        dx2 = (ix - x_f) ** 2
        for iy in range(y0, y1 + 1):
            dy2 = (iy - y_f) ** 2
            for iz in range(z0, z1 + 1):
                dz2 = (iz - z_f) ** 2
                val = amplitude * math.exp(-0.5 * (dx2 + dy2 + dz2) / s2)
                vox[ch_idx, ix, iy, iz] += val


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
    """Voxelize a pymatgen Structure into a multi-channel cubic voxel grid.

    map_mode: 'fractional' (default) or 'cartesian'
      - fractional: use site.frac_coords and lattice lengths -> lattice-aware mapping
      - cartesian: use Cartesian bounding box centering (original behaviour)

    per_atom_gauss: if True use per-atom gaussian splatting using estimate_sigma_vox_from_element.
      This is slower but higher-fidelity than a single post-hoc gaussian_filter.
    """
    if elem_channels is None:
        elem_channels = DEFAULT_ELEM_CHANNELS

    s = build_supercell(struct, Lmin)

    # choose mapping mode
    if map_mode not in ("fractional", "cartesian"):
        raise ValueError("map_mode must be 'fractional' or 'cartesian'")

    if map_mode == "fractional":
        frac_coords = np.array([site.frac_coords for site in s.sites], dtype=np.float64)
        if frac_coords.shape[0] == 0:
            channels = ["total", "metal", "organic"] + list(elem_channels)
            if include_charge:
                channels.append("charge")
            vox0 = np.zeros((len(channels), grid, grid, grid), dtype=np.float32)
            return vox0, channels, float(Lmin)
        # box physical size uses largest lattice vector of the supercell
        lattice_lengths = np.array(s.lattice.abc, dtype=float)
        box_size_ang = float(max(lattice_lengths.max(), float(Lmin)))
        # fractional coords in [0,1) -> voxel coordinates
        uvw = frac_coords % 1.0
        vox_coords_f = uvw * (grid - 1.0)
    else:
        # cartesian bounding box approach (keeps original behavior as option)
        coords = np.array([site.coords for site in s.sites], dtype=np.float64)
        if coords.shape[0] == 0:
            channels = ["total", "metal", "organic"] + list(elem_channels)
            if include_charge:
                channels.append("charge")
            vox0 = np.zeros((len(channels), grid, grid, grid), dtype=np.float32)
            return vox0, channels, float(Lmin)
        mins = coords.min(axis=0)
        maxs = coords.max(axis=0)
        spans = maxs - mins
        box_size_ang = max(float(spans.max()), float(Lmin))
        center = (mins + maxs) / 2.0
        coords_centered = coords - center
        uvw = (coords_centered + box_size_ang / 2.0) / box_size_ang
        uvw = np.clip(uvw, 0.0, 1.0 - 1e-12)
        vox_coords_f = uvw * (grid - 1.0)

    channels = ["total", "metal", "organic"] + list(elem_channels)
    if include_charge:
        channels = channels + ["charge"]
    C = len(channels)
    ch_to_idx = {ch: i for i, ch in enumerate(channels)}

    vox = np.zeros((C, grid, grid, grid), dtype=np.float64)

    # iterate atoms and splat
    for i, site in enumerate(s.sites):
        x_f, y_f, z_f = vox_coords_f[i]
        sym, Z, occ = get_symbol_Z_and_occupancy(site)
        ent = [("total", 1.0), (("metal" if Z >= METAL_Z_CUTOFF else "organic"), 1.0)]
        symu = str(sym).capitalize()
        if symu in elem_channels:
            ent.append((symu, 1.0))
        for name, val in ent:
            idx = ch_to_idx[name]
            if per_atom_gauss:
                # per-atom gaussian splatting: amplitude scaled by val*occ
                sigma = estimate_sigma_vox_from_element(symu, grid, box_size_ang)
                _add_local_gaussian(vox, x_f, y_f, z_f, amplitude=val * occ, sigma=sigma, ch_idx=idx)
            else:
                if use_trilinear:
                    trilinear_splat(vox, x_f, y_f, z_f, val * occ, idx)
                else:
                    ix = int(round(x_f))
                    iy = int(round(y_f))
                    iz = int(round(z_f))
                    ix = min(max(ix, 0), grid - 1)
                    iy = min(max(iy, 0), grid - 1)
                    iz = min(max(iz, 0), grid - 1)
                    vox[idx, ix, iy, iz] += val * occ

        # partial charge channel: trilinear or per-atom gaussian to be consistent
        if include_charge:
            chv = extract_site_charge(site) * occ
            idxc = ch_to_idx["charge"]
            if per_atom_gauss:
                sigma = default_sigma_vox
                _add_local_gaussian(vox, x_f, y_f, z_f, amplitude=chv, sigma=sigma, ch_idx=idxc)
            else:
                if use_trilinear:
                    trilinear_splat(vox, x_f, y_f, z_f, chv, idxc)
                else:
                    ix = int(round(x_f))
                    iy = int(round(y_f))
                    iz = int(round(z_f))
                    ix = min(max(ix, 0), grid - 1)
                    iy = min(max(iy, 0), grid - 1)
                    iz = min(max(iz, 0), grid - 1)
                    vox[idxc, ix, iy, iz] += chv

    # optional gaussian smoothing per-channel using estimated per-atom sigmas
    if apply_gaussian and not per_atom_gauss:
        for ci in range(C):
            tmp = gaussian_filter(vox[ci].astype(np.float64), sigma=default_sigma_vox, mode="constant", cval=0.0)
            vox[ci] = tmp

    # convert to float32 and normalize according to chosen mode
    vox = vox.astype(np.float32)
    vox = normalize_voxels(vox, mode=normalize)

    sanity_checks(vox, channels)

    meta_box_size_ang = float(box_size_ang)
    return vox, channels, meta_box_size_ang


# ----------------------- Processing folder & IO -------------------------

def save_voxel_file(out_dir: Path, stem: str, vox: np.ndarray, channels: List[str], meta: dict):
    out_npz = out_dir / f"{stem}_vox.npz"
    meta_json = out_dir / f"{stem}_meta.json"
    # save npz with vox and channels
    # store channels as JSON-string array to be robust when loading with numpy
    np.savez_compressed(str(out_npz), vox=vox.astype(np.float32), channels=np.array(channels, dtype=object))
    # save readable metadata
    with open(str(meta_json), "w") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True)


def process_folder(
    cif_dir,
    out_dir,
    grid=64,
    Lmin=35.0,
    default_sigma_vox=0.75,
    elem_channels=None,
    include_charge=True,
    overwrite=False,
    normalize="per_channel_max",
    verbose=False,
    save_torch=False,
    map_mode: str = "fractional",
    per_atom_gauss: bool = False,
):
    setup_logger(verbose)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # case-insensitive CIF scanning
    p = Path(cif_dir)
    if not p.exists():
        logging.error("CIF directory not found: %s", cif_dir)
        return
    cifs = sorted([f for f in p.iterdir() if f.is_file() and f.suffix.lower() == ".cif"])
    if len(cifs) == 0:
        logging.warning("No .cif files found in %s", cif_dir)
        return

    summary = {"processed": 0, "skipped": 0, "failed": 0}

    for cif in tqdm(cifs, desc="Voxelizing CIFs"):
        stem = cif.stem
        out_npz = out_dir / f"{stem}_vox.npz"
        out_meta = out_dir / f"{stem}_meta.json"
        if not overwrite and out_npz.exists() and out_meta.exists():
            logging.info("Skipping existing: %s", stem)
            summary["skipped"] += 1
            continue
        try:
            s = Structure.from_file(str(cif))
        except Exception as e:
            logging.exception("Failed to read CIF %s: %s", cif, e)
            summary["failed"] += 1
            continue

        try:
            vox, channels, box_size_ang = voxelize_structure(
                s,
                grid=grid,
                elem_channels=elem_channels,
                Lmin=Lmin,
                default_sigma_vox=default_sigma_vox,
                normalize=normalize,
                use_trilinear=True,
                apply_gaussian=True,
                include_charge=include_charge,
                map_mode=map_mode,
                per_atom_gauss=per_atom_gauss,
            )
        except Exception as e:
            logging.exception("Voxelization failed for %s: %s", cif, e)
            summary["failed"] += 1
            continue

        meta = {
            "id": stem,
            "cif": str(cif),
            "channels": channels,
            "grid": grid,
            "Lmin": float(Lmin),
            "box_size_ang": box_size_ang,
            "normalize": normalize,
            "map_mode": map_mode,
            "per_atom_gauss": bool(per_atom_gauss),
            "timestamp_utc": datetime.utcnow().isoformat() + "Z",
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pymatgen": getattr(pymatgen, "__version__", "unknown"),
        }

        try:
            save_voxel_file(out_dir, stem, vox, channels, meta)
            if save_torch:
                try:
                    import torch as _torch

                    _torch.save({"vox": _torch.from_numpy(vox), "meta": meta}, str(out_dir / f"{stem}_vox.pt"))
                except Exception:
                    logging.exception("torch save failed for %s", stem)
        except Exception as e:
            logging.exception("Saving failed for %s: %s", stem, e)
            summary["failed"] += 1
            continue

        summary["processed"] += 1

    logging.info("Done. processed=%d skipped=%d failed=%d", summary["processed"], summary["skipped"], summary["failed"])


# --------------------------- CLI ---------------------------------------

def run_sanity_test():
    logging.info("Running quick sanity test...")
    from pymatgen.core import Lattice, Structure

    L = Lattice.cubic(10.0)
    s = Structure(L, ["C"], [[0.0, 0.0, 0.0]])
    vox, channels, box = voxelize_structure(s, grid=32, Lmin=10.0)
    print("vox shape:", vox.shape)
    print("channels:", channels)
    print("box:", box)
    print("max per channel:", [float(vox[i].max()) for i in range(vox.shape[0])])


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Publishable voxelizer: CIF -> multi-channel 3D voxels")
    p.add_argument("--cif-dir", default="repeat_cifs")
    p.add_argument("--out-dir", default="voxels_publishable")
    p.add_argument("--grid", type=int, default=64, help="voxel grid size (G). result shape (C,G,G,G)")
    p.add_argument("--Lmin", type=float, default=35.0, help="min linear cell size (Å) by supercell expansion")
    p.add_argument("--sigma", type=float, default=0.75, help="default gaussian blur sigma (in voxels)")
    p.add_argument("--include-charge", action="store_true", help="include partial charge channel if present")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--save-torch", action="store_true", help="also save a .pt torch file")
    p.add_argument(
        "--elem-channels",
        default=",".join(DEFAULT_ELEM_CHANNELS),
        help="comma-separated element channels to include (e.g. C,O,N,H)",
    )
    p.add_argument(
        "--normalize",
        choices=["none", "per_channel_max", "global_max", "sum_normalize"],
        default="per_channel_max",
        help="normalization mode applied after voxelization",
    )
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--map-mode", choices=["fractional", "cartesian"], default="fractional", help="mapping mode: fractional (lattice-aware) or cartesian bounding-box")
    p.add_argument("--per-atom-gauss", action="store_true", help="use per-atom gaussian splatting (slow, higher fidelity)")
    p.add_argument("--test", action="store_true", help="run a small sanity test and exit")
    args = p.parse_args()

    if args.test:
        setup_logger(args.verbose)
        run_sanity_test()
        sys.exit(0)

    ecs = [e.strip() for e in args.elem_channels.split(",") if e.strip()]

    process_folder(
        cif_dir=args.cif_dir,
        out_dir=args.out_dir,
        grid=args.grid,
        Lmin=args.Lmin,
        default_sigma_vox=args.sigma,
        elem_channels=ecs,
        include_charge=args.include_charge,
        overwrite=args.overwrite,
        normalize=args.normalize,
        verbose=args.verbose,
        save_torch=args.save_torch,
        map_mode=args.map_mode,
        per_atom_gauss=args.per_atom_gauss,
    )

