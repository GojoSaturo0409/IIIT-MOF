#!/usr/bin/env python3

from __future__ import annotations

import logging
import random
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.ndimage import zoom
from tqdm import tqdm

warnings.filterwarnings("ignore")

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
    print("[WARN] pymatgen not found. CIF-based metrics will be skipped.")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[WARN] matplotlib not found. Plots will be skipped.")

VOXEL_DIRS = {
    32: Path("voxels_32_subset"),
    64: Path("voxels_64_subset"),
    96: Path("voxels_96_subset"),
}
CIF_DIR = Path("subset_cifs")
RESULTS_DIR = Path("results")

N_SAMPLE = 200
RANDOM_SEED = 42
REFERENCE_GRID = 96
GRIDS = [32, 64, 96]

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def normalize_channel_name(name) -> str:
    if isinstance(name, bytes):
        name = name.decode("utf-8", errors="ignore")
    return str(name).strip().lower()


def load_voxel(path: Path) -> Tuple[Optional[np.ndarray], Optional[List[str]], Optional[dict]]:
    try:
        data = np.load(str(path), allow_pickle=True)
        vox = data["vox"].astype(np.float32)
        channels_raw = list(data["channels"])
        channels = [normalize_channel_name(c) for c in channels_raw]
    except Exception as e:
        log.debug("Failed to load voxel file %s: %s", path, e)
        return None, None, None
    return vox, channels, {}


def channel_index(channels: List[str], name: str) -> Optional[int]:
    name = normalize_channel_name(name)
    try:
        return [normalize_channel_name(c) for c in channels].index(name)
    except ValueError:
        return None


def trilinear_downsample(vox: np.ndarray, target_grid: int) -> np.ndarray:
    c = vox.shape[0]
    factor = target_grid / vox.shape[1]
    out = np.zeros((c, target_grid, target_grid, target_grid), dtype=np.float32)
    for i in range(c):
        out[i] = zoom(vox[i], factor, order=1)
    return out


def effective_resolution(meta: dict, grid: int) -> float:
    box = meta.get("box_size_ang", None)
    if box is not None:
        try:
            return float(box) / float(grid)
        except Exception:
            pass
    return np.nan


def build_cif_occupancy_mask(cif_path: Path, grid: int) -> Optional[np.ndarray]:
    if not HAS_PYMATGEN or not cif_path.exists():
        return None
    try:
        s = Structure.from_file(str(cif_path))
        lattice = s.lattice
        lin = np.linspace(0, 1, grid, endpoint=False)
        gx, gy, gz = np.meshgrid(lin, lin, lin, indexing="ij")
        grid_frac = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)

        occupied = np.zeros(grid ** 3, dtype=bool)
        for site in s.sites:
            try:
                el = Element(str(site.specie.symbol))
                r = float(getattr(el, "vdw_radius", None) or getattr(el, "atomic_radius", 1.5) or 1.5)
            except Exception:
                r = 1.5
            diff = grid_frac - site.frac_coords[None, :]
            diff = diff - np.round(diff)
            cart_diff = diff @ lattice.matrix
            dist = np.linalg.norm(cart_diff, axis=1)
            occupied |= (dist < r)

        return occupied.reshape(grid, grid, grid).astype(np.float32)
    except Exception as e:
        log.debug("Failed to build CIF mask for %s: %s", cif_path, e)
        return None


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


def compute_ssim_psnr_vs_cif(cif_mask: np.ndarray, vox: np.ndarray, channels: List[str]) -> Tuple[float, float]:
    idx = channel_index(channels, "total")
    if idx is None or not HAS_SKIMAGE:
        return np.nan, np.nan

    total = vox[idx].astype(np.float64)
    ref = cif_mask.astype(np.float64)

    if ref.shape != total.shape:
        return np.nan, np.nan

    data_range = max(ref.max(), total.max()) - min(ref.min(), total.min())
    if data_range <= 0:
        data_range = 1.0

    try:
        s = ssim(ref, total, data_range=data_range)
    except Exception:
        s = np.nan

    mse = np.mean((ref - total) ** 2)
    psnr = 10.0 * np.log10(data_range ** 2 / (mse + 1e-12)) if mse > 0 else 100.0

    return float(s), float(psnr)


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


def mi_vs_cif(cif_mask: np.ndarray, vox: np.ndarray, channels: List[str]) -> float:
    idx = channel_index(channels, "total")
    if idx is None:
        return np.nan
    total = vox[idx]
    if cif_mask.shape != total.shape:
        return np.nan
    return mutual_information_3d(
        np.clip(cif_mask, 0, 1),
        np.clip(total / (total.max() + 1e-12), 0, 1),
    )


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


def run_analysis(stems: List[str]) -> pd.DataFrame:
    records = []

    for stem in tqdm(stems, desc="Analysing structures"):
        cif_path = CIF_DIR / f"{stem}.cif"

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

        cif_masks = {g: build_cif_occupancy_mask(cif_path, g) for g in GRIDS}

        ref96 = voxels[96]
        refs_from96 = {
            96: ref96,
            64: trilinear_downsample(ref96, 64),
            32: trilinear_downsample(ref96, 32),
        }

        for g in GRIDS:
            vox = voxels[g]
            ch = channels_map[g]
            meta = metas[g]
            cif_mask = cif_masks[g]

            rec = {"stem": stem, "grid": g}

            rec["eff_res_ang"] = effective_resolution(meta, g)

            if cif_mask is not None:
                rec["ssim_vs_cif"], rec["psnr_vs_cif"] = compute_ssim_psnr_vs_cif(cif_mask, vox, ch)
                rec["mi_vs_cif"] = mi_vs_cif(cif_mask, vox, ch)
            else:
                rec["ssim_vs_cif"] = np.nan
                rec["psnr_vs_cif"] = np.nan
                rec["mi_vs_cif"] = np.nan

            if g < REFERENCE_GRID:
                rec["ssim_vs96"], rec["psnr_vs96"] = compute_ssim_psnr(refs_from96[g], vox)
                idx_total = channel_index(ch, "total")
                if idx_total is not None:
                    rec["mi_vs96"] = mutual_information_3d(
                        np.clip(refs_from96[g][idx_total], 0, 1),
                        np.clip(vox[idx_total], 0, 1),
                    )
                else:
                    rec["mi_vs96"] = np.nan
            else:
                rec["ssim_vs96"] = 1.0
                rec["psnr_vs96"] = 100.0
                rec["mi_vs96"] = np.nan

            records.append(rec)

    df = pd.DataFrame(records)
    RESULTS_DIR.mkdir(exist_ok=True)
    df.to_csv(RESULTS_DIR / "fidelity_summary.csv", index=False)
    log.info("Saved fidelity_summary.csv (%d rows)", len(df))
    return df


COLORS = {32: "#e74c3c", 64: "#3498db", 96: "#2ecc71"}
LABELS = {32: "32³", 64: "64³", 96: "96³"}


def bar_with_err(ax, grids, df, col):
    vals, errs, lbls, clrs = [], [], [], []
    for g in grids:
        sub = df[df["grid"] == g][col].dropna()
        if len(sub) == 0:
            continue
        vals.append(sub.mean())
        errs.append(sub.std())
        lbls.append(LABELS[g])
        clrs.append(COLORS[g])
    if vals:
        ax.bar(lbls, vals, yerr=errs, color=clrs, capsize=5, alpha=0.85)


def make_plots(df: pd.DataFrame):
    if not HAS_MPL:
        return

    plot_dir = RESULTS_DIR / "plots"
    plot_dir.mkdir(exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Signal Fidelity Metrics", fontsize=15, fontweight="bold")

    ax = axes[0, 0]
    bar_with_err(ax, GRIDS, df, "ssim_vs_cif")
    ax.set_title("SSIM vs CIF")
    ax.set_ylabel("SSIM")

    ax = axes[0, 1]
    bar_with_err(ax, GRIDS, df, "psnr_vs_cif")
    ax.set_title("PSNR vs CIF (dB)")
    ax.set_ylabel("PSNR (dB)")

    ax = axes[0, 2]
    bar_with_err(ax, GRIDS, df, "mi_vs_cif")
    ax.set_title("MI vs CIF (nats)")
    ax.set_ylabel("Mutual Information")

    ax = axes[1, 0]
    bar_with_err(ax, [32, 64], df, "ssim_vs96")
    ax.set_title("SSIM vs 96³")
    ax.set_ylabel("SSIM")

    ax = axes[1, 1]
    bar_with_err(ax, [32, 64], df, "psnr_vs96")
    ax.set_title("PSNR vs 96³ (dB)")
    ax.set_ylabel("PSNR (dB)")

    ax = axes[1, 2]
    bar_with_err(ax, [32, 64], df, "mi_vs96")
    ax.set_title("MI vs 96³ (nats)")
    ax.set_ylabel("Mutual Information")

    plt.tight_layout()
    fig.savefig(plot_dir / "signal_fidelity.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle("Effective Resolution Distribution", fontsize=13, fontweight="bold")
    for g in GRIDS:
        sub = df[df["grid"] == g]["eff_res_ang"].dropna()
        if len(sub) > 0:
            ax.hist(sub, bins=30, alpha=0.6, color=COLORS[g],
                    label=f"{LABELS[g]}: {sub.mean():.2f}±{sub.std():.2f} Å/vox", density=True)
    ax.set_xlabel("Å / voxel")
    ax.set_ylabel("Density")
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(plot_dir / "effective_resolution.png", dpi=150)
    plt.close(fig)

    log.info("Plots saved to %s", plot_dir)


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

    lines.append("\n── Signal Fidelity vs CIF Ground Truth ─────────────────────────────")
    for g in GRIDS:
        ssim_s = df[df["grid"] == g]["ssim_vs_cif"].dropna()
        psnr_s = df[df["grid"] == g]["psnr_vs_cif"].dropna()
        mi_s   = df[df["grid"] == g]["mi_vs_cif"].dropna()
        if len(ssim_s) > 0:
            lines.append(
                f"  {g}³:  SSIM={ssim_s.mean():.4f} ± {ssim_s.std():.4f}  "
                f"PSNR={psnr_s.mean():.2f} dB  "
                f"MI={mi_s.mean():.4f}"
            )

    lines.append("\n── Signal Fidelity vs 96³ Reference ───────────────────────────────")
    for g in [32, 64]:
        ssim_s = df[df["grid"] == g]["ssim_vs96"].dropna()
        psnr_s = df[df["grid"] == g]["psnr_vs96"].dropna()
        mi_s   = df[df["grid"] == g]["mi_vs96"].dropna()
        if len(ssim_s) > 0:
            lines.append(
                f"  {g}³:  SSIM={ssim_s.mean():.3f} ± {ssim_s.std():.3f}  "
                f"PSNR={psnr_s.mean():.1f} dB  "
                f"MI={mi_s.mean():.3f}"
            )

    lines.append("\n── Interpretation ──────────────────────────────────────────────────")
    ssim32 = df[df["grid"] == 32]["ssim_vs96"].mean()
    ssim64 = df[df["grid"] == 64]["ssim_vs96"].mean()
    psnr32 = df[df["grid"] == 32]["psnr_vs96"].mean()
    psnr64 = df[df["grid"] == 64]["psnr_vs96"].mean()
    mi32   = df[df["grid"] == 32]["mi_vs96"].mean()
    mi64   = df[df["grid"] == 64]["mi_vs96"].mean()
    lines.append(f"  SSIM gain 32→64 (vs 96³): {ssim64 - ssim32:+.3f}")
    lines.append(f"  PSNR gain 32→64 (vs 96³): {psnr64 - psnr32:+.1f} dB")
    lines.append(f"  MI   gain 32→64 (vs 96³): {mi64 - mi32:+.3f}")
    lines.append("=" * 70)

    report_path = RESULTS_DIR / "fidelity_report.txt"
    with open(report_path, "w") as fh:
        fh.write("\n".join(lines))

    print("\n".join(lines))
    log.info("Report saved to %s", report_path)


def main():
    log.info("Starting resolution fidelity analysis")
    stems = find_common_stems(N_SAMPLE, RANDOM_SEED)
    df = run_analysis(stems)
    make_plots(df)
    write_report(df)
    log.info("All done. Results in: %s", RESULTS_DIR)


if __name__ == "__main__":
    main()