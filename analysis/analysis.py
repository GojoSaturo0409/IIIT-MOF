from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import StandardScaler


DEFAULT_TOP_K = 100
DEFAULT_OUTDIR = "analysis_output"



def setup_logger(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )



def safe_float_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")



def load_and_clean(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    required = ["filename", "prediction", "target"]
    for r in required:
        if r not in df.columns:
            raise ValueError(f"CSV missing required column: {r}")

    df["prediction"] = safe_float_series(df["prediction"])
    df["target"] = safe_float_series(df["target"])

    before = len(df)
    df = df.dropna(subset=["prediction", "target"]).reset_index(drop=True)
    after = len(df)

    if after < before:
        logging.warning("Dropped %d rows with missing prediction/target", before - after)

    df["filename"] = df["filename"].astype(str).str.strip()
    return df



def _strip_voxel_suffix(name: str) -> str:
    s = str(name)
    suffixes = ["_vox.npz", "_vox.pt", "_vox.npy", "_vox", ".npz", ".npy", ".pt"]
    s_lower = s.lower()
    for suf in suffixes:
        if s_lower.endswith(suf):
            return s[:len(s) - len(suf)]
    return Path(s).stem



def find_cif_for_entry(filename_value: str, cif_root: Path, repeat_subdir: str = "repeat_cifs") -> Optional[Path]:
    cand = str(filename_value).strip()
    if cand == "":
        return None

    p = Path(cand)
    if p.exists() and p.is_file() and p.suffix.lower() == ".cif":
        return p.resolve()

    stem = _strip_voxel_suffix(p.name)
    search_root = cif_root / repeat_subdir if (cif_root / repeat_subdir).exists() else cif_root

    matches = []
    for fp in search_root.rglob("*.cif"):
        if fp.stem.lower() == stem.lower():
            return fp.resolve()
        if stem.lower() in fp.name.lower():
            matches.append(fp)

    if len(matches) == 1:
        return matches[0].resolve()
    if len(matches) > 1:
        for m in matches:
            if m.name.lower().startswith(stem.lower()):
                return m.resolve()
        return matches[0].resolve()

    return None



def copy_cifs_to_temp(found_cifs: List[Path], tmpdir: Path) -> List[Path]:
    tmpdir.mkdir(parents=True, exist_ok=True)
    copied = []
    for cif in found_cifs:
        try:
            dest = tmpdir / cif.name
            if not dest.exists():
                shutil.copy2(str(cif), str(dest))
            copied.append(dest.resolve())
        except Exception as e:
            logging.warning("Failed to copy %s: %s", cif, e)
    return copied



def run_voxelizer_on_folder(voxel_script: Path, cif_folder: Path, out_dir: Path, grid: int = 64, extra_args: List[str] = None) -> Tuple[bool, str]:
    extra_args = extra_args or []

    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("voxelizer_module", str(voxel_script))
        voxel_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(voxel_mod)

        if hasattr(voxel_mod, "process_folder"):
            logging.info("Using import-based voxelizer.process_folder()")
            voxel_mod.process_folder(
                cif_dir=str(cif_folder),
                out_dir=str(out_dir),
                grid=grid,
                Lmin=35.0,
                default_sigma_vox=0.75,
                elem_channels="C,O,N,H",
                include_charge=True,
                overwrite=False,
                normalize="per_channel_max",
                verbose=False,
                save_torch=False,
                map_mode="fractional",
                per_atom_gauss=False,
            )
            return True, "Voxelizer completed successfully"
    except Exception as e:
        logging.debug("Import-based voxelizer failed: %s", e)

    cmd = [
        sys.executable,
        str(voxel_script),
        "--cif-dir",
        str(cif_folder),
        "--out-dir",
        str(out_dir),
        "--grid",
        str(grid),
        "--Lmin",
        "35.0",
        "--normalize",
        "per_channel_max",
        "--include-charge",
    ]
    cmd.extend(extra_args)

    logging.info("Running voxelizer subprocess")
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=3600)
        out = proc.stdout + "\n" + proc.stderr
        if proc.returncode == 0:
            return True, out
        else:
            return False, out
    except Exception as e:
        return False, f"Subprocess failed: {e}"



def extract_voxel_features_enhanced(npz_path: Path) -> Dict[str, float]:
    try:
        data = np.load(npz_path, allow_pickle=True)

        vox = data['vox']
        channels_raw = data['channels']
        channels = [str(ch) for ch in channels_raw]

        if vox.ndim != 4:
            logging.warning("Unexpected voxel shape %s for %s", vox.shape, npz_path)
            return {}

        C, G, G_y, G_z = vox.shape
        if not (G == G_y == G_z):
            logging.warning("Non-cubic grid: %s", vox.shape)
            return {}

        features = {}
        ch_to_idx = {ch: i for i, ch in enumerate(channels)}

        features['grid_size'] = float(G)
        features['n_channels'] = float(C)
        features['global_density_mean'] = float(np.mean(vox))
        features['global_density_std'] = float(np.std(vox))
        features['global_density_max'] = float(np.max(vox))
        features['global_sparsity'] = float(np.mean(vox == 0))

        if 'total' in ch_to_idx:
            total_ch = vox[ch_to_idx['total']]
            features['total_occupancy'] = float(np.sum(total_ch))
            features['total_density_mean'] = float(np.mean(total_ch))
            features['total_density_max'] = float(np.max(total_ch))
            features['total_sparsity'] = float(np.mean(total_ch == 0))

        metal_idx = ch_to_idx.get('metal')
        organic_idx = ch_to_idx.get('organic')

        if metal_idx is not None and organic_idx is not None:
            metal_ch = vox[metal_idx]
            organic_ch = vox[organic_idx]

            total_occ = np.sum(metal_ch) + np.sum(organic_ch)

            features['metal_occupancy'] = float(np.sum(metal_ch))
            features['organic_occupancy'] = float(np.sum(organic_ch))
            features['metal_fraction'] = float(np.sum(metal_ch) / (total_occ + 1e-10))
            features['organic_fraction'] = float(np.sum(organic_ch) / (total_occ + 1e-10))

            features['metal_density_mean'] = float(np.mean(metal_ch))
            features['organic_density_mean'] = float(np.mean(organic_ch))
            features['metal_sparsity'] = float(np.mean(metal_ch == 0))
            features['organic_sparsity'] = float(np.mean(organic_ch == 0))

            metal_nonzero = metal_ch > 0
            organic_nonzero = organic_ch > 0
            overlap = np.sum(metal_nonzero & organic_nonzero)
            union = np.sum(metal_nonzero | organic_nonzero)
            features['metal_organic_colocalization'] = float(overlap / (union + 1e-10))

        element_channels = ['C', 'O', 'N', 'H']
        elem_occupancies = {}

        for elem in element_channels:
            if elem in ch_to_idx:
                elem_ch = vox[ch_to_idx[elem]]
                occ = float(np.sum(elem_ch))
                elem_occupancies[elem] = occ

                features[f'{elem}_occupancy'] = occ
                features[f'{elem}_density_mean'] = float(np.mean(elem_ch))
                features[f'{elem}_density_max'] = float(np.max(elem_ch))
                features[f'{elem}_density_std'] = float(np.std(elem_ch))
                features[f'{elem}_sparsity'] = float(np.mean(elem_ch == 0))

        if elem_occupancies:
            total_elem_occ = sum(elem_occupancies.values())
            if total_elem_occ > 0:
                occupancy_frac = {k: v / total_elem_occ for k, v in elem_occupancies.items()}

                entropy_elem = -sum(p * np.log(p + 1e-10) for p in occupancy_frac.values() if p > 0)
                features['element_diversity_entropy'] = float(entropy_elem)

                max_elem_frac = max(occupancy_frac.values())
                features['dominant_element_fraction'] = float(max_elem_frac)

                herfindahl = sum(p**2 for p in occupancy_frac.values())
                features['element_concentration_herfindahl'] = float(herfindahl)

        for ch_name, ch_idx in ch_to_idx.items():
            if ch_name in ('total', 'metal', 'organic', 'C', 'O', 'N', 'H'):
                ch_data = vox[ch_idx]
                total_density = np.sum(ch_data)

                if total_density > 0:
                    coords = np.indices(ch_data.shape)
                    com_x = float(np.sum(coords[0] * ch_data) / total_density)
                    com_y = float(np.sum(coords[1] * ch_data) / total_density)
                    com_z = float(np.sum(coords[2] * ch_data) / total_density)

                    features[f'{ch_name}_com_x'] = com_x / G
                    features[f'{ch_name}_com_y'] = com_y / G
                    features[f'{ch_name}_com_z'] = com_z / G

                    dist_from_center = np.sqrt((com_x - G / 2)**2 + (com_y - G / 2)**2 + (com_z - G / 2)**2)
                    max_possible_dist = (G * np.sqrt(3) / 2)
                    features[f'{ch_name}_com_distance_from_center'] = float(dist_from_center / max_possible_dist)

                    sq_dists = (coords[0] - com_x)**2 + (coords[1] - com_y)**2 + (coords[2] - com_z)**2
                    rg = np.sqrt(np.sum(sq_dists * ch_data) / total_density)
                    features[f'{ch_name}_radius_of_gyration'] = float(rg / G)

                    nonzero_vals = ch_data[ch_data > 0]
                    if len(nonzero_vals) > 0:
                        probs = nonzero_vals / np.sum(nonzero_vals)
                        dist_entropy = -np.sum(probs * np.log(probs + 1e-10))
                        features[f'{ch_name}_distribution_entropy'] = float(dist_entropy)

        for ch_name in ('total', 'metal', 'organic'):
            if ch_name in ch_to_idx:
                ch_data = vox[ch_to_idx[ch_name]]
                gradients = []
                for ax in range(3):
                    grad = np.gradient(ch_data, axis=ax)
                    gradients.append(float(np.std(grad)))
                features[f'{ch_name}_gradient_std'] = float(np.mean(gradients))

        if 'charge' in ch_to_idx:
            charge_ch = vox[ch_to_idx['charge']]
            features['charge_occupancy'] = float(np.sum(charge_ch))
            features['charge_mean'] = float(np.mean(charge_ch))
            features['charge_std'] = float(np.std(charge_ch))
            features['charge_max'] = float(np.max(charge_ch))
            features['charge_min'] = float(np.min(charge_ch))

            nonzero_charges = charge_ch[charge_ch != 0]
            if len(nonzero_charges) > 0:
                net_charge = np.sum(charge_ch)
                total_abs_charge = np.sum(np.abs(charge_ch))
                features['charge_net_to_total_ratio'] = float(net_charge / (total_abs_charge + 1e-10))

        elem_channels_present = [e for e in element_channels if e in ch_to_idx]
        for i, elem1 in enumerate(elem_channels_present):
            for elem2 in elem_channels_present[i + 1:]:
                ch1 = vox[ch_to_idx[elem1]].flatten()
                ch2 = vox[ch_to_idx[elem2]].flatten()
                corr = float(np.corrcoef(ch1, ch2)[0, 1])
                if not np.isnan(corr):
                    features[f'{elem1}_{elem2}_correlation'] = corr

        features['void_fraction'] = float(np.mean(vox == 0))

        if 'total' in ch_to_idx:
            empty_mask = vox[ch_to_idx['total']] == 0
            features['contiguous_empty_voxels'] = float(np.sum(empty_mask))

        if 'total' in ch_to_idx:
            total_ch = vox[ch_to_idx['total']]
            for axis, axis_name in enumerate(['x', 'y', 'z']):
                other_axes = tuple(i for i in [0, 1, 2] if i != axis)
                projection = np.sum(total_ch, axis=other_axes)
                proj_std = float(np.std(projection))
                proj_mean = float(np.mean(projection) + 1e-10)
                features[f'layering_ratio_{axis_name}'] = float(proj_std / proj_mean)

        return features

    except Exception as e:
        logging.error("Failed to extract features from %s: %s", npz_path, e)
        return {}



def extract_features_from_voxels(voxel_dir: Path, stem_to_npz: Dict[str, Path]) -> pd.DataFrame:
    results = []

    for stem, npz_path in stem_to_npz.items():
        if not npz_path.exists():
            logging.warning("NPZ file not found: %s", npz_path)
            continue

        features = extract_voxel_features_enhanced(npz_path)
        if features:
            features['stem'] = stem
            results.append(features)

    if not results:
        logging.error("No voxel features extracted!")
        return pd.DataFrame()

    return pd.DataFrame(results).fillna(0)



def compute_feature_statistics(best_features: pd.DataFrame, worst_features: pd.DataFrame) -> Dict:
    stats_results = {}

    feature_cols = [c for c in best_features.columns if c != 'stem']

    for feat in feature_cols:
        try:
            best_vals = best_features[feat].dropna()
            worst_vals = worst_features[feat].dropna()

            if len(best_vals) == 0 or len(worst_vals) == 0:
                continue

            t_stat, p_val = stats.ttest_ind(best_vals, worst_vals)

            pooled_std = np.sqrt((np.std(best_vals)**2 + np.std(worst_vals)**2) / 2)
            cohens_d = (np.mean(best_vals) - np.mean(worst_vals)) / (pooled_std + 1e-10)

            u_stat, u_pval = stats.mannwhitneyu(best_vals, worst_vals)

            stats_results[feat] = {
                'best_mean': float(np.mean(best_vals)),
                'best_std': float(np.std(best_vals)),
                'worst_mean': float(np.mean(worst_vals)),
                'worst_std': float(np.std(worst_vals)),
                'mean_difference': float(np.mean(best_vals) - np.mean(worst_vals)),
                't_stat': float(t_stat),
                'p_value': float(p_val),
                'cohens_d': float(cohens_d),
                'mann_whitney_u': float(u_stat),
                'mann_whitney_p': float(u_pval),
                'abs_cohens_d': float(abs(cohens_d)),
            }
        except Exception as e:
            logging.debug(f"Error computing stats for {feat}: {e}")

    return stats_results



def plot_feature_comparison(best_features: pd.DataFrame, worst_features: pd.DataFrame, stats_results: Dict, out_dir: Path, top_n: int = 25):
    sorted_feats = sorted(stats_results.items(), key=lambda x: abs(x[1]['cohens_d']), reverse=True)[:top_n]

    fig, ax = plt.subplots(figsize=(12, 8))
    feats_names = [f[0] for f in sorted_feats]
    cohens_ds = [f[1]['cohens_d'] for f in sorted_feats]
    colors = ['green' if d > 0 else 'red' for d in cohens_ds]

    y_pos = np.arange(len(feats_names))
    ax.barh(y_pos, cohens_ds, color=colors, alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feats_names, fontsize=9)
    ax.axvline(0, color='black', linestyle='-', linewidth=0.8)
    ax.set_xlabel("Cohen's d (effect size)", fontsize=11)
    ax.set_title(f"Top {top_n} Discriminating Features", fontsize=12)
    ax.grid(True, alpha=0.3, axis='x')
    fig.tight_layout()
    fig.savefig(out_dir / 'feature_effect_sizes.png', dpi=200, bbox_inches='tight')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 8))
    p_vals = [max(f[1]['p_value'], 1e-10) for f in sorted_feats]

    y_pos = np.arange(len(feats_names))
    ax.barh(y_pos, [-np.log10(p) for p in p_vals], color='steelblue', alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feats_names, fontsize=9)
    ax.axvline(-np.log10(0.05), color='red', linestyle='--', linewidth=1)
    ax.set_xlabel("-log10(p-value)", fontsize=11)
    ax.set_title(f"Statistical Significance of Top {top_n} Features", fontsize=12)
    ax.grid(True, alpha=0.3, axis='x')
    fig.tight_layout()
    fig.savefig(out_dir / 'feature_pvalues.png', dpi=200, bbox_inches='tight')
    plt.close(fig)

    top_n_box = min(9, len(sorted_feats))
    top_box_feats = [f[0] for f in sorted_feats[:top_n_box]]
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    axes = axes.flatten()

    for idx, feat in enumerate(top_box_feats):
        ax = axes[idx]
        data_to_plot = [best_features[feat].dropna(), worst_features[feat].dropna()]
        bp = ax.boxplot(data_to_plot, labels=['Best', 'Worst'], patch_artist=True)

        for patch, color in zip(bp['boxes'], ['lightgreen', 'lightcoral']):
            patch.set_facecolor(color)

        ax.set_ylabel(feat, fontsize=10)
        p_val = stats_results[feat]['p_value']
        sig = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else "ns"))
        ax.set_title(f"{feat}\n(d={stats_results[feat]['cohens_d']:.2f}, {sig})", fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')

    for idx in range(top_n_box, 9):
        axes[idx].axis('off')

    fig.tight_layout()
    fig.savefig(out_dir / 'feature_distributions_top9.png', dpi=200, bbox_inches='tight')
    plt.close(fig)

    categories = {
        'Global Density': ['global_density_mean', 'global_density_std', 'global_sparsity'],
        'Metal/Organic': ['metal_fraction', 'organic_fraction', 'metal_organic_colocalization'],
        'Element Diversity': ['element_diversity_entropy', 'dominant_element_fraction'],
        'Spatial Order': ['layering_ratio_x', 'layering_ratio_y', 'layering_ratio_z'],
        'Element Occupancy': ['C_occupancy', 'O_occupancy', 'N_occupancy', 'H_occupancy'],
    }

    fig, ax = plt.subplots(figsize=(12, 6))
    category_means = []
    category_names = []

    for cat_name, cat_feats in categories.items():
        cat_d_vals = []
        for feat in cat_feats:
            if feat in stats_results:
                cat_d_vals.append(abs(stats_results[feat]['cohens_d']))
        if cat_d_vals:
            category_means.append(np.mean(cat_d_vals))
            category_names.append(cat_name)

    colors_cat = plt.cm.Set2(np.linspace(0, 1, len(category_names)))
    ax.bar(category_names, category_means, color=colors_cat, alpha=0.8, edgecolor='black')
    ax.set_ylabel("Mean |Cohen's d|", fontsize=11)
    ax.set_title("Feature Importance by Category", fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    fig.tight_layout()
    fig.savefig(out_dir / 'feature_categories.png', dpi=200, bbox_inches='tight')
    plt.close(fig)



def analyze_best_worst(csv_path: Path, cif_root: Path, repeat_subdir: str, voxel_script: Path, out_dir: Path, top_k: int = DEFAULT_TOP_K, grid: int = 64, verbose: bool = False):
    setup_logger(verbose)
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.info("=" * 80)
    logging.info("BEST vs WORST PERFORMERS STRUCTURAL ANALYSIS")
    logging.info("=" * 80)

    logging.info("Loading predictions from %s", csv_path)
    df = load_and_clean(csv_path)
    logging.info("Loaded %d samples", len(df))

    logging.info("Identifying top %d best and worst performers", top_k)
    df['residual'] = df['prediction'] - df['target']
    df['abs_residual'] = df['residual'].abs()

    best_df = df.nsmallest(top_k, 'abs_residual').copy()
    worst_df = df.nlargest(top_k, 'abs_residual').copy()

    selected_dir = out_dir / "selected_samples"
    selected_dir.mkdir(parents=True, exist_ok=True)
    best_df.to_csv(selected_dir / f"top_{top_k}_best.csv", index=False)
    worst_df.to_csv(selected_dir / f"top_{top_k}_worst.csv", index=False)

    best_cifs = []
    best_stems = []
    for _, row in best_df.iterrows():
        cif_path = find_cif_for_entry(row['filename'], cif_root, repeat_subdir)
        if cif_path:
            best_cifs.append(cif_path)
            best_stems.append(cif_path.stem)

    worst_cifs = []
    worst_stems = []
    for _, row in worst_df.iterrows():
        cif_path = find_cif_for_entry(row['filename'], cif_root, repeat_subdir)
        if cif_path:
            worst_cifs.append(cif_path)
            worst_stems.append(cif_path.stem)

    if len(best_cifs) == 0 and len(worst_cifs) == 0:
        logging.error("No CIF files found")
        return

    best_voxel_dir = out_dir / "voxels_best"
    worst_voxel_dir = out_dir / "voxels_worst"

    if len(best_cifs) > 0:
        temp_dir_best = Path(tempfile.mkdtemp(prefix="best_cifs_"))
        copy_cifs_to_temp(best_cifs, temp_dir_best)
        ok, msg = run_voxelizer_on_folder(voxel_script, temp_dir_best, best_voxel_dir, grid=grid)
        (best_voxel_dir / "voxelizer.log").write_text(msg)
        if not ok:
            logging.warning("Voxelization warnings for best set")

    if len(worst_cifs) > 0:
        temp_dir_worst = Path(tempfile.mkdtemp(prefix="worst_cifs_"))
        copy_cifs_to_temp(worst_cifs, temp_dir_worst)
        ok, msg = run_voxelizer_on_folder(voxel_script, temp_dir_worst, worst_voxel_dir, grid=grid)
        (worst_voxel_dir / "voxelizer.log").write_text(msg)
        if not ok:
            logging.warning("Voxelization warnings for worst set")

    best_stem_to_npz = {}
    for stem in best_stems:
        npz_files = list(best_voxel_dir.glob(f"{stem}_vox.npz"))
        if npz_files:
            best_stem_to_npz[stem] = npz_files[0]

    worst_stem_to_npz = {}
    for stem in worst_stems:
        npz_files = list(worst_voxel_dir.glob(f"{stem}_vox.npz"))
        if npz_files:
            worst_stem_to_npz[stem] = npz_files[0]

    best_features = extract_features_from_voxels(best_voxel_dir, best_stem_to_npz) if best_stem_to_npz else pd.DataFrame()
    worst_features = extract_features_from_voxels(worst_voxel_dir, worst_stem_to_npz) if worst_stem_to_npz else pd.DataFrame()

    stats_results = compute_feature_statistics(best_features, worst_features)

    analysis_dir = out_dir / "analysis_results"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    try:
        plot_feature_comparison(best_features, worst_features, stats_results, analysis_dir, top_n=25)
    except Exception as e:
        logging.warning("Plot generation failed: %s", e)

    with open(analysis_dir / 'feature_statistics.json', 'w') as f:
        json.dump(stats_results, f, indent=2)

    best_features.to_csv(analysis_dir / 'best_performer_features.csv', index=False)
    worst_features.to_csv(analysis_dir / 'worst_performer_features.csv', index=False)



def parse_args():
    p = argparse.ArgumentParser(description="Analyze structural features of best vs worst performing MOFs")
    p.add_argument("--csv", type=str, required=True)
    p.add_argument("--cif-root", type=str, required=True)
    p.add_argument("--repeat-subdir", type=str, default="repeat_cifs")
    p.add_argument("--voxel-script", type=str, default="voxel.py")
    p.add_argument("--out-dir", type=str, default=DEFAULT_OUTDIR)
    p.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    p.add_argument("--grid", type=int, default=64)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    setup_logger(args.verbose)

    csv_path = Path(args.csv).expanduser().resolve()
    cif_root = Path(args.cif_root).expanduser().resolve()
    voxel_script = Path(args.voxel_script).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()

    if not csv_path.exists():
        logging.error("Predictions CSV not found: %s", csv_path)
        sys.exit(1)
    if not cif_root.exists():
        logging.error("CIF root not found: %s", cif_root)
        sys.exit(1)

    analyze_best_worst(
        csv_path=csv_path,
        cif_root=cif_root,
        repeat_subdir=args.repeat_subdir,
        voxel_script=voxel_script,
        out_dir=out_dir,
        top_k=args.top_k,
        grid=args.grid,
        verbose=args.verbose,
    )
