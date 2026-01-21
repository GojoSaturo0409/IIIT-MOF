import pandas as pd
import numpy as np
import json
import logging
import tempfile
from pathlib import Path
from sklearn.metrics import mean_absolute_error

from .utils import load_and_clean_csv
from .stats import compute_metrics, bootstrap_metric, compare_distributions, cohen_d
from .plotting import plot_regression_suite, plot_mae_thresholds, plot_feature_importance, plot_feature_boxplots
from .voxels import find_cif, run_voxelizer, extract_features_from_npz

def run_prediction_analysis(csv_path, out_dir, bootstrap_iters=1000):
    logging.info("Starting Prediction Analysis...")
    df = load_and_clean_csv(Path(csv_path))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Overall Metrics
    metrics = {}
    for label, sub in [("all", df), ("valid", df[df.is_valid])]:
        if sub.empty: continue
        m = compute_metrics(sub.target, sub.prediction)
        boot = bootstrap_metric(sub.target.values, sub.prediction.values, mean_absolute_error, iters=bootstrap_iters)
        m.update({'mae_ci_lower': boot['lower'], 'mae_ci_upper': boot['upper']})
        metrics[label] = m
        plot_regression_suite(sub.target, sub.prediction, out_dir / label, label)

    # 2. Threshold Analysis
    thresholds = [np.inf, 1.0, 0.5, 0.25]
    thr_data = []
    for t in thresholds:
        sub = df[df.abs_residual < t]
        thr_data.append({
            'threshold_label': f"<{t}" if t != np.inf else "All",
            'MAE': mean_absolute_error(sub.target, sub.prediction),
            'N': len(sub)
        })
    thr_df = pd.DataFrame(thr_data)
    plot_mae_thresholds(thr_df, out_dir / "threshold_analysis.png")

    # Save
    with open(out_dir / "metrics.json", "w") as f: json.dump(metrics, f, indent=2)
    thr_df.to_csv(out_dir / "thresholds.csv", index=False)
    logging.info(f"Prediction analysis complete. Results in {out_dir}")

def run_structural_analysis(csv_path, cif_root, voxel_script, out_dir, top_k=100, grid=64):
    logging.info("Starting Structural (Best vs Worst) Analysis...")
    df = load_and_clean_csv(Path(csv_path))
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Select Samples
    best = df.nsmallest(top_k, 'abs_residual')
    worst = df.nlargest(top_k, 'abs_residual')
    
    # 2. Process Voxelization
    feature_sets = {}
    for name, group_df in [('best', best), ('worst', worst)]:
        cifs = []
        for fn in group_df.filename:
            path = find_cif(fn, Path(cif_root))
            if path: cifs.append(path)
        
        logging.info(f"Found {len(cifs)}/{len(group_df)} CIFs for {name} group.")
        if not cifs: continue

        with tempfile.TemporaryDirectory() as tmp_cif_dir:
            tmp_path = Path(tmp_cif_dir)
            for c in cifs: shutil.copy(c, tmp_path)
            
            vox_out = out_dir / f"voxels_{name}"
            run_voxelizer(Path(voxel_script), tmp_path, vox_out, grid)
            
            feats = []
            for npz in vox_out.glob("*.npz"):
                f = extract_features_from_npz(npz)
                if f: feats.append(f)
            feature_sets[name] = pd.DataFrame(feats)

    if 'best' not in feature_sets or 'worst' not in feature_sets:
        logging.error("Could not extract features for both groups.")
        return

    # 3. Statistical Comparison
    best_df, worst_df = feature_sets['best'], feature_sets['worst']
    common_cols = [c for c in best_df.columns if c in worst_df.columns]
    
    stats_res = {}
    for col in common_cols:
        stats_res[col] = compare_distributions(best_df[col], worst_df[col])

    # 4. Visualization & Reporting
    plot_feature_importance(stats_res, out_dir)
    sorted_feats = sorted(stats_res.keys(), key=lambda k: abs(stats_res[k]['cohens_d']), reverse=True)
    plot_feature_boxplots(best_df, worst_df, sorted_feats, out_dir)
    
    pd.DataFrame(stats_res).T.to_csv(out_dir / "structural_stats.csv")
    logging.info(f"Structural analysis complete. Results in {out_dir}")
