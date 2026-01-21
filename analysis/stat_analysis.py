import os
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings


DEFAULT_CSV = "test_predictions.csv"
OUTPUT_DIR = Path("stat_analysis_output")
BOOTSTRAP_ITERS = 5000  
THRESHOLDS = [np.inf, 1.0, 0.5, 0.25]
THRESHOLD_LABELS = ["all", "resid_lt_1.0", "resid_lt_0.5", "resid_lt_0.25"]


BINS = [(0.0, 2.0), (2.0, 4.0), (4.0, np.inf)]
BIN_LABELS = ["0-2", "2-4", ">=4"]



def safe_float_series(s):
    
    return pd.to_numeric(s, errors="coerce")


def load_and_clean(csv_path):
    df = pd.read_csv(csv_path)
   
    df.columns = [c.strip() for c in df.columns]
   
    rename_map = {}
    col_lower = [c.lower() for c in df.columns]
    if "prediction" not in col_lower:
        for c in df.columns:
            if "pred" in c.lower():
                rename_map[c] = "prediction"
                break
    if "target" not in col_lower:
        for c in df.columns:
            if c.lower() in ("label", "truth", "y", "target_value"):
                rename_map[c] = "target"
                break
    if "is_valid" not in col_lower:
        for c in df.columns:
            if c.lower() in ("valid", "isvalid", "is_valid"):
                rename_map[c] = "is_valid"
                break
    if rename_map:
        df = df.rename(columns=rename_map)

   
    required = ["prediction", "target"]
    for r in required:
        if r not in df.columns:
            raise ValueError(f"CSV missing required column: {r}")

    
    df["prediction"] = safe_float_series(df["prediction"])
    df["target"] = safe_float_series(df["target"])
    if "is_valid" in df.columns:
       
        df["is_valid"] = df["is_valid"].astype(str).str.lower().map(
            {"true": True, "t": True, "1": True, "yes": True, "y": True,
             "false": False, "f": False, "0": False, "no": False, "n": False}
        ).fillna(df["is_valid"])  
        df["is_valid"] = df["is_valid"].apply(lambda x: bool(x) if pd.notna(x) else np.nan)
    else:
        
        df["is_valid"] = True

    
    before = len(df)
    df = df.dropna(subset=["prediction", "target"])
    after = len(df)
    if after < before:
        print(f"Dropped {before-after} rows with missing prediction/target")

  
    df = df.copy()
    df["residual"] = df["prediction"] - df["target"]
    df["abs_residual"] = np.abs(df["residual"])

    return df



def compute_metrics(y_true, y_pred):
    residuals = y_pred - y_true
    abs_res = np.abs(residuals)
    mae = np.mean(abs_res)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    median_ae = np.median(abs_res)
    bias = np.mean(residuals)  
    std_res = np.std(residuals, ddof=1) if len(residuals) > 1 else 0.0
  
    try:
        pearson_r, pearson_p = stats.pearsonr(y_true, y_pred) if len(y_true) > 1 else (np.nan, np.nan)
    except Exception:
        pearson_r, pearson_p = np.nan, np.nan
    try:
        spearman_r, spearman_p = stats.spearmanr(y_true, y_pred) if len(y_true) > 1 else (np.nan, np.nan)
    except Exception:
        spearman_r, spearman_p = np.nan, np.nan
    try:
        r2 = r2_score(y_true, y_pred) if len(y_true) > 1 else np.nan
    except Exception:
        r2 = np.nan
    
    denom = np.mean(np.abs(y_true)) if np.mean(np.abs(y_true)) != 0 else 1.0
    nmae = mae / denom

    return {
        "MAE": float(mae),
        "MSE": float(mse),
        "RMSE": float(rmse),
        "MedianAE": float(median_ae),
        "Bias": float(bias),
        "StdResidual": float(std_res),
        "PearsonR": float(pearson_r) if np.isfinite(pearson_r) else None,
        "Pearson_p": float(pearson_p) if np.isfinite(pearson_p) else None,
        "SpearmanR": float(spearman_r) if np.isfinite(spearman_r) else None,
        "Spearman_p": float(spearman_p) if np.isfinite(spearman_p) else None,
        "R2": float(r2) if np.isfinite(r2) else None,
        "NormalizedMAE": float(nmae),
    }


def bootstrap_metric(y_true, y_pred, metric_fn, iters=5000, seed=0):
   
    rng = np.random.default_rng(seed)
    n = len(y_true)
    boots = []
    if n == 0:
        return {"boot_mean": np.nan, "ci_lower": np.nan, "ci_upper": np.nan}
    for i in range(iters):
        idx = rng.integers(0, n, n)  # sample with replacement
        boots.append(metric_fn(y_true[idx], y_pred[idx]))
    boots = np.array(boots)
    lower = np.percentile(boots, 2.5)
    upper = np.percentile(boots, 97.5)
    return {"boot_mean": float(boots.mean()), "ci_lower": float(lower), "ci_upper": float(upper)}



def cohen_d(x, y):
   
    x = np.asarray(x)
    y = np.asarray(y)
    nx = len(x)
    ny = len(y)
    if nx < 2 or ny < 2:
        return np.nan
    mx = x.mean()
    my = y.mean()
    sx = x.std(ddof=1)
    sy = y.std(ddof=1)
    pooled = np.sqrt(((nx - 1) * sx ** 2 + (ny - 1) * sy ** 2) / (nx + ny - 2))
    if pooled == 0:
        return np.nan
    return (mx - my) / pooled



def build_thresholds_table(df, thresholds=THRESHOLDS, labels=THRESHOLD_LABELS, bootstrap_iters=BOOTSTRAP_ITERS):
    rows = []
    for thr, label in zip(thresholds, labels):
        if np.isinf(thr):
            sub = df.copy()
        else:
            sub = df[df["abs_residual"] < thr].copy()
        n = len(sub)
        if n == 0:
            metrics = {"N": 0}
        else:
            metrics = compute_metrics(sub["target"].values, sub["prediction"].values)
            
            boot = bootstrap_metric(sub["target"].values, sub["prediction"].values,
                                    lambda a,b: float(mean_absolute_error(a,b)), iters=bootstrap_iters)
            metrics["MAE_bootstrap_mean"] = boot["boot_mean"]
            metrics["MAE_CI_2.5%"] = boot["ci_lower"]
            metrics["MAE_CI_97.5%"] = boot["ci_upper"]
        metrics["N"] = int(n)
        metrics["threshold_label"] = label
        metrics["threshold_value"] = float(thr) if not np.isinf(thr) else None
        rows.append(metrics)
    df_table = pd.DataFrame(rows)
   
    cols = ["threshold_label", "threshold_value", "N", "MAE", "MAE_bootstrap_mean", "MAE_CI_2.5%", "MAE_CI_97.5%",
            "MedianAE", "MSE", "RMSE", "Bias", "StdResidual", "NormalizedMAE", "R2", "PearsonR", "Pearson_p", "SpearmanR", "Spearman_p"]
    cols_present = [c for c in cols if c in df_table.columns]
    df_table = df_table[cols_present]
    return df_table


def build_bin_stats_and_tests(df, bins=BINS, bin_labels=BIN_LABELS, bootstrap_iters=BOOTSTRAP_ITERS):
    bin_rows = []
    abs_res_by_bin = []
    for (low, high), label in zip(bins, bin_labels):
        if np.isinf(high):
            sub = df[(df["target"] >= low)].copy()
        else:
            sub = df[(df["target"] >= low) & (df["target"] < high)].copy()
        n = len(sub)
        if n == 0:
            metrics = {"bin_label": label, "N": 0}
        else:
            metrics = compute_metrics(sub["target"].values, sub["prediction"].values)
            boot = bootstrap_metric(sub["target"].values, sub["prediction"].values,
                                    lambda a,b: float(mean_absolute_error(a,b)), iters=bootstrap_iters)
            metrics["MAE_bootstrap_mean"] = boot["boot_mean"]
            metrics["MAE_CI_2.5%"] = boot["ci_lower"]
            metrics["MAE_CI_97.5%"] = boot["ci_upper"]
        metrics["bin_label"] = label
        metrics["bin_range"] = f"[{low},{high})" if not np.isinf(high) else f"[{low},+inf)"
        metrics["N"] = int(n)
        bin_rows.append(metrics)
        abs_res_by_bin.append(sub["abs_residual"].values)

    df_bins = pd.DataFrame(bin_rows)

   
    tests = []
    
    non_empty = [arr for arr in abs_res_by_bin if len(arr) > 0]
    if len(non_empty) >= 2:
        try:
            kw_stat, kw_p = stats.kruskal(*non_empty)
        except Exception:
            kw_stat, kw_p = np.nan, np.nan
    else:
        kw_stat, kw_p = np.nan, np.nan
    tests.append({"test": "kruskal_wallis", "statistic": kw_stat, "p_value": kw_p})

    
    from itertools import combinations
    pairwise_rows = []
    pairs = list(combinations(range(len(bin_labels)), 2))
    raw_pvals = []
    for i, j in pairs:
        xi = abs_res_by_bin[i]
        xj = abs_res_by_bin[j]
        if len(xi) < 2 or len(xj) < 2:
            U, p = np.nan, np.nan
        else:
            try:
                U, p = stats.mannwhitneyu(xi, xj, alternative='two-sided')
            except Exception:
                U, p = np.nan, np.nan
        d = cohen_d(xi, xj)
        pairwise_rows.append({
            "bin_i": bin_labels[i], "bin_j": bin_labels[j],
            "N_i": len(xi), "N_j": len(xj),
            "U_stat": U, "p_raw": p, "cohen_d": d
        })
        raw_pvals.append(p if p is not None else np.nan)

    
    m = len(pairwise_rows)
    for idx, row in enumerate(pairwise_rows):
        p_raw = row["p_raw"]
        if np.isfinite(p_raw):
            p_corr = min(p_raw * m, 1.0)
        else:
            p_corr = np.nan
        row["p_bonferroni"] = p_corr

    df_pairwise = pd.DataFrame(pairwise_rows)

    return df_bins, pd.DataFrame(tests), df_pairwise



def plot_scatter_pred_vs_true(y_true, y_pred, outpath, title="Prediction vs Target"):
    fig, ax = plt.subplots(figsize=(6,6))
    ax.scatter(y_true, y_pred, alpha=0.6, s=20)
 
    minv = min(np.nanmin(y_true), np.nanmin(y_pred))
    maxv = max(np.nanmax(y_true), np.nanmax(y_pred))
    ax.plot([minv, maxv], [minv, maxv], linestyle='--', linewidth=1)
    ax.set_xlabel("Target")
    ax.set_ylabel("Prediction")
    ax.set_title(title)
    ax.grid(True, linestyle=":", linewidth=0.5)
    ax.set_aspect('equal', adjustable='box')
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def plot_residuals_vs_target(y_true, y_pred, outpath, title="Residuals vs Target"):
    resid = y_pred - y_true
    fig, ax = plt.subplots(figsize=(6,4))
    ax.scatter(y_true, resid, alpha=0.6, s=20)
    ax.axhline(0, linestyle='--', linewidth=1)
    ax.set_xlabel("Target")
    ax.set_ylabel("Residual (Pred - Target)")
    ax.set_title(title)
    ax.grid(True, linestyle=":", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def plot_residual_hist_kde(y_true, y_pred, outpath, title="Residuals Distribution"):
    resid = y_pred - y_true
    fig, ax = plt.subplots(figsize=(6,4))
    ax.hist(resid, bins=40, alpha=0.8, density=True)
   
    try:
        kde = stats.gaussian_kde(resid)
        xs = np.linspace(np.nanpercentile(resid, 0.5), np.nanpercentile(resid, 99.5), 200)
        ax.plot(xs, kde(xs), linewidth=1.2)
    except Exception:
        pass
    ax.set_xlabel("Residual (Pred - Target)")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.grid(True, linestyle=":", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def plot_cumulative_abs_error(y_true, y_pred, outpath, title="Cumulative Absolute Error"):
    abs_res = np.abs(y_pred - y_true)
    sorted_abs = np.sort(abs_res)[::-1]  
    if len(sorted_abs) == 0:
        return
    cum = np.cumsum(sorted_abs)
    frac = np.arange(1, len(sorted_abs)+1) / len(sorted_abs)
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(frac, cum / cum[-1]) 
    ax.set_xlabel("Fraction of samples (top ...)")
    ax.set_ylabel("Cumulative fraction of total absolute error")
    ax.set_title(title)
    ax.grid(True, linestyle=":", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def plot_qq(residuals, outpath, title="QQ plot (Residuals)"):
    fig = plt.figure(figsize=(6,4))
    ax = fig.add_subplot(111)
    try:
        stats.probplot(residuals, dist="norm", plot=ax)
    except Exception:
        
        res = np.sort(residuals)
        theoretical = stats.norm.ppf(np.linspace(0.01, 0.99, len(res)))
        ax.plot(theoretical, res, marker='o', linestyle='')
        ax.plot([theoretical.min(), theoretical.max()], [theoretical.min(), theoretical.max()], linestyle='--')
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def plot_mae_vs_threshold(df_table, outpath):
    fig, ax = plt.subplots(figsize=(6,4))
    labels = df_table["threshold_label"].astype(str).tolist()
    maes = df_table["MAE"].fillna(0).tolist()
    ns = df_table["N"].tolist()
    ax.bar(range(len(maes)), maes)
    ax.set_xticks(range(len(maes)))
    ax.set_xticklabels([f"{lab}N={n}" for lab, n in zip(labels, ns)], rotation=30, ha='right')
    ax.set_ylabel("MAE")
    ax.set_title("MAE after removing large residuals (threshold filter)")
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def plot_bin_boxplots(df, bins=BINS, bin_labels=BIN_LABELS, df_in=None, outpath=None):
    if df_in is None:
        return
    data = []
    labels = []
    for (low, high), label in zip(bins, bin_labels):
        if np.isinf(high):
            sub = df_in[(df_in["target"] >= low)].copy()
        else:
            sub = df_in[(df_in["target"] >= low) & (df_in["target"] < high)].copy()
        data.append(sub["abs_residual"].values)
        labels.append(f"{label}N={len(sub)}")
    fig, ax = plt.subplots(figsize=(6,4))
    ax.boxplot(data, labels=labels, showfliers=False)
    ax.set_ylabel("Absolute residual")
    ax.set_title("Absolute residuals by target bin")
    fig.tight_layout()
    if outpath is not None:
        fig.savefig(outpath, dpi=200)
    plt.close(fig)


def analyze(df, output_dir=OUTPUT_DIR, bootstrap_iters=BOOTSTRAP_ITERS, top_k=10):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

   
    results = {}
    for label, subdf in [("all", df), ("valid", df[df["is_valid"]==True]), ("invalid", df[df["is_valid"]==False])]:
        if subdf.shape[0] == 0:
            continue
        y_true = subdf["target"].values
        y_pred = subdf["prediction"].values
        metrics = compute_metrics(y_true, y_pred)
       
        boot = bootstrap_metric(y_true, y_pred, lambda a,b: float(mean_absolute_error(a,b)), iters=bootstrap_iters)
        metrics["MAE_bootstrap_mean"] = boot["boot_mean"]
        metrics["MAE_CI_2.5%"] = boot["ci_lower"]
        metrics["MAE_CI_97.5%"] = boot["ci_upper"]
        results[label] = metrics

 
        prefix = output_dir / label
        prefix.mkdir(parents=True, exist_ok=True)
        plot_scatter_pred_vs_true(y_true, y_pred, prefix / "pred_vs_target.png",
                                  title=f"Prediction vs Target ({label}, N={len(y_true)})")
        plot_residuals_vs_target(y_true, y_pred, prefix / "residuals_vs_target.png", title=f"Residuals vs Target ({label})")
        plot_residual_hist_kde(y_true, y_pred, prefix / "residuals_hist_kde.png",title=f"Residuals Distribution ({label})")
        plot_cumulative_abs_error(y_true, y_pred, prefix / "cumulative_abs_error.png",
                                  title=f"Cumulative Absolute Error ({label})")
        plot_qq(y_pred - y_true, prefix / "residuals_qq.png", title=f"QQ plot resid ({label})")

      
        subdf = subdf.assign(residual=(subdf["prediction"] - subdf["target"]).abs(),
                             signed_residual=(subdf["prediction"] - subdf["target"]))
        worst = subdf.sort_values("residual", ascending=False).head(top_k)
        worst_file = prefix / f"top_{top_k}_worst.csv"
        worst.to_csv(worst_file, index=False)
        metrics["top_k_worst_csv"] = str(worst_file)


    with open(output_dir / "metrics_summary.json", "w") as fh:
        json.dump(results, fh, indent=2)

  
    df_full = df.copy()
    df_full["residual"] = df_full["prediction"] - df_full["target"]
    df_full["abs_residual"] = df_full["residual"].abs()
    df_full.to_csv(output_dir / "predictions_with_residuals.csv", index=False)


    thr_table = build_thresholds_table(df_full, thresholds=THRESHOLDS, labels=THRESHOLD_LABELS, bootstrap_iters=bootstrap_iters)
    thr_table_file = output_dir / "thresholds_table.csv"
    thr_table.to_csv(thr_table_file, index=False)
   
    thr_table_md = thr_table.to_markdown(index=False)
    with open(output_dir / "thresholds_table.md", "w") as fh:
        fh.write("# Thresholds summary\n\n")
        fh.write(thr_table_md)

  
    try:
        plot_mae_vs_threshold(thr_table, output_dir / "mae_vs_threshold.png")
    except Exception:
        warnings.warn("failed to create mae_vs_threshold plot")

  
    df_bins, df_tests_global, df_pairwise = build_bin_stats_and_tests(df_full, bins=BINS, bin_labels=BIN_LABELS, bootstrap_iters=bootstrap_iters)
    df_bins.to_csv(output_dir / "bin_stats.csv", index=False)
    df_tests_global.to_csv(output_dir / "bin_tests_global.csv", index=False)
    df_pairwise.to_csv(output_dir / "bin_pairwise_tests.csv", index=False)


    try:
        plot_bin_boxplots(df=df_full, outpath=output_dir / "bins_abs_residual_boxplot.png")
    except Exception:
        warnings.warn("failed to create bin boxplot")


    print("\n=== Summary metrics (saved to analysis_output/metrics_summary.json) ===")
    for k,v in results.items():
        print(f"--- {k} (N = {len(df if k=='all' else df[df['is_valid']==(k=='valid')])}) ---")
        print(f"MAE = {v['MAE']:.6g} (bootstrap 95% CI: [{v['MAE_CI_2.5%']:.6g}, {v['MAE_CI_97.5%']:.6g}])")
        print(f"RMSE = {v['RMSE']:.6g}, R2 = {v.get('R2', None)}")
        print(f"Bias = {v['Bias']:.6g}, StdResid = {v['StdResidual']:.6g}")
        print(f"Pearson R = {v.get('PearsonR')} (p={v.get('Pearson_p')}), Spearman R = {v.get('SpearmanR')}")

    print(f"\nThresholds table saved to: {thr_table_file.resolve()}")
    print(f"Bin stats saved to: { (output_dir / 'bin_stats.csv').resolve() }")
    print(f"Pairwise bin tests saved to: { (output_dir / 'bin_pairwise_tests.csv').resolve() }")
    print(f"Plots and CSV outputs saved to: {output_dir.resolve()}")

    return results, thr_table, df_bins, df_tests_global, df_pairwise



def required_percent_reduction(current_mae, desired_mae):
   
    if desired_mae >= current_mae:
        return 0.0
    return 100.0 * (current_mae - desired_mae) / current_mae


def parse_args():
    p = argparse.ArgumentParser(description="Enhanced analyze test_predictions.csv for errors/MAE analysis")
    p.add_argument("--csv", type=str, default=DEFAULT_CSV, help="Path to CSV with columns: filename,prediction,target,is_valid")
    p.add_argument("--bootstrap-iters", type=int, default=BOOTSTRAP_ITERS, help="Bootstrap iterations for CI")
    p.add_argument("--top-k", type=int, default=10, help="Save top-K worst examples")
    p.add_argument("--desired-mae", type=float, default=None, help="(optional) desired MAE: compute percent reduction required")
    p.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR), help="Directory to save outputs")
    return p.parse_args()


def main():
    args = parse_args()
    csv_path = Path(args.csv)
    outdir = Path(args.output_dir)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = load_and_clean(csv_path)
    global BOOTSTRAP_ITERS
    BOOTSTRAP_ITERS = args.bootstrap_iters
    results, thr_table, df_bins, df_tests_global, df_pairwise = analyze(df, output_dir=outdir, bootstrap_iters=args.bootstrap_iters, top_k=args.top_k)

    
    if args.desired_mae is not None:
        print("\n=== Percent reduction required to reach desired MAE ===")
        for key in ["valid", "all"]:
            if key in results:
                curr = results[key]["MAE"]
                pct = required_percent_reduction(curr, args.desired_mae)
                print(f"{key}: current MAE = {curr:.6g}; desired = {args.desired_mae:.6g}; reduction required = {pct:.2f}%")

if __name__ == "__main__":
    main()
