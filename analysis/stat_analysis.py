import argparse
import json
import warnings
from itertools import combinations
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from scipy import stats
from scipy.stats import gaussian_kde
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


DEFAULT_CSV = "test_predictions.csv"
OUTPUT_DIR = Path("stat_analysis_output")
BOOTSTRAP_ITERS = 5000

THRESHOLDS = [np.inf, 1.0, 0.5, 0.25]
THRESHOLD_LABELS = ["all", "resid_lt_1.0", "resid_lt_0.5", "resid_lt_0.25"]

BINS = [(0.0, 2.0), (2.0, 4.0), (4.0, np.inf)]
BIN_LABELS = ["0-2", "2-4", ">=4"]


def safe_float_series(s):
    return pd.to_numeric(s, errors="coerce")


def parse_bool_series(s: pd.Series) -> pd.Series:
    true_set = {"true", "t", "1", "yes", "y"}
    false_set = {"false", "f", "0", "no", "n"}

    def _parse(x):
        if pd.isna(x):
            return np.nan
        if isinstance(x, bool):
            return x
        sx = str(x).strip().lower()
        if sx in true_set:
            return True
        if sx in false_set:
            return False
        return np.nan

    return s.apply(_parse)


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
        df["is_valid"] = parse_bool_series(df["is_valid"])
    else:
        df["is_valid"] = True

    before = len(df)
    df = df.dropna(subset=["prediction", "target"]).copy()
    after = len(df)
    if after < before:
        print(f"Dropped {before - after} rows with missing prediction/target")

    df["residual"] = df["prediction"] - df["target"]
    df["abs_residual"] = np.abs(df["residual"])

    return df


def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    residuals = y_pred - y_true
    abs_res = np.abs(residuals)

    mae = float(np.mean(abs_res)) if len(abs_res) else np.nan
    mse = float(mean_squared_error(y_true, y_pred)) if len(y_true) else np.nan
    rmse = float(np.sqrt(mse)) if np.isfinite(mse) else np.nan
    median_ae = float(np.median(abs_res)) if len(abs_res) else np.nan
    bias = float(np.mean(residuals)) if len(residuals) else np.nan
    std_res = float(np.std(residuals, ddof=1)) if len(residuals) > 1 else 0.0

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

    denom = np.mean(np.abs(y_true)) if len(y_true) and np.mean(np.abs(y_true)) != 0 else 1.0
    nmae = mae / denom if np.isfinite(mae) else np.nan

    return {
        "MAE": float(mae) if np.isfinite(mae) else None,
        "MSE": float(mse) if np.isfinite(mse) else None,
        "RMSE": float(rmse) if np.isfinite(rmse) else None,
        "MedianAE": float(median_ae) if np.isfinite(median_ae) else None,
        "Bias": float(bias) if np.isfinite(bias) else None,
        "StdResidual": float(std_res) if np.isfinite(std_res) else None,
        "PearsonR": float(pearson_r) if np.isfinite(pearson_r) else None,
        "Pearson_p": float(pearson_p) if np.isfinite(pearson_p) else None,
        "SpearmanR": float(spearman_r) if np.isfinite(spearman_r) else None,
        "Spearman_p": float(spearman_p) if np.isfinite(spearman_p) else None,
        "R2": float(r2) if np.isfinite(r2) else None,
        "NormalizedMAE": float(nmae) if np.isfinite(nmae) else None,
    }


def bootstrap_metric(y_true, y_pred, metric_fn, iters=5000, seed=0):
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)

    if n == 0:
        return {"boot_mean": np.nan, "ci_lower": np.nan, "ci_upper": np.nan}

    boots = []
    for _ in range(iters):
        idx = rng.integers(0, n, n)
        boots.append(metric_fn(y_true[idx], y_pred[idx]))

    boots = np.asarray(boots, dtype=float)
    lower = np.percentile(boots, 2.5)
    upper = np.percentile(boots, 97.5)
    return {"boot_mean": float(np.mean(boots)), "ci_lower": float(lower), "ci_upper": float(upper)}


def cohen_d(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    nx = len(x)
    ny = len(y)

    if nx < 2 or ny < 2:
        return np.nan

    mx = np.mean(x)
    my = np.mean(y)
    sx = np.std(x, ddof=1)
    sy = np.std(y, ddof=1)

    pooled = np.sqrt(((nx - 1) * sx ** 2 + (ny - 1) * sy ** 2) / (nx + ny - 2))
    if pooled == 0:
        return np.nan
    return float((mx - my) / pooled)


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
            boot = bootstrap_metric(
                sub["target"].values,
                sub["prediction"].values,
                lambda a, b: float(mean_absolute_error(a, b)),
                iters=bootstrap_iters,
            )
            metrics["MAE_bootstrap_mean"] = boot["boot_mean"]
            metrics["MAE_CI_2.5%"] = boot["ci_lower"]
            metrics["MAE_CI_97.5%"] = boot["ci_upper"]

        metrics["N"] = int(n)
        metrics["threshold_label"] = label
        metrics["threshold_value"] = float(thr) if not np.isinf(thr) else None
        rows.append(metrics)

    df_table = pd.DataFrame(rows)

    cols = [
        "threshold_label", "threshold_value", "N", "MAE", "MAE_bootstrap_mean",
        "MAE_CI_2.5%", "MAE_CI_97.5%", "MedianAE", "MSE", "RMSE", "Bias",
        "StdResidual", "NormalizedMAE", "R2", "PearsonR", "Pearson_p",
        "SpearmanR", "Spearman_p"
    ]
    cols_present = [c for c in cols if c in df_table.columns]
    return df_table[cols_present]


def build_bin_stats_and_tests(df, bins=BINS, bin_labels=BIN_LABELS, bootstrap_iters=BOOTSTRAP_ITERS):
    bin_rows = []
    abs_res_by_bin = []

    for (low, high), label in zip(bins, bin_labels):
        if np.isinf(high):
            sub = df[df["target"] >= low].copy()
        else:
            sub = df[(df["target"] >= low) & (df["target"] < high)].copy()

        n = len(sub)
        if n == 0:
            metrics = {"bin_label": label, "N": 0}
        else:
            metrics = compute_metrics(sub["target"].values, sub["prediction"].values)
            boot = bootstrap_metric(
                sub["target"].values,
                sub["prediction"].values,
                lambda a, b: float(mean_absolute_error(a, b)),
                iters=bootstrap_iters,
            )
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
    df_tests_global = pd.DataFrame(tests)

    pairwise_rows = []
    pairs = list(combinations(range(len(bin_labels)), 2))
    m = len(pairs)

    for i, j in pairs:
        xi = abs_res_by_bin[i]
        xj = abs_res_by_bin[j]

        if len(xi) < 2 or len(xj) < 2:
            U, p = np.nan, np.nan
        else:
            try:
                U, p = stats.mannwhitneyu(xi, xj, alternative="two-sided")
            except Exception:
                U, p = np.nan, np.nan

        d = cohen_d(xi, xj)
        p_bonf = min(p * m, 1.0) if np.isfinite(p) else np.nan

        pairwise_rows.append({
            "bin_i": bin_labels[i],
            "bin_j": bin_labels[j],
            "N_i": len(xi),
            "N_j": len(xj),
            "U_stat": U,
            "p_raw": p,
            "p_bonferroni": p_bonf,
            "cohen_d": d,
        })

    df_pairwise = pd.DataFrame(pairwise_rows)
    return df_bins, df_tests_global, df_pairwise


def safe_corrs(y_true, y_pred):
    pearson_r, pearson_p = np.nan, np.nan
    spearman_r, spearman_p = np.nan, np.nan

    try:
        if len(y_true) > 1 and np.std(y_true) > 0 and np.std(y_pred) > 0:
            pearson_r, pearson_p = stats.pearsonr(y_true, y_pred)
            spearman_r, spearman_p = stats.spearmanr(y_true, y_pred)
    except Exception:
        pass

    return pearson_r, pearson_p, spearman_r, spearman_p


def plot_density_parity(y_true, y_pred, outpath, title_prefix="", n_bins=80, rolling_window=200):
    """
    Density-based parity plot that avoids overplotting.
    Panel (a): 2D histogram parity plot with log-scaled density.
    Panel (b): Rolling-window density map, sorted by target.
    Panel (c): Residual distribution with histogram + KDE.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    finite_mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[finite_mask]
    y_pred = y_pred[finite_mask]

    if len(y_true) == 0:
        raise ValueError("No finite prediction/target values available for plotting.")

    res = y_pred - y_true
    mae = float(np.mean(np.abs(res))) if len(res) else np.nan
    rmse = float(np.sqrt(np.mean(res**2))) if len(res) else np.nan
    r2 = float(r2_score(y_true, y_pred)) if len(y_true) > 1 else np.nan
    pr, _, sp, _ = safe_corrs(y_true, y_pred)

    lim_lo = 0.0
    lim_hi = float(np.nanpercentile(np.concatenate([y_true, y_pred]), 99.5)) * 1.05
    lim_hi = max(lim_hi, 0.5)
    LIMS = (lim_lo, lim_hi)

    fig = plt.figure(figsize=(13, 5.2), dpi=180)
    fig.patch.set_facecolor("white")
    gs = gridspec.GridSpec(
        1, 3, figure=fig,
        left=0.07, right=0.97, bottom=0.13, top=0.91, wspace=0.40
    )
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])

    # (a) 2D histogram parity plot
    h, xe, ye = np.histogram2d(y_true, y_pred, bins=n_bins, range=[LIMS, LIMS])
    h = h.T

    cmap1 = plt.cm.YlOrRd.copy()
    cmap1.set_bad("white")

    if np.max(h) > 0:
        hm = np.ma.masked_where(h == 0, h)
        im1 = ax1.pcolormesh(
            xe, ye, hm,
            norm=LogNorm(vmin=1, vmax=max(int(h.max()), 1)),
            cmap=cmap1,
            rasterized=True,
            shading="auto",
        )
        cb1 = fig.colorbar(im1, ax=ax1, pad=0.02, fraction=0.046)
        cb1.set_label("Count (log scale)", fontsize=8)
        cb1.ax.tick_params(labelsize=7)

    ax1.plot(LIMS, LIMS, "k--", lw=1.2, label="Identity")
    ax1.set_xlim(LIMS)
    ax1.set_ylim(LIMS)
    ax1.set_xlabel("Target working capacity (mmol g$^{-1}$)", fontsize=9)
    ax1.set_ylabel("Predicted working capacity (mmol g$^{-1}$)", fontsize=9)
    ax1.set_title(f"({title_prefix}a) Parity — 2D density histogram", fontsize=9, fontweight="bold")
    ax1.tick_params(labelsize=8)
    ax1.grid(True, lw=0.4, alpha=0.4)

    stats_str = (
        f"$R^2$={r2:.4f}\n"
        f"MAE={mae:.4f} mmol g$^{{-1}}$\n"
        f"RMSE={rmse:.4f} mmol g$^{{-1}}$\n"
        f"Pearson $r$={pr:.4f}\n"
        f"Spearman $\\rho$={sp:.4f}"
    )
    ax1.text(
        0.03, 0.97, stats_str,
        transform=ax1.transAxes,
        fontsize=7.5,
        va="top",
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#bbb", alpha=0.92),
    )

    # (b) Rolling-window density map
    order = np.argsort(y_true)
    t_s = y_true[order]
    p_s = y_pred[order]
    n_tot = len(t_s)

    n_cols = max(1, int(np.ceil(n_tot / rolling_window)))
    p_edges = np.linspace(LIMS[0], LIMS[1], n_bins + 1)
    Z = np.zeros((n_bins, n_cols))
    t_centres = np.zeros(n_cols)
    t_med = np.zeros(n_cols)
    t_q25 = np.zeros(n_cols)
    t_q75 = np.zeros(n_cols)

    for i in range(n_cols):
        start = i * rolling_window
        stop = min((i + 1) * rolling_window, n_tot)
        sl_p = p_s[start:stop]
        sl_t = t_s[start:stop]

        if len(sl_t) == 0:
            t_centres[i] = np.nan
            t_med[i] = np.nan
            t_q25[i] = np.nan
            t_q75[i] = np.nan
            continue

        counts, _ = np.histogram(sl_p, bins=p_edges)
        Z[:, i] = counts
        t_centres[i] = float(np.mean(sl_t))
        t_med[i] = float(np.median(sl_p))
        t_q25[i] = float(np.percentile(sl_p, 25))
        t_q75[i] = float(np.percentile(sl_p, 75))

    valid_cols = np.isfinite(t_centres)
    t_centres = t_centres[valid_cols]
    t_med = t_med[valid_cols]
    t_q25 = t_q25[valid_cols]
    t_q75 = t_q75[valid_cols]
    Z = Z[:, valid_cols]
    n_cols = Z.shape[1]

    if n_cols == 1:
        width = max((LIMS[1] - LIMS[0]) * 0.05, 1e-6)
        col_edges = np.array([t_centres[0] - width, t_centres[0] + width])
    else:
        col_edges = np.empty(n_cols + 1)
        col_edges[1:-1] = 0.5 * (t_centres[:-1] + t_centres[1:])
        first_gap = t_centres[1] - t_centres[0]
        last_gap = t_centres[-1] - t_centres[-2]
        col_edges[0] = t_centres[0] - first_gap / 2
        col_edges[-1] = t_centres[-1] + last_gap / 2

    Zm = np.ma.masked_where(Z == 0, Z)
    cmap2 = plt.cm.Blues.copy()
    cmap2.set_bad("white")

    if np.max(Z) > 0:
        im2 = ax2.pcolormesh(
            col_edges, p_edges, Zm,
            norm=LogNorm(vmin=1, vmax=max(int(Z.max()), 1)),
            cmap=cmap2,
            rasterized=True,
            shading="auto",
        )
        cb2 = fig.colorbar(im2, ax=ax2, pad=0.02, fraction=0.046)
        cb2.set_label("Count (log scale)", fontsize=8)
        cb2.ax.tick_params(labelsize=7)

    ax2.plot(LIMS, LIMS, "k--", lw=1.2, label="Identity")
    if len(t_centres) > 0:
        ax2.plot(t_centres, t_med, "r-", lw=1.5, label="Median")
        ax2.fill_between(t_centres, t_q25, t_q75, color="red", alpha=0.15, label="IQR")
    ax2.set_xlim(LIMS)
    ax2.set_ylim(LIMS)
    ax2.set_xlabel("Target working capacity (mmol g$^{-1}$)", fontsize=9)
    ax2.set_ylabel("Predicted working capacity (mmol g$^{-1}$)", fontsize=9)
    ax2.set_title(f"({title_prefix}b) Rolling density\n(sorted by target value)", fontsize=9, fontweight="bold")
    ax2.tick_params(labelsize=8)
    ax2.grid(True, lw=0.4, alpha=0.4)
    ax2.legend(fontsize=7.5, loc="upper left")

    # (c) residual distribution
    clip_lo = float(np.nanpercentile(res, 0.5))
    clip_hi = float(np.nanpercentile(res, 99.5))
    if clip_hi <= clip_lo:
        clip_lo -= 1.0
        clip_hi += 1.0

    res_clip = np.clip(res, clip_lo, clip_hi)
    ax3.hist(
        res_clip, bins=100, color="steelblue", alpha=0.75,
        edgecolor="none", density=True, label="All structures"
    )

    try:
        kde_x = np.linspace(clip_lo, clip_hi, 400)
        kde_fn = gaussian_kde(res_clip, bw_method=0.15)
        ax3.plot(kde_x, kde_fn(kde_x), color="navy", lw=1.8, label="KDE")
    except Exception:
        pass

    ax3.axvline(0, color="black", lw=1.0, ls="--")
    if np.isfinite(mae):
        ax3.axvline(mae, color="crimson", lw=1.0, ls=":", label=f"+MAE ({mae:.3f})")
        ax3.axvline(-mae, color="crimson", lw=1.0, ls=":")

    tail_thr = 1.0
    if clip_hi > tail_thr:
        ax3.axvspan(tail_thr, clip_hi, alpha=0.10, color="orange", label=f"|res|>{tail_thr} mmol g$^{{-1}}$")
    if clip_lo < -tail_thr:
        ax3.axvspan(clip_lo, -tail_thr, alpha=0.10, color="orange")

    pct_out = 100.0 * float(np.mean(np.abs(res) > tail_thr))
    ax3.text(
        0.97, 0.97, f"{pct_out:.1f}% with\n|res|>{tail_thr}",
        transform=ax3.transAxes, fontsize=7.5, va="top", ha="right",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#bbb", alpha=0.9),
    )

    ax3.set_xlabel("Residual: prediction $-$ target (mmol g$^{-1}$)", fontsize=9)
    ax3.set_ylabel("Probability density", fontsize=9)
    ax3.set_title(f"({title_prefix}c) Residual distribution", fontsize=9, fontweight="bold")
    ax3.tick_params(labelsize=8)
    ax3.grid(True, lw=0.4, alpha=0.4)
    ax3.legend(fontsize=7.5)

    fig.savefig(outpath, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_residuals_vs_target(y_true, y_pred, outpath, title="Residuals vs Target"):
    resid = y_pred - y_true
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(y_true, resid, alpha=0.6, s=20)
    ax.axhline(0, linestyle="--", linewidth=1)
    ax.set_xlabel("Target")
    ax.set_ylabel("Residual (Pred - Target)")
    ax.set_title(title)
    ax.grid(True, linestyle=":", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def plot_residual_hist_kde(y_true, y_pred, outpath, title="Residuals Distribution"):
    resid = y_pred - y_true
    fig, ax = plt.subplots(figsize=(6, 4))
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
    frac = np.arange(1, len(sorted_abs) + 1) / len(sorted_abs)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(frac, cum / cum[-1])
    ax.set_xlabel("Fraction of samples (top ...)")
    ax.set_ylabel("Cumulative fraction of total absolute error")
    ax.set_title(title)
    ax.grid(True, linestyle=":", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def plot_qq(residuals, outpath, title="QQ plot (Residuals)"):
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)

    try:
        stats.probplot(residuals, dist="norm", plot=ax)
    except Exception:
        res = np.sort(residuals)
        theoretical = stats.norm.ppf(np.linspace(0.01, 0.99, len(res)))
        ax.plot(theoretical, res, marker="o", linestyle="")
        ax.plot(
            [theoretical.min(), theoretical.max()],
            [theoretical.min(), theoretical.max()],
            linestyle="--",
        )

    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def plot_mae_vs_threshold(df_table, outpath):
    if df_table is None or len(df_table) == 0:
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    labels = df_table["threshold_label"].astype(str).tolist()
    maes = df_table["MAE"].fillna(0).tolist()
    ns = df_table["N"].tolist()

    ax.bar(range(len(maes)), maes)
    ax.set_xticks(range(len(maes)))
    ax.set_xticklabels([f"{lab}\nN={n}" for lab, n in zip(labels, ns)], rotation=0, ha="center")
    ax.set_ylabel("MAE")
    ax.set_title("MAE after removing large residuals (threshold filter)")
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def plot_bin_boxplots(df_in=None, bins=BINS, bin_labels=BIN_LABELS, outpath=None):
    if df_in is None or len(df_in) == 0:
        return

    data = []
    labels = []
    for (low, high), label in zip(bins, bin_labels):
        if np.isinf(high):
            sub = df_in[df_in["target"] >= low].copy()
        else:
            sub = df_in[(df_in["target"] >= low) & (df_in["target"] < high)].copy()

        if len(sub) == 0:
            continue

        data.append(sub["abs_residual"].values)
        labels.append(f"{label}\nN={len(sub)}")

    if len(data) == 0:
        return

    fig, ax = plt.subplots(figsize=(6, 4))
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

    for label, subdf in [
        ("all", df),
        ("valid", df[df["is_valid"] == True]),
        ("invalid", df[df["is_valid"] == False]),
    ]:
        if subdf.shape[0] == 0:
            continue

        y_true = subdf["target"].values
        y_pred = subdf["prediction"].values

        metrics = compute_metrics(y_true, y_pred)
        boot = bootstrap_metric(
            y_true,
            y_pred,
            lambda a, b: float(mean_absolute_error(a, b)),
            iters=bootstrap_iters,
        )
        metrics["MAE_bootstrap_mean"] = boot["boot_mean"]
        metrics["MAE_CI_2.5%"] = boot["ci_lower"]
        metrics["MAE_CI_97.5%"] = boot["ci_upper"]
        results[label] = metrics

        prefix = output_dir / label
        prefix.mkdir(parents=True, exist_ok=True)

        plot_density_parity(
            y_true, y_pred,
            outpath=prefix / "pred_vs_target_density.png",
            title_prefix=f"{label} — ",
        )

        plot_residuals_vs_target(
            y_true, y_pred,
            prefix / "residuals_vs_target.png",
            title=f"Residuals vs Target ({label})",
        )
        plot_residual_hist_kde(
            y_true, y_pred,
            prefix / "residuals_hist_kde.png",
            title=f"Residuals Distribution ({label})",
        )
        plot_cumulative_abs_error(
            y_true, y_pred,
            prefix / "cumulative_abs_error.png",
            title=f"Cumulative Absolute Error ({label})",
        )
        plot_qq(
            y_pred - y_true,
            prefix / "residuals_qq.png",
            title=f"QQ plot resid ({label})",
        )

        subdf = subdf.assign(
            residual=(subdf["prediction"] - subdf["target"]).abs(),
            signed_residual=(subdf["prediction"] - subdf["target"]),
        )
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

    thr_table = build_thresholds_table(
        df_full,
        thresholds=THRESHOLDS,
        labels=THRESHOLD_LABELS,
        bootstrap_iters=bootstrap_iters,
    )
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

    df_bins, df_tests_global, df_pairwise = build_bin_stats_and_tests(
        df_full,
        bins=BINS,
        bin_labels=BIN_LABELS,
        bootstrap_iters=bootstrap_iters,
    )
    df_bins.to_csv(output_dir / "bin_stats.csv", index=False)
    df_tests_global.to_csv(output_dir / "bin_tests_global.csv", index=False)
    df_pairwise.to_csv(output_dir / "bin_pairwise_tests.csv", index=False)

    try:
        plot_bin_boxplots(df=df_full, outpath=output_dir / "bins_abs_residual_boxplot.png")
    except Exception:
        warnings.warn("failed to create bin boxplot")

    print("\n=== Summary metrics (saved to metrics_summary.json) ===")
    for k, v in results.items():
        n_subset = len(df if k == "all" else df[df["is_valid"] == (k == "valid")])
        print(f"--- {k} (N = {n_subset}) ---")
        print(f"MAE = {v['MAE']:.6g} (bootstrap 95% CI: [{v['MAE_CI_2.5%']:.6g}, {v['MAE_CI_97.5%']:.6g}])")
        print(f"RMSE = {v['RMSE']:.6g}, R2 = {v.get('R2', None)}")
        print(f"Bias = {v['Bias']:.6g}, StdResid = {v['StdResidual']:.6g}")
        print(f"Pearson R = {v.get('PearsonR')} (p={v.get('Pearson_p')}), Spearman R = {v.get('SpearmanR')}")

    print(f"\nThresholds table saved to: {thr_table_file.resolve()}")
    print(f"Bin stats saved to: {(output_dir / 'bin_stats.csv').resolve()}")
    print(f"Pairwise bin tests saved to: {(output_dir / 'bin_pairwise_tests.csv').resolve()}")
    print(f"Plots and CSV outputs saved to: {output_dir.resolve()}")
    print(f"\nDensity parity figures saved to: <output_dir>/<subset>/pred_vs_target_density.png")

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

    results, thr_table, df_bins, df_tests_global, df_pairwise = analyze(
        df,
        output_dir=outdir,
        bootstrap_iters=args.bootstrap_iters,
        top_k=args.top_k,
    )

    if args.desired_mae is not None:
        print("\n=== Percent reduction required to reach desired MAE ===")
        for key in ["valid", "all"]:
            if key in results and results[key].get("MAE") is not None:
                curr = results[key]["MAE"]
                pct = required_percent_reduction(curr, args.desired_mae)
                print(f"{key}: current MAE = {curr:.6g}; desired = {args.desired_mae:.6g}; reduction required = {pct:.2f}%")


if __name__ == "__main__":
    main()