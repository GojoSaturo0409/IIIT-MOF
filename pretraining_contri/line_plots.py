import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import r2_score, mean_absolute_error

# ── Config ────────────────────────────────────────────────────────────────────
root_dir = "."
methods = ["cgcnn", "cnn", "mae", "mae_ft_only"]
fractions = ["01pct", "05pct", "10pct", "20pct"]
x = [1, 5, 10, 20]  # numeric x-axis (% of data)

METHOD_LABELS = {
    "cgcnn": "CGCNN",
    "cnn": "CNN (3D)",
    "mae": "MAE (pretrain+FT)",
    "mae_ft_only": "MAE-FT-only",
}

COLORS = {
    "cgcnn": "#2D6BB5",
    "cnn": "#E07B2A",
    "mae": "#3AAD6E",
    "mae_ft_only": "#C93A3A",
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def standardize_columns(df, method):
    df = df.copy()

    if method == "cgcnn":
        if "y_true" not in df.columns or "y_pred" not in df.columns:
            raise ValueError(f"Expected columns y_true and y_pred for {method}")
        df["y_true"] = pd.to_numeric(df["y_true"], errors="coerce")
        df["y_pred"] = pd.to_numeric(df["y_pred"], errors="coerce")
    else:
        if "target" not in df.columns or "prediction" not in df.columns:
            raise ValueError(f"Expected columns target and prediction for {method}")
        df["y_true"] = pd.to_numeric(df["target"], errors="coerce")
        df["y_pred"] = pd.to_numeric(df["prediction"], errors="coerce")

    return df


def get_dedup_key_columns(df):
    preferred_keys = [
        "id", "ID", "sample_id", "structure_id", "material_id",
        "name", "cif_id", "entry_id", "index", "idx"
    ]
    key_cols = [c for c in preferred_keys if c in df.columns]
    if key_cols:
        return key_cols

    exclude = {"y_true", "y_pred", "target", "prediction", "abs_error", "is_valid"}
    return [c for c in df.columns if c not in exclude]


def deduplicate_by_lowest_mae(df, method):
    """
    For repeated entries, keep the row with the smallest absolute error.
    Returns:
        cleaned_df, removed_rows
    """
    df = standardize_columns(df, method)

    if "is_valid" in df.columns:
        df = df[df["is_valid"] == True].copy()

    df = df.dropna(subset=["y_true", "y_pred"]).copy()
    df["abs_error"] = (df["y_true"] - df["y_pred"]).abs()

    key_cols = get_dedup_key_columns(df)

    if len(key_cols) == 0:
        cleaned = df.sort_values("abs_error").drop_duplicates().copy()
        removed = len(df) - len(cleaned)
        cleaned = cleaned.drop(columns=["abs_error"])
        return cleaned, removed

    cleaned = df.loc[df.groupby(key_cols)["abs_error"].idxmin()].copy()
    cleaned = cleaned.sort_values(key_cols).reset_index(drop=True)
    removed = len(df) - len(cleaned)
    cleaned = cleaned.drop(columns=["abs_error"])

    return cleaned, removed


def aulc(values, x_vals=x):
    """
    Area Under the Learning Curve via trapezoidal integration,
    normalized by the x-range so the result is on the same scale as values.
    NaN-safe.
    """
    xv = np.array(x_vals, dtype=float)
    yv = np.array(values, dtype=float)
    mask = ~np.isnan(yv)

    if mask.sum() < 2:
        return np.nan

    area = np.trapezoid(yv[mask], xv[mask])
    return area / (xv[mask][-1] - xv[mask][0])


def relative_efficiency(values, baseline_idx=-1):
    """
    R²(f) / R²(20%) — how close each fraction is to the full-data ceiling.
    """
    arr = np.array(values, dtype=float)
    base = arr[baseline_idx]
    if np.isnan(base) or base == 0:
        return [np.nan] * len(arr)
    return (arr / base).tolist()


def fraction_to_threshold(r2_values, threshold=0.80, x_vals=x):
    """
    Linearly interpolate the training-data fraction at which a model first
    reaches `threshold` R². Returns np.nan if never reached.
    """
    xv = np.array(x_vals, dtype=float)
    yv = np.array(r2_values, dtype=float)

    for i in range(len(yv) - 1):
        if np.isnan(yv[i]) or np.isnan(yv[i + 1]):
            continue

        y0, y1 = yv[i], yv[i + 1]
        x0, x1 = xv[i], xv[i + 1]

        if y0 <= threshold <= y1 or y1 <= threshold <= y0:
            t = (threshold - y0) / (y1 - y0 + 1e-12)
            return x0 + t * (x1 - x0)

        if y0 >= threshold:
            return x0

    if not np.isnan(yv[-1]) and yv[-1] >= threshold:
        return xv[-1]

    return np.nan


def nmae(mae_values, y_true_all):
    """
    Normalized MAE = MAE / std(y_true), per fraction.
    """
    out = []
    for mae_val, yt in zip(mae_values, y_true_all):
        if np.isnan(mae_val) or len(yt) == 0:
            out.append(np.nan)
        else:
            sigma = np.std(yt)
            out.append(mae_val / sigma if sigma > 0 else np.nan)
    return out


# ── Load results ──────────────────────────────────────────────────────────────

results = {m: {"r2": [], "mae": [], "y_true_all": [], "y_pred_all": []}
           for m in methods}

for method in methods:
    for frac in fractions:
        file_path = os.path.join(root_dir, method, frac, "test_predictions.csv")

        if not os.path.exists(file_path):
            print(f"Missing: {file_path}")
            results[method]["r2"].append(np.nan)
            results[method]["mae"].append(np.nan)
            results[method]["y_true_all"].append(np.array([]))
            results[method]["y_pred_all"].append(np.array([]))
            continue

        df = pd.read_csv(file_path)

        try:
            cleaned_df, removed = deduplicate_by_lowest_mae(df, method)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            results[method]["r2"].append(np.nan)
            results[method]["mae"].append(np.nan)
            results[method]["y_true_all"].append(np.array([]))
            results[method]["y_pred_all"].append(np.array([]))
            continue

        cleaned_path = os.path.join(root_dir, method, frac, "test_predictions_nodup.csv")
        cleaned_df.to_csv(cleaned_path, index=False)
        print(f"Saved: {cleaned_path} | removed {removed} duplicate rows")

        y_true = cleaned_df["y_true"].values
        y_pred = cleaned_df["y_pred"].values

        results[method]["r2"].append(r2_score(y_true, y_pred))
        results[method]["mae"].append(mean_absolute_error(y_true, y_pred))
        results[method]["y_true_all"].append(y_true)
        results[method]["y_pred_all"].append(y_pred)


# ── Derived metrics ───────────────────────────────────────────────────────────

summary_rows = []

for method in methods:
    r2_vals = results[method]["r2"]
    mae_vals = results[method]["mae"]
    yt_all = results[method]["y_true_all"]

    aulc_r2 = aulc(r2_vals)
    aulc_mae = aulc([-v for v in mae_vals])  # negate so higher = better
    rel_eff = relative_efficiency(r2_vals)
    thr_075 = fraction_to_threshold(r2_vals, threshold=0.75)
    thr_080 = fraction_to_threshold(r2_vals, threshold=0.80)
    nmae_vals = nmae(mae_vals, yt_all)
    mean_nmae = np.nanmean(nmae_vals) if len(nmae_vals) else np.nan

    summary_rows.append({
        "Model": METHOD_LABELS[method],
        "AULC (R²)": round(aulc_r2, 4) if not np.isnan(aulc_r2) else "N/A",
        "AULC (-MAE)": round(aulc_mae, 4) if not np.isnan(aulc_mae) else "N/A",
        "Rel. eff. @ 1%": round(rel_eff[0], 3) if not np.isnan(rel_eff[0]) else "N/A",
        "Rel. eff. @ 5%": round(rel_eff[1], 3) if not np.isnan(rel_eff[1]) else "N/A",
        "Rel. eff. @ 10%": round(rel_eff[2], 3) if not np.isnan(rel_eff[2]) else "N/A",
        "% data for R²≥0.75": round(thr_075, 1) if not np.isnan(thr_075) else ">20%",
        "% data for R²≥0.80": round(thr_080, 1) if not np.isnan(thr_080) else ">20%",
        "Mean nMAE": round(mean_nmae, 4) if not np.isnan(mean_nmae) else "N/A",
    })

df_summary = pd.DataFrame(summary_rows)
df_summary.to_csv("ablation_metrics_summary.csv", index=False)

print("\n── Ablation Metrics Summary ──────────────────────────────────────────")
print(df_summary.to_string(index=False))


# ── Plotting ──────────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(18, 14))
fig.suptitle(
    "Data Efficiency Ablation — MAE Pretraining Contribution",
    fontsize=15, fontweight="bold", y=0.98
)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35)

# ── Panel 1: R² vs Training Data ─────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
for method in methods:
    yv = results[method]["r2"]
    ax1.plot(
        x, yv, marker="o", color=COLORS[method],
        label=METHOD_LABELS[method], linewidth=2, markersize=7
    )
ax1.axhline(0.80, color="grey", linestyle=":", linewidth=1.2, label="R²=0.80 threshold")
ax1.axhline(0.75, color="silver", linestyle=":", linewidth=1.0, label="R²=0.75 threshold")
ax1.set_xlabel("Training Data (%)")
ax1.set_ylabel("R² Score  (↑ better)")
ax1.set_title("R² vs Training Data Size", fontweight="bold")
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.4)

# ── Panel 2: MAE vs Training Data ────────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
for method in methods:
    yv = results[method]["mae"]
    ax2.plot(
        x, yv, marker="o", color=COLORS[method],
        label=METHOD_LABELS[method], linewidth=2, markersize=7
    )
ax2.set_xlabel("Training Data (%)")
ax2.set_ylabel("MAE (mmol/g)  (↓ better)")
ax2.set_title("MAE vs Training Data Size", fontweight="bold")
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.4)

# ── Panel 3: AULC bar chart ───────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[0, 2])
aulc_vals = [aulc(results[m]["r2"]) for m in methods]
bar_colors = [COLORS[m] for m in methods]
bars = ax3.bar(
    [METHOD_LABELS[m] for m in methods],
    aulc_vals,
    color=bar_colors, edgecolor="white", linewidth=0.8, alpha=0.88
)
for bar, val in zip(bars, aulc_vals):
    if not np.isnan(val):
        ax3.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold"
        )
ax3.set_ylabel("AULC (R²)  (↑ better)")
ax3.set_title("Area Under Learning Curve\n(data-efficiency summary)", fontweight="bold")
ax3.set_xticks(range(len(methods)))
ax3.set_xticklabels([METHOD_LABELS[m] for m in methods], fontsize=8, rotation=12)
ax3.grid(True, axis="y", alpha=0.4)

# ── Panel 4: Relative efficiency ──────────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 0])
for method in methods:
    rel = relative_efficiency(results[method]["r2"])
    ax4.plot(
        x, rel, marker="s", color=COLORS[method],
        label=METHOD_LABELS[method], linewidth=2, markersize=7
    )
ax4.axhline(1.0, color="black", linestyle="--", linewidth=1.2, label="Full ceiling (20%)")
ax4.set_xlabel("Training Data (%)")
ax4.set_ylabel("R²(f) / R²(20%)  (↑ better)")
ax4.set_title("Relative Efficiency\nvs. Full-Data Ceiling", fontweight="bold")
ax4.legend(fontsize=8)
ax4.grid(True, alpha=0.4)

# ── Panel 5: % data to reach threshold ───────────────────────────────────────
ax5 = fig.add_subplot(gs[1, 1])
thresholds = [0.70, 0.75, 0.80]
bar_width = 0.18
n_methods = len(methods)
thr_x = np.arange(len(thresholds))

for i, method in enumerate(methods):
    r2_vals = results[method]["r2"]
    thr_vals = [fraction_to_threshold(r2_vals, t) for t in thresholds]
    thr_plot = [v if not np.isnan(v) else 21 for v in thr_vals]
    offset = (i - n_methods / 2 + 0.5) * bar_width

    bars_thr = ax5.bar(
        thr_x + offset, thr_plot,
        width=bar_width, color=COLORS[method],
        label=METHOD_LABELS[method], alpha=0.88, edgecolor="white"
    )

    for bar, val in zip(bars_thr, thr_vals):
        label_txt = f"{val:.1f}%" if not np.isnan(val) else ">20%"
        ax5.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            label_txt, ha="center", va="bottom", fontsize=7, rotation=45
        )

ax5.set_xticks(thr_x)
ax5.set_xticklabels([f"R²≥{t}" for t in thresholds])
ax5.set_ylabel("Training data required (%)")
ax5.set_title("Data Required to Reach\nPerformance Threshold", fontweight="bold")
ax5.legend(fontsize=8)
ax5.axhline(20, color="grey", linestyle=":", linewidth=1)
ax5.grid(True, axis="y", alpha=0.4)

# ── Panel 6: Normalized MAE ───────────────────────────────────────────────────
ax6 = fig.add_subplot(gs[1, 2])
for method in methods:
    nmae_vals = nmae(results[method]["mae"], results[method]["y_true_all"])
    ax6.plot(
        x, nmae_vals, marker="^", color=COLORS[method],
        label=METHOD_LABELS[method], linewidth=2, markersize=7
    )
ax6.axhline(0.5, color="grey", linestyle=":", linewidth=1.2, label="nMAE=0.5 (good threshold)")
ax6.set_xlabel("Training Data (%)")
ax6.set_ylabel("nMAE = MAE / σ(target)  (↓ better)")
ax6.set_title("Normalized MAE\n(scale-independent comparison)", fontweight="bold")
ax6.legend(fontsize=8)
ax6.grid(True, alpha=0.4)

plt.savefig("ablation_full_comparison.png", dpi=300, bbox_inches="tight")
plt.show()

print("\nFigure saved: ablation_full_comparison.png")
print("Summary CSV saved: ablation_metrics_summary.csv")