import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from pathlib import Path

def _save_fig(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)

# --- Regression Plots ---

def plot_regression_suite(y_true, y_pred, out_dir: Path, label=""):
    out_dir.mkdir(parents=True, exist_ok=True)
    resid = y_pred - y_true
    
    # 1. Scatter
    fig, ax = plt.subplots(figsize=(6,6))
    ax.scatter(y_true, y_pred, alpha=0.5, s=15)
    lims = [min(min(y_true), min(y_pred)), max(max(y_true), max(y_pred))]
    ax.plot(lims, lims, 'k--', alpha=0.7)
    ax.set_xlabel("Target"); ax.set_ylabel("Prediction")
    ax.set_title(f"Pred vs Target ({label})")
    _save_fig(fig, out_dir / f"{label}_scatter.png")

    # 2. Residuals vs Target
    fig, ax = plt.subplots(figsize=(6,4))
    ax.scatter(y_true, resid, alpha=0.5, s=15)
    ax.axhline(0, c='k', ls='--')
    ax.set_xlabel("Target"); ax.set_ylabel("Residual")
    ax.set_title(f"Residuals ({label})")
    _save_fig(fig, out_dir / f"{label}_residuals.png")

    # 3. Residual Hist
    fig, ax = plt.subplots(figsize=(6,4))
    ax.hist(resid, bins=30, density=True, alpha=0.7)
    try:
        kde = stats.gaussian_kde(resid)
        x = np.linspace(resid.min(), resid.max(), 100)
        ax.plot(x, kde(x), 'r-')
    except: pass
    ax.set_title(f"Residual Dist ({label})")
    _save_fig(fig, out_dir / f"{label}_resid_hist.png")

def plot_mae_thresholds(df_table, out_path):
    fig, ax = plt.subplots(figsize=(8,5))
    ax.bar(df_table['threshold_label'], df_table['MAE'])
    ax.set_ylabel("MAE")
    ax.set_title("MAE vs Outlier Removal Thresholds")
    plt.xticks(rotation=45)
    _save_fig(fig, out_path)

# --- Structural Analysis Plots ---

def plot_feature_importance(stats_results: dict, out_dir: Path, top_n=20):
    sorted_feats = sorted(stats_results.items(), key=lambda x: abs(x[1]['cohens_d']), reverse=True)[:top_n]
    names = [x[0] for x in sorted_feats]
    values = [x[1]['cohens_d'] for x in sorted_feats]
    colors = ['g' if v > 0 else 'r' for v in values]

    fig, ax = plt.subplots(figsize=(10, 8))
    y_pos = np.arange(len(names))
    ax.barh(y_pos, values, color=colors)
    ax.set_yticks(y_pos); ax.set_yticklabels(names)
    ax.set_xlabel("Cohen's d (Effect Size)")
    ax.set_title(f"Top {top_n} Discriminating Features (Green=Higher in Best)")
    ax.axvline(0, c='k', lw=0.5)
    _save_fig(fig, out_dir / "feature_importance.png")

def plot_feature_boxplots(best_df, worst_df, features, out_dir: Path):
    top_n = min(9, len(features))
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()
    
    for i, feat in enumerate(features[:top_n]):
        data = [best_df[feat].dropna(), worst_df[feat].dropna()]
        axes[i].boxplot(data, labels=['Best', 'Worst'], patch_artist=True)
        axes[i].set_title(feat)
    
    for i in range(top_n, 9): axes[i].axis('off')
    _save_fig(fig, out_dir / "top_feature_boxplots.png")
