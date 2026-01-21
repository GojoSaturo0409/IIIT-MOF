import numpy as np
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def compute_metrics(y_true, y_pred):
    residuals = y_pred - y_true
    abs_res = np.abs(residuals)
    
    metrics = {
        "MAE": float(np.mean(abs_res)),
        "MSE": float(mean_squared_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MedianAE": float(np.median(abs_res)),
        "Bias": float(np.mean(residuals)),
        "StdResidual": float(np.std(residuals, ddof=1)) if len(residuals) > 1 else 0.0,
        "NormalizedMAE": float(np.mean(abs_res) / (np.mean(np.abs(y_true)) + 1e-10))
    }
    
    try:
        metrics["R2"] = float(r2_score(y_true, y_pred)) if len(y_true) > 1 else np.nan
        metrics["PearsonR"], metrics["Pearson_p"] = stats.pearsonr(y_true, y_pred) if len(y_true) > 1 else (np.nan, np.nan)
        metrics["SpearmanR"], metrics["Spearman_p"] = stats.spearmanr(y_true, y_pred) if len(y_true) > 1 else (np.nan, np.nan)
    except Exception:
        pass
        
    return metrics

def bootstrap_metric(y_true, y_pred, metric_fn, iters=1000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    if n == 0: return {"mean": np.nan, "lower": np.nan, "upper": np.nan}
    
    boots = []
    for _ in range(iters):
        idx = rng.integers(0, n, n)
        boots.append(metric_fn(y_true[idx], y_pred[idx]))
    
    boots = np.array(boots)
    return {
        "mean": float(boots.mean()),
        "lower": float(np.percentile(boots, 2.5)),
        "upper": float(np.percentile(boots, 97.5))
    }

def cohen_d(x, y):
    x, y = np.asarray(x), np.asarray(y)
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2: return np.nan
    
    pooled_std = np.sqrt(((nx - 1) * x.std(ddof=1) ** 2 + (ny - 1) * y.std(ddof=1) ** 2) / (nx + ny - 2))
    return (x.mean() - y.mean()) / (pooled_std + 1e-10)

def compare_distributions(group_a, group_b):
    """Runs T-test, Mann-Whitney U, and Cohen's d."""
    res = {
        'mean_a': float(np.mean(group_a)), 'std_a': float(np.std(group_a)),
        'mean_b': float(np.mean(group_b)), 'std_b': float(np.std(group_b)),
        'mean_diff': float(np.mean(group_a) - np.mean(group_b))
    }
    
    try:
        _, res['t_pval'] = stats.ttest_ind(group_a, group_b)
        _, res['u_pval'] = stats.mannwhitneyu(group_a, group_b)
        res['cohens_d'] = cohen_d(group_a, group_b)
    except Exception:
        res['t_pval'] = res['u_pval'] = res['cohens_d'] = np.nan
        
    return res
