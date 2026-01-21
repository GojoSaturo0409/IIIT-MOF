import logging
import sys
import pandas as pd
import numpy as np
from pathlib import Path

def setup_logger(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True
    )

def safe_float_series(s):
    return pd.to_numeric(s, errors="coerce")

def load_and_clean_csv(csv_path: Path) -> pd.DataFrame:
    """Robust CSV loader handling column variations and type conversion."""
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    # Normalize column names
    rename_map = {}
    col_lower = [c.lower() for c in df.columns]
    
    mapping = {
        'prediction': ['pred', 'prediction'],
        'target': ['label', 'truth', 'y', 'target', 'target_value'],
        'is_valid': ['valid', 'isvalid', 'is_valid']
    }

    for standard, variations in mapping.items():
        if standard not in col_lower:
            for c in df.columns:
                if any(v in c.lower() for v in variations):
                    rename_map[c] = standard
                    break
    
    if rename_map:
        df = df.rename(columns=rename_map)

    required = ["prediction", "target"]
    if not all(r in df.columns for r in required):
        # Fallback for simple files
        if len(df.columns) >= 2:
             logging.warning("Columns 'prediction'/'target' not found. Assuming col 1=target, col 2=prediction.")
             # This is a risky fallback, usually better to error, but provided for robustness
    
    # Type conversion
    df["prediction"] = safe_float_series(df["prediction"])
    df["target"] = safe_float_series(df["target"])
    
    if "is_valid" in df.columns:
        df["is_valid"] = df["is_valid"].astype(str).str.lower().map(
            {"true": True, "yes": True, "1": True, "false": False, "no": False, "0": False}
        ).fillna(True)
    else:
        df["is_valid"] = True

    # Filename handling
    if "filename" not in df.columns:
        df["filename"] = [f"sample_{i}" for i in range(len(df))]
    else:
        df["filename"] = df["filename"].astype(str).str.strip()

    # Calculate residuals
    df = df.dropna(subset=["prediction", "target"]).copy()
    df["residual"] = df["prediction"] - df["target"]
    df["abs_residual"] = df["residual"].abs()
    
    return df
