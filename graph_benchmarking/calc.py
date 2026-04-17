import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.stats import spearmanr

# Path to your file
file_path = "exp1/test_predictions.csv"

# Load data
df = pd.read_csv(file_path)

# Handle your known formats
if "y_true" in df.columns and "y_pred" in df.columns:
    y_true = df["y_true"]
    y_pred = df["y_pred"]
elif "target" in df.columns and "prediction" in df.columns:
    # Filter valid rows if present
    if "is_valid" in df.columns:
        df = df[df["is_valid"] == True]
    y_true = df["target"]
    y_pred = df["prediction"]
else:
    raise ValueError("Unknown column format in CSV")

# Metrics
r2 = r2_score(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)
spearman_rho, _ = spearmanr(y_true, y_pred)

# Print results
print(f"R2 Score        : {r2:.4f}")
print(f"MAE             : {mae:.4f}")
print(f"Spearman's rho  : {spearman_rho:.4f}")
