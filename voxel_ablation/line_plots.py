import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error

root_dir = "."   # change if needed

sizes = ["32", "64", "96"]

r2_list = []
mae_list = []

for size in sizes:
    file_path = os.path.join(root_dir, size, "test_predictions.csv")

    if not os.path.exists(file_path):
        print(f"Missing: {file_path}")
        r2_list.append(np.nan)
        mae_list.append(np.nan)
        continue

    df = pd.read_csv(file_path)

    # Filter valid rows if column exists
    if "is_valid" in df.columns:
        df = df[df["is_valid"] == True]

    y_true = df["target"]
    y_pred = df["prediction"]

    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)

    r2_list.append(r2)
    mae_list.append(mae)

# Convert x-axis to integers
x = [int(s) for s in sizes]

# -------- R2 Plot --------
plt.figure()
plt.plot(x, r2_list, marker='o')
plt.xlabel("Model Size / Dimension")
plt.ylabel("R² Score")
plt.title("R² vs Model Size")
plt.grid()
plt.savefig("r2_vs_size.png", dpi=300)
plt.show()

# -------- MAE Plot --------
plt.figure()
plt.plot(x, mae_list, marker='o')
plt.xlabel("Model Size / Dimension")
plt.ylabel("Mean Absolute Error")
plt.title("MAE vs Model Size")
plt.grid()
plt.savefig("mae_vs_size.png", dpi=300)
plt.show()
