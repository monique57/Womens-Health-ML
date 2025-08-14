# src/stability_eval.py
import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from math import ceil
from sklearn.model_selection import train_test_split
from sklearn.utils.multiclass import type_of_target
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    r2_score, mean_absolute_error, mean_squared_error
)

# ------------------ CONFIG ------------------
DATA_PATH  = "data/namcs2019_clean.csv"
MODEL_PATH = "models/best_model_namcs2019.joblib"
RESULTS_DIR = "results"
PLOTS_DIR   = os.path.join(RESULTS_DIR, "plots")
N_RUNS      = 30
TEST_SIZE   = 0.30
RANDOM_SEED = 42
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
# --------------------------------------------

def detect_target_column(df: pd.DataFrame) -> str:
    keys = ["illness", "diagnosis", "disease", "condition", "target", "label"]
    for c in df.columns:
        if any(k in c.lower() for k in keys):
            return c
    return df.columns[-1]

def safe_split(X, y, run_seed):
    """Use stratified split if feasible; else fall back."""
    try:
        n_classes = len(pd.Series(y).unique())
        n_test = ceil(len(y) * TEST_SIZE)
        if n_classes > n_test:
            raise ValueError("Too many classes for stratify")
        return train_test_split(
            X, y, test_size=TEST_SIZE, random_state=run_seed, stratify=y
        )
    except Exception:
        return train_test_split(
            X, y, test_size=TEST_SIZE, random_state=run_seed, stratify=None
        )

print("🔌 Loading pipeline and data…")
pipe = joblib.load(MODEL_PATH)  # full pipeline saved by train script
df = pd.read_csv(DATA_PATH, low_memory=False)

target_col = detect_target_column(df)
df = df.dropna(subset=[target_col])
X = df.drop(columns=[target_col])
y = df[target_col]

# Decide task type from raw target (classification vs regression)
y_type = type_of_target(y)
task_is_classification = y_type in ["binary", "multiclass"]

# Convert y for metrics when needed
if task_is_classification:
    y_eval = y.astype("category").cat.codes if y.dtype == "object" else y
else:
    y_eval = pd.to_numeric(y, errors="coerce")
    mask = ~pd.isna(y_eval)
    X, y_eval = X[mask], y_eval[mask]

rows = []

print(f"🏃 Running {N_RUNS} evaluation splits (TEST_SIZE={TEST_SIZE:.2f})…")
for i in range(N_RUNS):
    seed = RANDOM_SEED + i + 1
    X_train, X_test, y_train, y_test = safe_split(X, y_eval, seed)
    # pipeline is already trained; we only evaluate (no fit here)
    y_pred = pipe.predict(X_test)

    if task_is_classification:
        rows.append({
            "run": i + 1,
            "accuracy": accuracy_score(y_test, y_pred),
            "precision_weighted": precision_score(y_test, y_pred, average="weighted", zero_division=0),
            "recall_weighted": recall_score(y_test, y_pred, average="weighted", zero_division=0),
            "f1_weighted": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        })
    else:
        mse  = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        rows.append({
            "run": i + 1,
            "r2": r2_score(y_test, y_pred),
            "mae": mean_absolute_error(y_test, y_pred),
            "mse": mse,
            "rmse": rmse,
        })

metrics_df = pd.DataFrame(rows)
csv_path = os.path.join(RESULTS_DIR, "stability_metrics.csv")
metrics_df.to_csv(csv_path, index=False)
print(f"🧾 Saved per-run metrics → {csv_path}")

# -------- plotting helpers (one figure per metric) --------
def plot_box(metric_name, title):
    vals = metrics_df[metric_name].dropna().values
    if len(vals) == 0:
        return
    mean_val = np.mean(vals)

    plt.figure(figsize=(6, 5))
    plt.boxplot(vals, vert=True, showmeans=False)
    plt.axhline(mean_val, linestyle="--", linewidth=1.5)
    plt.title(f"{title} (n={len(vals)})")
    plt.ylabel(metric_name)
    plt.tight_layout()
    out_path = os.path.join(PLOTS_DIR, f"{metric_name}_box.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"📈 Saved {out_path}")

if task_is_classification:
    plot_box("accuracy", "Accuracy Stability")
    plot_box("precision_weighted", "Precision (weighted) Stability")
    plot_box("recall_weighted", "Recall (weighted) Stability")
    plot_box("f1_weighted", "F1 (weighted) Stability")
else:
    plot_box("r2", "R² Stability")
    plot_box("mae", "MAE Stability")
    plot_box("mse", "MSE Stability")
    plot_box("rmse", "RMSE Stability")

# -------- summary table printed to console --------
print("\n📊 Stability summary:")
summary = metrics_df.describe().T[["mean", "std", "min", "max"]]
print(summary.to_string(float_format=lambda x: f"{x:.4f}"))

summary_path = os.path.join(RESULTS_DIR, "stability_summary.csv")
summary.to_csv(summary_path)
print(f"🧾 Saved summary → {summary_path}")

print("\n✅ Done.")