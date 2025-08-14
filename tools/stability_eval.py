# src/stability_eval.py
# Re-train with multiple random seeds to assess stability.
# Outputs:
#   results/stability/stability_runs.csv
#   results/stability/stability_summary.csv
#   results/stability/plots/<metric>_box.png

import os
import warnings
import inspect
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, balanced_accuracy_score,
    r2_score, mean_absolute_error, mean_squared_error
)

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

# ------------------- CONFIG -------------------
DATA_PATH = "data/namcs2019_clean.csv"
RESULT_DIR = Path("results/stability")
N_RUNS = 30
TEST_SIZE = 0.30
RANDOM_STATE_BASE = 42
TARGET_HINTS = ["illness", "diagnosis", "disease", "condition", "target", "label"]
FALLBACK_TARGET = "physwt"  # used if no hint is found
MIN_CLASS_COUNT = 2         # drop classes with < MIN_CLASS_COUNT samples
# ----------------------------------------------

warnings.filterwarnings("ignore", category=UserWarning)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

def detect_target_column(df: pd.DataFrame) -> str:
    for col in df.columns:
        low = col.lower()
        if any(k in low for k in TARGET_HINTS):
            return col
    if FALLBACK_TARGET in df.columns:
        return FALLBACK_TARGET
    return df.columns[-1]

def split_feature_types(df: pd.DataFrame):
    cats, nums = [], []
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            nums.append(c)
        else:
            cats.append(c)
    return cats, nums

def make_preprocessor(cat_cols, num_cols):
    # Compatibility with different scikit-learn versions
    ohe_params = {"handle_unknown": "ignore"}
    if "sparse_output" in inspect.signature(OneHotEncoder).parameters:
        ohe_params["sparse_output"] = False  # sklearn >= 1.2
    else:
        ohe_params["sparse"] = False         # sklearn < 1.2

    ct = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(**ohe_params), cat_cols),
            ("num", Pipeline([("scaler", StandardScaler())]), num_cols),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )
    return ct

def is_classification(y: pd.Series) -> bool:
    if y.dtype == bool or y.dtype.name in ("category", "object"):
        return True
    nunique = y.nunique(dropna=True)
    if nunique <= max(20, int(0.05 * len(y))):
        return True
    return False

def drop_rare_classes(X, y, min_count=MIN_CLASS_COUNT):
    vc = y.value_counts()
    keep_labels = vc[vc >= min_count].index
    mask = y.isin(keep_labels)
    return X[mask].reset_index(drop=True), y[mask].reset_index(drop=True)

def metrics_for_task(y_true, y_pred, task: str) -> dict:
    if task == "classification":
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        }
    else:
        mse = mean_squared_error(y_true, y_pred)
        rmse = float(np.sqrt(mse))
        return {
            "r2": r2_score(y_true, y_pred),
            "mae": mean_absolute_error(y_true, y_pred),
            "mse": mse,
            "rmse": rmse,
        }

def plot_box(metric_name: str, df: pd.DataFrame, title: str, outdir: Path):
    plt.figure(figsize=(6, 4))
    data = df[metric_name].dropna().values
    if data.size == 0:
        plt.close()
        return
    plt.boxplot(data, vert=True, labels=[metric_name])
    plt.title(title)
    plt.ylabel(metric_name)
    plt.tight_layout()
    out_path = outdir / f"{metric_name}_box.png"
    plt.savefig(out_path, dpi=300)
    plt.close()

def main():
    print(f"🔌 Loading: {DATA_PATH}")
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"{DATA_PATH} not found.")
    df = pd.read_csv(DATA_PATH, low_memory=False)

    target_col = detect_target_column(df)
    print(f"🎯 Target column: {target_col}")

    df = df.dropna(subset=[target_col])
    y = df[target_col]
    X = df.drop(columns=[target_col])

    cat_cols, num_cols = split_feature_types(X)
    pre = make_preprocessor(cat_cols, num_cols)

    task = "classification" if is_classification(y) else "regression"
    print(f"🧪 Stability task: {task}  |  Runs: {N_RUNS}")

    if task == "classification":
        models = {
            "LogisticRegression": LogisticRegression(max_iter=200),
            "RandomForest": RandomForestClassifier(n_estimators=300, n_jobs=-1),
            "SVC_RBF": SVC(kernel="rbf", probability=True),
            "KNN": KNeighborsClassifier(n_neighbors=7),
        }
        metric_cols = ["accuracy", "macro_f1", "balanced_accuracy"]
    else:
        models = {
            "LinearRegression": LinearRegression(),
            "RandomForestRegressor": RandomForestRegressor(n_estimators=300, n_jobs=-1),
            "SVR_RBF": SVR(kernel="rbf"),
            "KNNRegressor": KNeighborsRegressor(n_neighbors=7),
        }
        metric_cols = ["r2", "mae", "mse", "rmse"]

    rows = []
    for run in range(N_RUNS):
        seed = RANDOM_STATE_BASE + run
        X_run, y_run = X.copy(), y.copy()

        if task == "classification":
            X_run, y_run = drop_rare_classes(X_run, y_run, min_count=MIN_CLASS_COUNT)
            if y_run.nunique() < 2:
                print(f"⚠️  Run {run+1}: <2 classes after rare-class drop — skipping.")
                continue

        try:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X_run, y_run,
                test_size=TEST_SIZE,
                random_state=seed,
                stratify=y_run if task == "classification" else None
            )
        except ValueError as e:
            print(f"⚠️  Run {run+1}: split error ({e}) — skipping.")
            continue

        for mname, model in models.items():
            pipe = Pipeline([("preprocessor", pre), ("model", model)])
            try:
                pipe.fit(X_tr, y_tr)
                preds = pipe.predict(X_te)
                met = metrics_for_task(y_te, preds, task)
                row = {"run": run+1, "model": mname, **met}
                rows.append(row)
                pretty = ", ".join([f"{k}={v:.4f}" for k, v in met.items()])
                print(f"✔️  Run {run+1:02d} {mname}: {pretty}")
            except Exception as e:
                print(f"⚠️  Run {run+1}: {mname} failed ({e})")

    if not rows:
        print("❌ No successful runs; nothing to save.")
        return

    res = pd.DataFrame(rows)
    out_csv = RESULT_DIR / "stability_runs.csv"
    res.to_csv(out_csv, index=False)
    print(f"🧾 Saved {out_csv}")

    summary = (
        res.groupby("model")[metric_cols]
           .agg(["mean", "std"])
           .sort_values((metric_cols[0], "mean"), ascending=False)
    )
    out_sum = RESULT_DIR / "stability_summary.csv"
    summary.to_csv(out_sum)
    print(f"🧾 Saved {out_sum}")

    plot_dir = RESULT_DIR / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    for metric in metric_cols:
        plot_box(metric, res, f"{metric} Stability (over {N_RUNS} runs)", plot_dir)
        print(f"📈 Saved {plot_dir / (metric + '_box.png')}")

    print(f"✅ Done. Open this folder: {RESULT_DIR.resolve()}")

if __name__ == "__main__":
    main()
