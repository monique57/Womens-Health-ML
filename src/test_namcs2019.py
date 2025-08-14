# src/test_namcs2019.py
import os
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, f1_score,
    mean_squared_error, r2_score
)
from sklearn.utils.multiclass import type_of_target
from math import ceil

# ----------------- CONFIG -----------------
DATA_PATH  = "data/namcs2019_clean.csv"         # same file you trained on (or comparable)
MODEL_PATH = "models/best_model_namcs2019.joblib"
OUTPUT_DIR = "outputs"
TOP_N_FEATURES = 20
N_SHAP_SAMPLES = 500     # subsample for SHAP speed; increase if you want
os.makedirs(OUTPUT_DIR, exist_ok=True)
# ------------------------------------------

def detect_target_column(df: pd.DataFrame) -> str:
    keys = ["illness", "diagnosis", "disease", "condition", "target", "label"]
    for c in df.columns:
        if any(k in c.lower() for k in keys):
            return c
    return df.columns[-1]

print("🔌 Loading model & data...")
pipe = joblib.load(MODEL_PATH)       # full pipeline saved by the train script
df   = pd.read_csv(DATA_PATH, low_memory=False)

# ----- target & X/y -----
target_col = detect_target_column(df)
print(f"🎯 Using target column: {target_col}")
df = df.dropna(subset=[target_col])

X = df.drop(columns=[target_col])
y = df[target_col]

# basic clean: drop NaN targets
mask = ~y.isna()
X, y = X[mask], y[mask]

# ----- predictions -----
print("✨ Making predictions...")
y_pred = pipe.predict(X)

# ----- choose metrics path -----
y_type = type_of_target(y)
if y_type in ["binary", "multiclass"]:
    acc = accuracy_score(y, y_pred)
    f1m = f1_score(y, y_pred, average="macro", zero_division=0)
    print("\n=== Classification Metrics ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"Macro F1 : {f1m:.4f}\n")
    print(classification_report(y, y_pred, zero_division=0))

    cm = confusion_matrix(y, y_pred)
    pd.DataFrame(cm).to_csv(os.path.join(OUTPUT_DIR, "confusion_matrix.csv"), index=False)
    print(f"🧾 Saved confusion_matrix.csv to {OUTPUT_DIR}")

else:
    # regression fallback (shouldn’t happen if you used the new train script, but safe)
    mse = mean_squared_error(y, y_pred)
    r2  = r2_score(y, y_pred)
    print("\n=== Regression Metrics (fallback) ===")
    print(f"MSE : {mse:.4f}")
    print(f"R²  : {r2:.4f}")
    pd.DataFrame([{"mse": mse, "r2": r2}]).to_csv(
        os.path.join(OUTPUT_DIR, "regression_summary.csv"), index=False
    )
    print(f"🧾 Saved regression_summary.csv to {OUTPUT_DIR}")

# ----- SHAP candlestick (robust to sparse & model type) -----
import scipy.sparse as sp
from statistics import mode

print("\n🔎 Building SHAP explanations (candlestick)…")

# 1) Extract model + features
model = pipe
feature_names = list(X.columns)
Xt = X

if hasattr(pipe, "named_steps") and "preprocessor" in pipe.named_steps:
    pre = pipe.named_steps["preprocessor"]
    Xt = pre.transform(X)
    try:
        feature_names = list(pre.get_feature_names_out())
    except Exception:
        # fallback if get_feature_names_out is missing
        feature_names = list(X.columns)
    model = pipe.named_steps.get("model", pipe)

# 2) Subsample for speed
rng = np.random.default_rng(42)
n = Xt.shape[0]
take = min(N_SHAP_SAMPLES, n)
idx = rng.choice(n, size=take, replace=False)
Xt_sample = Xt[idx] if not isinstance(Xt, pd.DataFrame) else Xt.iloc[idx]
y_pred_sample = np.array(y_pred)[idx]

# 3) Ensure DENSE float input for SHAP when needed
def to_dense(a):
    if sp.issparse(a):
        return a.toarray().astype(float, copy=False)
    if isinstance(a, pd.DataFrame):
        return a.to_numpy(dtype=float, copy=False)
    return np.asarray(a, dtype=float)

Xt_dense = to_dense(Xt_sample)

# 4) Pick proper explainer by model type
mname = type(model).__name__.lower()

try:
    if "randomforest" in mname or "xgb" in mname or "lgbm" in mname or "catboost" in mname:
        # Tree models
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(Xt_dense)
        # shap_values shape handling
        V = np.array(shap_values)  # list->array
        if V.ndim == 3:  # (classes, n, features)
            try:
                maj_cls = int(mode(y_pred_sample))
            except Exception:
                maj_cls = 0
            V = V[maj_cls]          # (n, features)
    elif "logisticregression" in mname or "linearsvc" in mname:
        # Linear models
        explainer = shap.LinearExplainer(model, Xt_dense)
        V = explainer.shap_values(Xt_dense)  # (n, features)
    else:
        # SVM (rbf), KNN, etc. Use KernelExplainer (slower; keep samples small)
        bg = shap.sample(Xt_dense, min(50, Xt_dense.shape[0]), random_state=42)
        explainer = shap.KernelExplainer(model.predict_proba if hasattr(model, "predict_proba") else model.predict,
                                         bg)
        V = explainer.shap_values(Xt_dense, nsamples=100)  # small nsamples for speed
        # If multi-class returns list
        if isinstance(V, list):
            try:
                maj_cls = int(mode(y_pred_sample))
            except Exception:
                maj_cls = 0
            V = np.array(V[maj_cls])  # (n, features)

    # 5) Candlestick (colored bar)
    mean_abs = np.mean(np.abs(V), axis=0)
    mean_signed = np.mean(V, axis=0)
    order = np.argsort(mean_abs)[::-1][:TOP_N_FEATURES]

    top_names = [feature_names[i] for i in order]
    top_imp   = mean_abs[order]
    top_sign  = mean_signed[order]
    colors = ["red" if s > 0 else "blue" for s in top_sign]

    plt.figure(figsize=(9, 7))
    plt.barh(range(len(top_names)), top_imp, color=colors)
    plt.gca().invert_yaxis()
    plt.yticks(range(len(top_names)), top_names)
    plt.xlabel("Mean |SHAP value| (impact on model output)")
    plt.title(f"Top {TOP_N_FEATURES} Features — SHAP Candlestick")
    plt.tight_layout()
    candlestick_path = os.path.join(OUTPUT_DIR, "shap_summary_candlestick_colored.png")
    plt.savefig(candlestick_path, dpi=300)
    plt.close()
    print(f"✅ Saved {candlestick_path}")

except Exception as e:
    print(f"❌ SHAP failed gracefully: {e}")