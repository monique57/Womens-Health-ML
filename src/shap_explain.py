# src/shap_explain.py
import os
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, issparse

# ---------------- CONFIG ----------------
MODEL_PATH = "models/best_model_namcs2019.joblib"
DATA_PATH = "data/namcs2019_clean.csv"
TARGET_COLUMN = "physwt"           # change if needed
TOP_N = 15                         # number of bars
SAMPLE_SIZE = 200                  # speed-up
OUTPUT_PLOT = "results/shap_candlestick.png"
os.makedirs(os.path.dirname(OUTPUT_PLOT), exist_ok=True)
# ---------------------------------------

print("📦 Loading model...")
pipe = joblib.load(MODEL_PATH)

print("📄 Loading dataset...")
df = pd.read_csv(DATA_PATH, low_memory=False)
df = df.dropna(subset=[TARGET_COLUMN])

X = df.drop(columns=[TARGET_COLUMN])

# Sample to speed up SHAP
if SAMPLE_SIZE and SAMPLE_SIZE < len(X):
    Xs = X.sample(SAMPLE_SIZE, random_state=42)
else:
    Xs = X

print("⚙️ Transforming features...")
pre = pipe.named_steps.get("preprocessor")
model = pipe.named_steps.get("model") or pipe.named_steps[list(pipe.named_steps.keys())[-1]]

Xt = pre.transform(Xs) if pre is not None else Xs
if issparse(Xt):
    Xt = Xt.toarray()
Xt = np.nan_to_num(Xt, nan=0.0, posinf=0.0, neginf=0.0)

# Get feature names and guarantee a 1-D array of strings
if pre is not None:
    try:
        fn = pre.get_feature_names_out()
    except Exception:
        fn = Xs.columns
else:
    fn = Xs.columns
# Flatten/convert any odd name objects to strings
feature_names = np.array([str(n[0]) if isinstance(n, (list, np.ndarray)) else str(n) for n in fn], dtype=object)

# ---- Build a SHAP explainer appropriate for trees/linear/other ----
mname = type(model).__name__.lower()
print(f"🤖 Estimator: {type(model).__name__}")

explainer = None
shap_vals = None

try:
    if any(k in mname for k in ["randomforest", "extratrees", "xgb", "lgbm", "catboost", "gradientboost"]):
        explainer = shap.TreeExplainer(model)
        sv = explainer(Xt)
        # sv can be Explanation or list of Explanations (multiclass)
        if isinstance(sv, list):
            # stack to (n_samples, n_classes, n_features)
            vals = np.stack([s.values for s in sv], axis=1)
        else:
            vals = sv.values
    elif any(k in mname for k in ["logisticregression", "ridgeclassifier", "linear"]):
        explainer = shap.LinearExplainer(model, Xt)
        vals = explainer.shap_values(Xt)
    else:
        print("⚠️ Using KernelExplainer fallback (slower).")
        predict_fn = model.predict_proba if hasattr(model, "predict_proba") else model.predict
        bg = Xt[: min(100, Xt.shape[0])]
        explainer = shap.KernelExplainer(predict_fn, bg, link="identity")
        vals = explainer.shap_values(Xt, nsamples=100)
        if isinstance(vals, list):  # pick the class with largest mean |SHAP|
            vals = vals[int(np.argmax([np.mean(np.abs(v)) for v in vals]))]
except Exception as e:
    raise RuntimeError(f"Failed to compute SHAP values: {e}")

# ---- Normalize SHAP shapes to a single importance vector (n_features,) ----
vals = np.array(vals)
# Possible shapes:
# (n_samples, n_features)
# (n_samples, n_classes, n_features)  -> average over samples & classes
# (n_features,) -> already fine
if vals.ndim == 3:
    # average of absolute values over samples and classes, per feature
    mean_abs = np.mean(np.abs(vals), axis=(0, 1))
elif vals.ndim == 2:
    # average over samples
    mean_abs = np.mean(np.abs(vals), axis=0)
elif vals.ndim == 1:
    mean_abs = np.abs(vals)
else:
    # fallback: flatten everything except last axis as samples
    axes = tuple(range(vals.ndim - 1))
    mean_abs = np.mean(np.abs(vals), axis=axes)

# Sanitize lengths
n_features = min(len(feature_names), mean_abs.shape[-1])
feature_names = feature_names[:n_features]
mean_abs = mean_abs[:n_features]

# Rank and select top-N
order = np.argsort(mean_abs)[::-1][: min(TOP_N, n_features)]
names = feature_names[order]          # NumPy index -> safe
heights = mean_abs[order]

# Optional scaling for tiny values
mx = float(np.max(heights)) if heights.size else 0.0
xlabel = "Mean |SHAP value|"
if mx == 0.0:
    heights = np.ones_like(heights)
    xlabel += " (rank-only view)"
elif mx < 1e-3:
    heights = heights * 1_000_000.0
    xlabel += " (×1e6)"
elif mx < 1e-2:
    heights = heights * 1_000.0
    xlabel += " (×1000)"

# Ensure plain Python strings for Matplotlib categorical axis
names = [str(n) for n in names.tolist()]

print("📈 Creating candlestick plot…")
plt.figure(figsize=(9, 6))
plt.barh(names[::-1], heights[::-1])  # default color; consistent with your app style
plt.xlabel(xlabel)
plt.title(f"Top {len(names)} Features — SHAP (candlestick style)")
plt.tight_layout()
plt.savefig(OUTPUT_PLOT, dpi=300)
plt.show()

print(f"✅ Candlestick SHAP plot saved to {OUTPUT_PLOT}")