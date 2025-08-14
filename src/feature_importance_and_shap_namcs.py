import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
import os

# ==== SETTINGS ====
MODEL_PATH = "models/namcs_rf_model.joblib"
FEATURES_PATH = "models/namcs_rf_features.joblib"
DATA_PATH = "models/namcs_rf_data.joblib"

# Output files
PI_CSV = "models/namcs_perm_importance.csv"
PI_PNG = "models/namcs_perm_importance.png"
SHAP_PNG = "models/namcs_shap_summary.png"

# Controls (adjust if runtime is too long)
N_REPEATS = 5
SAMPLE_ROWS_PI = 300
MAX_FEATURES_PI = 300

print(f"✅ Using model: {MODEL_PATH}")
model = joblib.load(MODEL_PATH)

# If the saved model is a dict (pipeline parts), extract the classifier
if isinstance(model, dict) and "classifier" in model:
    print("🔄 Rebuilding classifier from saved dict...")
    estimator = model["classifier"]
else:
    estimator = model

# Load feature names
print(f"✅ Using features from: {FEATURES_PATH}")
feature_names = joblib.load(FEATURES_PATH)

# Load X, y from saved data
print(f"✅ Using X, y from: {DATA_PATH}")
data_obj = joblib.load(DATA_PATH)

if isinstance(data_obj, dict):
    X = data_obj["X"]
    y = data_obj["y"]
else:
    raise ValueError(f"❌ Unexpected format in {DATA_PATH}: {type(data_obj)}")

# Ensure DataFrame with correct feature order
if feature_names is not None:
    X = pd.DataFrame(X, columns=feature_names)
    missing = [f for f in feature_names if f not in X.columns]
    if missing:
        raise ValueError(f"❌ Missing features in X: {missing}")
    X = X[feature_names]

# Optionally limit for faster PI
if SAMPLE_ROWS_PI and len(X) > SAMPLE_ROWS_PI:
    X_pi = X.sample(SAMPLE_ROWS_PI, random_state=42)
    y_pi = y.loc[X_pi.index]
else:
    X_pi, y_pi = X, y

if MAX_FEATURES_PI and X_pi.shape[1] > MAX_FEATURES_PI:
    X_pi = X_pi.iloc[:, :MAX_FEATURES_PI]

print(f"✅ Samples: {X_pi.shape[0]} | Features: {X_pi.shape[1]}")

# === Permutation Importance ===
print("⏳ Computing permutation importance...")
pi = permutation_importance(
    estimator, X_pi, y_pi,
    n_repeats=N_REPEATS,
    random_state=42,
    n_jobs=-1
)

pi_df = pd.DataFrame({
    "Feature": X_pi.columns,
    "ImportanceMean": pi.importances_mean,
    "ImportanceStd": pi.importances_std
}).sort_values("ImportanceMean", ascending=False)

pi_df.to_csv(PI_CSV, index=False)
print(f"💾 Permutation importance saved to {PI_CSV}")

# Plot PI
plt.figure(figsize=(8, 10))
plt.barh(pi_df["Feature"][:20], pi_df["ImportanceMean"][:20])
plt.gca().invert_yaxis()
plt.xlabel("Mean Importance")
plt.title("Top 20 Features (Permutation Importance)")
plt.tight_layout()
plt.savefig(PI_PNG)
plt.close()
print(f"🖼️ Permutation importance plot saved to {PI_PNG}")

# === SHAP Values ===
print("⏳ Computing SHAP values...")
explainer = shap.TreeExplainer(estimator)
shap_values = explainer.shap_values(X_pi)

# Summary plot
plt.figure()
shap.summary_plot(shap_values, X_pi, show=False)
plt.tight_layout()
plt.savefig(SHAP_PNG)
plt.close()
print(f"🖼️ SHAP summary plot saved to {SHAP_PNG}")

print("✅ NAMCS feature importance & SHAP analysis complete.")
