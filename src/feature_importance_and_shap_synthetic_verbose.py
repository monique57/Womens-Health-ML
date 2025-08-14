# src/feature_importance_and_shap_synthetic.py
import joblib
import shap
import pandas as pd
from sklearn.inspection import permutation_importance
import os
import sys

def main():
    print("🚀 Starting Synthetic Feature Importance + SHAP...")

    # Paths
    MODEL_PATH = "models/synthetic_rf_model.joblib"
    FEATURES_PATH = "models/synthetic_rf_features.joblib"
    DATA_PATH = "models/synthetic_rf_data.joblib"

    # --- Load Model ---
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"❌ Model file not found: {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
    print(f"✅ Loaded model from: {MODEL_PATH}")

    # --- Load Feature Names ---
    if not os.path.exists(FEATURES_PATH):
        raise FileNotFoundError(f"❌ Features file not found: {FEATURES_PATH}")
    feature_names = joblib.load(FEATURES_PATH)
    print(f"✅ Loaded {len(feature_names)} features from: {FEATURES_PATH}")

    # --- Load Data ---
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"❌ Data file not found: {DATA_PATH}")
    data_obj = joblib.load(DATA_PATH)

    # Handle dict or tuple formats
    if isinstance(data_obj, dict):
        X = data_obj.get("X")
        y = data_obj.get("y")
    elif isinstance(data_obj, tuple) and len(data_obj) == 2:
        X, y = data_obj
    else:
        raise ValueError(f"❌ Unexpected format in {DATA_PATH}: {type(data_obj)}")

    # Convert to DataFrame if needed
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X, columns=feature_names)

    print(f"✅ Data loaded: {X.shape[0]} samples, {X.shape[1]} features")

    # --- Permutation Importance ---
    print("⏳ Computing permutation importance...")
    pi = permutation_importance(model, X, y, n_repeats=10, random_state=42, n_jobs=-1)
    importances = pd.DataFrame({
        "feature": feature_names,
        "importance_mean": pi.importances_mean,
        "importance_std": pi.importances_std
    }).sort_values(by="importance_mean", ascending=False)

    print("✅ Permutation importance computed.")
    print(importances.head(10))

    # Save permutation importance
    out_csv = "outputs/synthetic_permutation_importance.csv"
    os.makedirs("outputs", exist_ok=True)
    importances.to_csv(out_csv, index=False)
    print(f"💾 Saved permutation importance to {out_csv}")

    # --- SHAP Values ---
    print("⏳ Computing SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    print("✅ SHAP values computed.")

    # Save SHAP summary plot
    shap_summary_path = "outputs/synthetic_shap_summary.png"
    shap.summary_plot(shap_values, X, show=False)
    import matplotlib.pyplot as plt
    plt.savefig(shap_summary_path, bbox_inches="tight")
    plt.close()
    print(f"💾 SHAP summary plot saved to {shap_summary_path}")

    print("\n🎯 Done! Synthetic analysis complete.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"💥 ERROR: {e}")
        raise