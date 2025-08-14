# src/train_synthetic_rf.py

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from pathlib import Path

# ----------------------------
# Paths
# ----------------------------
DATA_PATH = Path("data/symptom_dataset_simulated.csv")
MODEL_PATH = Path("models/synthetic_rf_model.joblib")
FEATURES_PATH = Path("models/synthetic_rf_features.joblib")
DATA_SAVE_PATH = Path("models/synthetic_rf_data.joblib")
RUNS_PATH = Path("models/synthetic_rf_cv30_metrics.csv")

# ----------------------------
# Load dataset
# ----------------------------
print("📂 Loading Synthetic dataset...")
df = pd.read_csv(DATA_PATH)
print(f"✅ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# ----------------------------
# Features & target
# ----------------------------
target_column = "diagnosis"
if target_column not in df.columns:
    raise KeyError(f"❌ Target column '{target_column}' not found.")

X = df.drop(columns=[target_column])
y = df[target_column]

# One-hot encode categorical features
X_encoded = pd.get_dummies(X, drop_first=True)

# Save feature names for SHAP later
feature_names = list(X_encoded.columns)

# ----------------------------
# Save X and y for SHAP later
# ----------------------------
joblib.dump({"X": X_encoded, "y": y}, DATA_SAVE_PATH)
joblib.dump(feature_names, FEATURES_PATH)

# ----------------------------
# Train model with CV
# ----------------------------
rf = RandomForestClassifier(n_estimators=200, random_state=42)

cv_results = cross_validate(
    rf, X_encoded, y,
    cv=30,
    scoring=["accuracy", "precision_macro", "recall_macro", "f1_macro"],
    return_train_score=False,
    n_jobs=-1
)

# Save CV metrics
cv_df = pd.DataFrame(cv_results)
cv_df.to_csv(RUNS_PATH, index=False)

# Fit final model on all data
rf.fit(X_encoded, y)

# Save trained model
joblib.dump(rf, MODEL_PATH)

print(f"✅ Model saved to {MODEL_PATH}")
print(f"✅ Features saved to {FEATURES_PATH}")
print(f"✅ Data (X, y) saved to {DATA_SAVE_PATH}")
print(f"✅ CV metrics saved to {RUNS_PATH}")

print("\n📊 Mean CV Scores:")
print(cv_df.mean())