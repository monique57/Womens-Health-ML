# src/train_namcs2019.py
import os
import json
import numpy as np
import pandas as pd
import joblib
from math import ceil
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# ============== CONFIG ==============
DATA_PATH = "data/namcs2019_clean.csv"     # your cleaned CSV
TEST_SIZE = 0.30
RUNS = 30
RANDOM_SEED = 42
MIN_SAMPLES_PER_CLASS = 5                   # remove rare classes
N_QUANTILE_BINS = 5                         # if target is continuous, bin into these categories
SAVE_DIR = "models"
MODEL_PATH = os.path.join(SAVE_DIR, "best_model_namcs2019.joblib")
LABELS_PATH = os.path.join(SAVE_DIR, "class_labels.json")
# ====================================

def detect_target_column(df: pd.DataFrame) -> str:
    keys = ["illness", "diagnosis", "disease", "condition", "target", "label"]
    for c in df.columns:
        if any(k in c.lower() for k in keys):
            return c
    return df.columns[-1]  # fallback: last column

def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ],
        remainder="drop",
        sparse_threshold=1.0,
    )
    return pre

def safe_split(X, y, test_size, seed):
    """Try stratified split; if impossible, fall back to non-stratified."""
    try:
        return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
    except Exception:
        print("⚠️  Stratified split not feasible → using non-stratified split.")
        return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=None)

# 1) Load
df = pd.read_csv(DATA_PATH, low_memory=False)
print(f"Loaded dataset: {df.shape}")

# 2) Target
target_col = detect_target_column(df)
print(f"🎯 Target column: {target_col}")
df = df.dropna(subset=[target_col])

# 3) Force CLASSIFICATION
y_raw = df[target_col]

# If target looks continuous (numeric with many unique values), bin into quantile categories
if np.issubdtype(y_raw.dtype, np.number) and y_raw.nunique(dropna=True) > 20:
    # bin into quantiles; labels: 0..(bins-1)
    df[target_col] = pd.qcut(y_raw, q=min(N_QUANTILE_BINS, y_raw.nunique()), labels=False, duplicates="drop")
    class_label_map = {int(k): f"Q{k+1}" for k in sorted(df[target_col].unique())}
    print(f"ℹ️ Target was continuous. Binned into {len(class_label_map)} quantile classes.")
else:
    # Coerce to category codes if not already small integer labels
    if y_raw.dtype == "object" or str(y_raw.dtype).startswith("category"):
        df[target_col] = y_raw.astype("category").cat.codes
    else:
        # numeric but small unique set → treat as categorical codes directly
        df[target_col] = y_raw.astype(int, errors="ignore")
    # create readable class map from original values (best-effort)
    # NOTE: If original was string/category, above conversion loses names; we’ll just map codes to strings
    unique_vals = sorted(df[target_col].unique())
    class_label_map = {int(c): f"class_{int(c)}" for c in unique_vals}

# 4) Remove rare classes
counts = df[target_col].value_counts()
keep = counts[counts >= MIN_SAMPLES_PER_CLASS].index
removed = set(df[target_col].unique()) - set(keep)
if removed:
    print(f"⚠️  Removing {len(removed)} rare classes (<{MIN_SAMPLES_PER_CLASS} samples).")
    df = df[df[target_col].isin(keep)]

# 5) X / y
X = df.drop(columns=[target_col])
y = df[target_col].values
# If still too many classes relative to test size, warn (we’ll fall back inside safe_split when needed)
n_classes = len(np.unique(y))
n_test = ceil(len(y) * TEST_SIZE)
if n_classes > n_test:
    print(f"⚠️  Many classes ({n_classes}) vs test size ({n_test}): stratification may be disabled per run.")

# 6) Preprocessor
preprocessor = build_preprocessor(X)

# 7) Models
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000, n_jobs=None),
    "RandomForest": RandomForestClassifier(random_state=RANDOM_SEED),
    "SVM_RBF": SVC(kernel="rbf", probability=True, random_state=RANDOM_SEED),
    "KNN": KNeighborsClassifier(),
}

# 8) 30-run training & evaluation (macro-F1)
scores = {name: [] for name in models}
for i in range(RUNS):
    seed = RANDOM_SEED + i + 1
    X_train, X_test, y_train, y_test = safe_split(X, y, TEST_SIZE, seed)

    for name, clf in models.items():
        pipe = Pipeline(steps=[("preprocessor", preprocessor), ("model", clf)])
        try:
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)
            f1m = f1_score(y_test, y_pred, average="macro", zero_division=0)
            scores[name].append(f1m)
            print(f"Run {i+1:02d} | {name:<18} F1-macro={f1m:.4f}")
        except Exception as e:
            print(f"⚠️  Skipping {name} on run {i+1}: {e}")

# 9) Pick best by mean F1
mean_scores = {k: (np.mean(v) if len(v) else -np.inf) for k, v in scores.items()}
print("\n📊 Average Macro-F1 over 30 runs:")
for k, v in mean_scores.items():
    val = "N/A" if v == -np.inf else f"{v:.4f}"
    print(f"  {k:<18} {val}")

best_name = max(mean_scores, key=mean_scores.get)
print(f"\n🏆 Best model: {best_name} (Macro-F1={mean_scores[best_name]:.4f})")

# 10) Retrain best on ALL data and save pipeline
best_model = models[best_name]
final_pipe = Pipeline(steps=[("preprocessor", preprocessor), ("model", best_model)])
final_pipe.fit(X, y)

os.makedirs(SAVE_DIR, exist_ok=True)
# clean stale files with same name to avoid version confusion
if os.path.exists(MODEL_PATH):
    try:
        os.remove(MODEL_PATH)
    except Exception:
        pass

joblib.dump(final_pipe, MODEL_PATH)
with open(LABELS_PATH, "w", encoding="utf-8") as f:
    json.dump(class_label_map, f, ensure_ascii=False, indent=2)

print(f"✅ Saved pipeline to: {MODEL_PATH}")
print(f"✅ Saved class label map to: {LABELS_PATH}")