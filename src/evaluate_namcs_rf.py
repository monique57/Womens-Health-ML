# src/evaluate_namcs_rf.py

import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score

DATA_PATH = "data/namcs2019.csv"
OUT_DIR = "models"
RUNS_CSV = os.path.join(OUT_DIR, "namcs_rf_eval_runs.csv")
REPORTS_CSV = os.path.join(OUT_DIR, "namcs_rf_eval_reports.csv")

os.makedirs(OUT_DIR, exist_ok=True)

print("📂 Loading NAMCS 2019 dataset...")
df = pd.read_csv(DATA_PATH, low_memory=False)
print(f"✅ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# --- ICD→label mapping (Depression, Lupus, Fibromyalgia) ---
required_diag_cols = ["DIAG1", "DIAG2", "DIAG3"]
for c in required_diag_cols:
    if c not in df.columns:
        raise ValueError(f"Required diagnosis column '{c}' not found.")

def map_disease(row):
    for c in required_diag_cols:
        code = str(row.get(c, "")).strip()
        if not code or code in {"-9", "-8", "-7", "ZZZ0", "ZZZZ", "nan", "NaN"}:
            continue
        if code.startswith(("F32", "F33")):
            return "Depression"
        if code.startswith("M32"):
            return "Lupus"
        if code.startswith("M797"):
            return "Fibromyalgia"
    return None

print("🔎 Creating target labels from DIAG1/2/3 ...")
df["disease"] = df.apply(map_disease, axis=1)
df = df[df["disease"].notna()].copy()
if df.empty:
    raise ValueError("No rows matched the three target illnesses after mapping.")

print("✅ Disease counts:")
print(df["disease"].value_counts())

# Drop DIAG/PRDIAG to avoid leakage
leak_cols = [c for c in df.columns if c.upper().startswith(("DIAG", "PRDIAG"))]
df.drop(columns=leak_cols, inplace=True, errors="ignore")

y = df["disease"].astype(str)
X = df.drop(columns=["disease"])

# Handle NAMCS sentinels and coercion
MISSING_SENTINELS = {-7, -8, -9, -900, -999}
def normalize_missing(v):
    try:
        f = float(v)
        return np.nan if f in MISSING_SENTINELS else f
    except Exception:
        return v

X = X.applymap(normalize_missing)
X = X.apply(pd.to_numeric, errors="coerce")

all_nan_cols = X.columns[X.isna().all()].tolist()
if all_nan_cols:
    X.drop(columns=all_nan_cols, inplace=True)

X = X.fillna(X.median(numeric_only=True))
nunique = X.nunique(dropna=False)
zero_var_cols = nunique[nunique <= 1].index.tolist()
if zero_var_cols:
    X.drop(columns=zero_var_cols, inplace=True)

print(f"🧹 Features ready: {X.shape[1]} columns "
      f"(dropped {len(all_nan_cols)} all-NaN, {len(zero_var_cols)} zero-variance)")

# Encode labels
le = LabelEncoder()
y_enc = le.fit_transform(y)
class_names = list(le.classes_)
print(f"🏷 Classes: {class_names}")

# 30-run evaluation (train+test per run, different seeds)
print("🧪 Starting 30-run evaluation of RandomForest...")

records = []
reports_list = []

for run in range(30):
    # Split anew each run
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=run, stratify=y_enc
    )

    # Fit scaler ONLY on training data of this run
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # Train a fresh RF for this run (evaluation pass)
    rf = RandomForestClassifier(
        n_estimators=400,
        class_weight="balanced",
        random_state=run,
        n_jobs=-1
    )
    rf.fit(X_train_s, y_train)

    y_pred = rf.predict(X_test_s)

    acc = accuracy_score(y_test, y_pred)
    f1w = f1_score(y_test, y_pred, average="weighted")
    f1m = f1_score(y_test, y_pred, average="macro")

    rep = classification_report(
        y_test, y_pred, target_names=class_names, output_dict=True, zero_division=0
    )
    rep_df = pd.DataFrame(rep).transpose()
    rep_df["run"] = run + 1
    reports_list.append(rep_df)

    print(f"\n📊 Run {run+1}")
    print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))
    print(f"Accuracy={acc:.4f} | F1-weighted={f1w:.4f} | F1-macro={f1m:.4f}")

    records.append({"run": run+1, "accuracy": acc, "f1_weighted": f1w, "f1_macro": f1m})

# Save summaries
runs_df = pd.DataFrame(records)
runs_df.to_csv(RUNS_CSV, index=False)

reports_df = pd.concat(reports_list, axis=0)
reports_df.to_csv(REPORTS_CSV, index=True)

print("\n✅ 30-run evaluation complete.")
print("📈 Averages across 30 runs:")
print(runs_df[["accuracy", "f1_weighted", "f1_macro"]].mean())

print(f"\n💾 Saved per-run metrics → {RUNS_CSV}")
print(f"💾 Saved full classification reports → {REPORTS_CSV}")