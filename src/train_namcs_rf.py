# src/train_namcs_rf.py
# Train a RandomForest on NAMCS 2019 (women-focused) and save model + artifacts.

import os
from pathlib import Path
import re
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.utils import check_random_state
from imblearn.over_sampling import RandomOverSampler
import joblib

# ----------------------------
# 1) Load data (filtered first, then raw fallback)
# ----------------------------
DATA_DIR = Path("data")
CANDIDATES = [
    DATA_DIR / "namcs2019_filtered.csv",   # your women-only filtered subset
    DATA_DIR / "namcs2019.csv",            # full export from .sav
]

def load_first_existing(paths):
    for p in paths:
        if p.exists():
            return p
    raise FileNotFoundError(f"None of these files exist: {paths}")

data_path = load_first_existing(CANDIDATES)
print("📂 Loading NAMCS 2019 dataset from:", data_path)

df = pd.read_csv(data_path, low_memory=False)
print(f"✅ Dataset loaded: {len(df)} rows, {df.shape[1]} columns")

# ----------------------------
# 2) Build the target label disease
#    Priority: Lupus > Fibromyalgia > Depression
# ----------------------------
ICD_COLS = [c for c in df.columns if re.fullmatch(r"DIAG[1-5]", str(c))]
DEPR_FLAGS = [c for c in df.columns if c.upper() in {"DEPRN", "DEPRESS"}]

def any_icd_starts_with(row, prefixes):
    for c in ICD_COLS:
        val = str(row.get(c, "")).upper().strip()
        if val in {"", "-9", "-7", "NAN"}:
            continue
        # remove dots to match variants (e.g., M79.7 vs M797)
        val_nodot = val.replace(".", "")
        for p in prefixes:
            if val.startswith(p) or val_nodot.startswith(p.replace(".", "")):
                return True
    return False

def is_flag_positive(row, flags):
    for c in flags:
        v = row.get(c, np.nan)
        try:
            fv = float(v)
        except Exception:
            continue
        # NAMCS uses 1=yes, 0=no, negatives = missing/skip
        if fv == 1.0:
            return True
    return False

# ICD-10 groupings
LUPUS_PREFIXES = ["M32", "L93"]                     # SLE & cutaneous lupus
FIBRO_PREFIXES = ["M79.7", "M797"]                  # fibromyalgia variants
DEPR_PREFIXES  = ["F32", "F33", "F341"]             # major depressive disorders, dysthymia

def map_icd_to_disease(row):
    # Lupus
    if any_icd_starts_with(row, LUPUS_PREFIXES):
        return "Lupus"
    # Fibromyalgia
    if any_icd_starts_with(row, FIBRO_PREFIXES):
        return "Fibromyalgia"
    # Depression via ICD
    if any_icd_starts_with(row, DEPR_PREFIXES):
        return "Depression"
    # Depression via flags if present
    if is_flag_positive(row, DEPR_FLAGS):
        return "Depression"
    return np.nan

df["disease"] = df.apply(map_icd_to_disease, axis=1)

# Keep rows with a defined target
df = df[~df["disease"].isna()].copy()
if df.empty:
    raise RuntimeError("❌ After mapping, no rows have a 'disease' label. Check inputs.")

print("🔎 Class counts:\n", df["disease"].value_counts(dropna=False))

# ----------------------------
# 3) Feature selection
#    Use reasonable clinical + visit variables. Drop obvious IDs/weights.
# ----------------------------
drop_cols = {
    "CPSUM","CSTRATM","PATWT","PHYSWT","YEAR","SETTYPE",
    "PHYCODE","PATCODE"  # facility/patient linkage codes
}
candidate_features = [c for c in df.columns if c not in (drop_cols | {"disease"})]

# remove all-diag text columns from features (they fed the target)
feature_cols = [c for c in candidate_features if c not in ICD_COLS]

# separate into numeric/categorical by dtype
num_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
cat_cols = [c for c in feature_cols if c not in num_cols]
# If everything numeric, keep; if everything categorical, still okay — pipeline will handle it.
print(f"🧩 Using {len(feature_cols)} features -> numeric: {len(num_cols)}, categorical: {len(cat_cols)}")

X_full = df[feature_cols].copy()
y_full = df["disease"].astype("category")

# ----------------------------
# 4) Preprocessing + Model pipeline
# ----------------------------
preprocess = ColumnTransformer(
    transformers=[
        ("num", "passthrough", num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
    ]
)

rf = RandomForestClassifier(
    n_estimators=400,
    max_depth=None,
    class_weight="balanced_subsample",
    n_jobs=-1,
    random_state=42
)

pipe = Pipeline(steps=[
    ("pre", preprocess),
    ("clf", rf)
])

# ----------------------------
# 5) 30 independent splits with oversampling + metrics
# ----------------------------
runs = []
cv_seed = 2025  # base seed for reproducibility
sss = StratifiedShuffleSplit(n_splits=30, test_size=0.2, random_state=cv_seed)

# If a class has <2 samples, StratifiedShuffleSplit fails. Handle that gracefully:
labels, counts = np.unique(y_full, return_counts=True)
if counts.min() < 2:
    # fallback: manual 80/20 split without stratify (still oversample in train)
    print("⚠️ Least populated class has <2 samples. Using non-stratified repeated random splits.")
    rng = check_random_state(cv_seed)
    indices = np.arange(len(X_full))
    for i in range(30):
        rng.shuffle(indices)
        cut = int(0.8 * len(indices))
        tr_idx, te_idx = indices[:cut], indices[cut:]
        X_tr, X_te = X_full.iloc[tr_idx], X_full.iloc[te_idx]
        y_tr, y_te = y_full.iloc[tr_idx], y_full.iloc[te_idx]

        # fit transform on train, transform test
        X_tr_enc = pipe.named_steps["pre"].fit_transform(X_tr)
        X_te_enc = pipe.named_steps["pre"].transform(X_te)

        # oversample train encodings
        ros = RandomOverSampler(random_state=42)
        X_tr_bal, y_tr_bal = ros.fit_resample(X_tr_enc, y_tr)

        # clone a fresh RF for each run
        clf = RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=42 + i
        )
        clf.fit(X_tr_bal, y_tr_bal)

        # evaluate
        y_pred = clf.predict(X_te_enc)
        acc = accuracy_score(y_te, y_pred)
        pr, rc, f1, _ = precision_recall_fscore_support(y_te, y_pred, average="macro", zero_division=0)
        runs.append({"Run": i+1, "Accuracy": acc, "Precision": pr, "Recall": rc, "F1": f1})

    # After runs, refit full pipeline on all data (with oversampling)
    X_all_enc = pipe.named_steps["pre"].fit_transform(X_full)
    ros = RandomOverSampler(random_state=42)
    X_all_bal, y_all_bal = ros.fit_resample(X_all_enc, y_full)
    final_rf = RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=4242
    )
    final_rf.fit(X_all_bal, y_all_bal)

    # Build a final pipeline object that includes a frozen preprocessor and rf
    # (we’ll wrap manually because we trained rf on encoded features)
    model_artifact = {
        "preprocessor": pipe.named_steps["pre"],
        "classifier": final_rf,
        "feature_names": None  # filled below
    }
    # derive feature names from preprocessor
    try:
        feature_names = pipe.named_steps["pre"].get_feature_names_out()
    except Exception:
        feature_names = None
else:
    print("🚀 RandomForest: 30 independent stratified runs")
    # normal path with stratified splits
    feature_names = None
    for i, (tr_idx, te_idx) in enumerate(sss.split(X_full, y_full), start=1):
        X_tr, X_te = X_full.iloc[tr_idx], X_full.iloc[te_idx]
        y_tr, y_te = y_full.iloc[tr_idx], y_full.iloc[te_idx]

        # Fit preprocessor on train only
        pre = ColumnTransformer(
            transformers=[
                ("num", "passthrough", num_cols),("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
            ]
        )
        X_tr_enc = pre.fit_transform(X_tr)
        X_te_enc = pre.transform(X_te)

        # Oversample training set to balance classes
        ros = RandomOverSampler(random_state=42 + i)
        X_tr_bal, y_tr_bal = ros.fit_resample(X_tr_enc, y_tr)

        # Fresh RF each run
        clf = RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=42 + i
        )
        clf.fit(X_tr_bal, y_tr_bal)

        y_pred = clf.predict(X_te_enc)
        acc = accuracy_score(y_te, y_pred)
        pr, rc, f1, _ = precision_recall_fscore_support(y_te, y_pred, average="macro", zero_division=0)
        runs.append({"Run": i, "Accuracy": acc, "Precision": pr, "Recall": rc, "F1": f1})

        # keep the last run’s preprocessor feature names (they’re compatible across runs)
        try:
            feature_names = pre.get_feature_names_out()
        except Exception:
            feature_names = None

    # Refit on ALL data for the final model:
    pre_final = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_cols),
            ("cat", OnseHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ]
    )
    X_all_enc = pre_final.fit_transform(X_full)
    ros = RandomOverSampler(random_state=4242)
    X_all_bal, y_all_bal = ros.fit_resample(X_all_enc, y_full)

    final_rf = RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=4242
    )
    final_rf.fit(X_all_bal, y_all_bal)

    model_artifact = {
        "preprocessor": pre_final,
        "classifier": final_rf,
        "feature_names": None  # filled below
    }
    try:
        feature_names = pre_final.get_feature_names_out()
    except Exception:
        feature_names = None

# ----------------------------
# 6) Persist artifacts
# ----------------------------
Path("models").mkdir(exist_ok=True)

# Save a compact object with both steps (preprocessor + classifier)
joblib.dump(model_artifact, "models/namcs_rf_model.joblib")
if feature_names is not None:
    joblib.dump(list(feature_names), "models/namcs_rf_features.joblib")
else:
    # fallback: save numeric + categorical lists as a hint
    joblib.dump({"num_cols": num_cols, "cat_cols": cat_cols}, "models/namcs_rf_features.joblib")

# Save the *raw* (pre-encoding) X/y used to fit the final model
joblib.dump({"X": X_full, "y": y_full}, "models/namcs_rf_data.joblib")

# Save metrics
runs_df = pd.DataFrame(runs)
runs_df.to_csv("models/namcs_rf_runs.csv", index=False)
summary = runs_df[["Accuracy", "Precision", "Recall", "F1"]].mean().to_frame(name="Mean").T
summary.to_csv("models/namcs_rf_cv30_metrics.csv", index=False)

print("✅ Saved:")
print(" - models/namcs_rf_model.joblib")
print(" - models/namcs_rf_features.joblib")
print(" - models/namcs_rf_data.joblib")
print(" - models/namcs_rf_runs.csv")
print(" - models/namcs_rf_cv30_metrics.csv")