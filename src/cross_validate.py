import os, numpy as np, pandas as pd, joblib, matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.utils.multiclass import type_of_target
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             r2_score, mean_absolute_error, mean_squared_error)

DATA_PATH  = "data/namcs2019_clean.csv"
MODEL_PATH = "models/best_model_namcs2019.joblib"
OUT_DIR    = "results/crossval"
PLOTS_DIR  = os.path.join(OUT_DIR, "plots")
K_SPLITS, RANDOM_SEED = 5, 42
os.makedirs(PLOTS_DIR, exist_ok=True)

def detect_target_column(df: pd.DataFrame) -> str:
    keys = ["illness","diagnosis","disease","condition","target","label"]
    for c in df.columns:
        if any(k in c.lower() for k in keys): return c
    return df.columns[-1]

def cls_metrics(y_true, y_pred):
    return dict(
        accuracy=accuracy_score(y_true, y_pred),
        precision_weighted=precision_score(y_true, y_pred, average="weighted", zero_division=0),
        recall_weighted=recall_score(y_true, y_pred, average="weighted", zero_division=0),
        f1_weighted=f1_score(y_true, y_pred, average="weighted", zero_division=0),
    )

def reg_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return dict(r2=r2_score(y_true, y_pred), mae=mean_absolute_error(y_true, y_pred),
                mse=mse, rmse=np.sqrt(mse))

def save_box(vals, metric, title, fname):
    vals = np.asarray(vals, float)
    plt.figure(figsize=(6,5))
    plt.boxplot(vals, vert=True, showmeans=False)
    plt.axhline(vals.mean(), linestyle="--", linewidth=1.5)
    plt.title(f"{title} (n={len(vals)})"); plt.ylabel(metric); plt.tight_layout()
    path = os.path.join(PLOTS_DIR, fname); plt.savefig(path, dpi=300); plt.close()
    print(f"📈 Saved {path}")

print("🔌 Loading:", MODEL_PATH, "and", DATA_PATH)
pipe = joblib.load(MODEL_PATH)
df = pd.read_csv(DATA_PATH, low_memory=False)
target = detect_target_column(df)
print("🎯 Target column:", target)

df = df.dropna(subset=[target])
X, y_raw = df.drop(columns=[target]), df[target]
y_type = type_of_target(y_raw)
is_cls = y_type in ["binary","multiclass"]
y = y_raw.astype("category").cat.codes if (is_cls and y_raw.dtype=="object") else (
    pd.to_numeric(y_raw, errors="coerce") if not is_cls else y_raw)
if not is_cls:
    mask = ~pd.isna(y); X, y = X[mask], y[mask]

print(f"🧪 {K_SPLITS}-fold evaluation (task={'classification' if is_cls else 'regression'})")
rows = []
splitter = StratifiedKFold(n_splits=K_SPLITS, shuffle=True, random_state=RANDOM_SEED) if is_cls \
           else KFold(n_splits=K_SPLITS, shuffle=True, random_state=RANDOM_SEED)

for i,(tr,te) in enumerate(splitter.split(X, y if is_cls else None),1):
    Xte, yte = X.iloc[te], y.iloc[te] if hasattr(y,"iloc") else y[te]
    yhat = pipe.predict(Xte)
    rows.append({"fold": i, **(cls_metrics(yte, yhat) if is_cls else reg_metrics(yte, yhat))})

fold_df = pd.DataFrame(rows)
kcsv = os.path.join(OUT_DIR, "kfold_metrics.csv"); fold_df.to_csv(kcsv, index=False)
print("🧾 Saved", kcsv)

metrics = ["accuracy","precision_weighted","recall_weighted","f1_weighted"] if is_cls \
          else ["r2","mae","mse","rmse"]
for m in metrics: save_box(fold_df[m].values, m, f"K-Fold {m.upper()} Stability", f"kfold_{m}_box.png")

summary = fold_df.describe().T[["mean","std","min","max"]]
scsv = os.path.join(OUT_DIR, "kfold_summary.csv"); summary.to_csv(scsv)
print("🧾 Saved", scsv)

print("✅ Done. Open this folder:", os.path.abspath(OUT_DIR))