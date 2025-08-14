import os, shutil, glob, json, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FINAL = RESULTS / "final_report"
CROSS = RESULTS / "crossval"
STAB  = RESULTS / "stability"
OUTS  = ROOT / "outputs"
MODELS = ROOT / "models"

FINAL.mkdir(parents=True, exist_ok=True)

def cp(patterns, dest):
    dest.mkdir(parents=True, exist_ok=True)
    copied = []
    for pat in patterns:
        for p in glob.glob(str(pat), recursive=True):
            src = Path(p)
            if src.is_file():
                tgt = dest / src.name
                try:
                    shutil.copy2(src, tgt)
                    copied.append(tgt.name)
                except Exception as e:
                    print(f"  ⚠️  Skip {src}: {e}")
    return copied

print("📦 Collecting results → results/final_report")

# 1) Cross-validation artifacts
cv_files = []
if CROSS.exists():
    cv_files += cp([CROSS / "kfold_metrics.csv",
                    CROSS / "kfold_summary.csv",
                    CROSS / "plots" / "*.png"], FINAL)
else:
    print("  ℹ️ No crossval folder found.")

# 2) Stability artifacts
stab_files = []
if STAB.exists():
    stab_files += cp([STAB / "*.csv", STAB / "plots" / "*.png"], FINAL)
else:
    print("  ℹ️ No stability folder found.")

# 3) Test/regression summary if you saved one in outputs
test_files = cp([OUTS / "regression_summary.csv"], FINAL)

# 4) SHAP candlestick image (search in results and outputs)
shap_files = cp([RESULTS / "*shap*.*png", OUTS / "*shap*.*png"], FINAL)

# 5) Record best model artifact if present
best_model = None
for name in ["best_model_namcs2019.joblib", "best_model_v2.joblib"]:
    if (MODELS / name).exists():
        best_model = name
        break

meta = {
    "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    "best_model": best_model,
    "copied": {
        "crossval": cv_files,
        "stability": stab_files,
        "test": test_files,
        "shap": shap_files
    }
}
with open(FINAL / "manifest.json", "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2)

# 6) Friendly README
readme = FINAL / "README.md"
readme.write_text(
f"""# Final Report Bundle

Created: {meta['created_at']}

## Contents
- Cross-validation: {', '.join(cv_files) if cv_files else '—'}
- Stability: {', '.join(stab_files) if stab_files else '—'}
- Test summary: {', '.join(test_files) if test_files else '—'}
- SHAP candlestick: {', '.join(shap_files) if shap_files else '—'}

Best model artifact: {best_model or '—'}

> All files in this folder are ready to be inserted into the thesis (figures & tables).
""", encoding="utf-8"
)

print("✅ Done. See:", FINAL)
