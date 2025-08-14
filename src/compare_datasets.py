# src/compare_datasets.py
import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

NAMCS_PATH = "models/namcs_rf_cv30_metrics.csv"
SYNTH_PATH = "models/synthetic_rf_cv30_metrics.csv"
OUT_PNG = "models/model_comparison_v2.png"

ALIASES = {
    "accuracy": ["accuracy", "acc", "acc_score", "accuracy_mean"],
    "precision": ["precision", "prec", "precision_mean"],
    "recall": ["recall", "tpr", "sensitivity", "recall_mean"],
    "f1": ["f1", "f1_score", "f1score", "f1_mean"],
}

def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # lowercase + strip spaces
    df = df.rename(columns={c: re.sub(r"\s+", "", c.lower()) for c in df.columns})
    # build mapping from aliases -> canonical
    colmap = {}
    for canon, options in ALIASES.items():
        for opt in options:
            opt_n = re.sub(r"\s+", "", opt.lower())
            if opt_n in df.columns:
                colmap[opt_n] = canon
                break
    # If some metrics are missing, show what exists to help debugging
    missing = [m for m in ["accuracy", "precision", "recall", "f1"]
               if m not in colmap.values()]
    if missing:
        print("⚠️ Could not find columns for:", missing)
        print("   Available columns:", list(df.columns))
    # Apply rename for the metrics we did find
    inv = {k: v for k, v in colmap.items()}
    df = df.rename(columns=inv)
    return df

def load_metrics(path: str, label: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    df = pd.read_csv(path)
    df = _standardize_columns(df)
    needed = [c for c in ["accuracy", "precision", "recall", "f1"] if c in df.columns]
    if not needed:
        raise KeyError(
            f"No recognizable metric columns found in {path}. "
            f"Columns present: {list(df.columns)}"
        )
    df = df[needed].copy()
    df["dataset"] = label
    return df

def main():
    df_namcs = load_metrics(NAMCS_PATH, "NAMCS")
    df_synth = load_metrics(SYNTH_PATH, "Synthetic")

    df_all = pd.concat([df_namcs, df_synth], ignore_index=True)

    # Print means
    metrics = [m for m in ["accuracy", "precision", "recall", "f1"] if m in df_all.columns]
    print("\n📊 Mean performance by dataset:")
    means = df_all.groupby("dataset")[metrics].mean()
    print(means.round(4))

    # Plot bar chart of means
    x = np.arange(len(metrics))
    width = 0.35
    fig, ax = plt.subplots()
    namcs_vals = means.loc["NAMCS", metrics].values
    synth_vals = means.loc["Synthetic", metrics].values

    ax.bar(x - width/2, namcs_vals, width, label="NAMCS")
    ax.bar(x + width/2, synth_vals, width, label="Synthetic")

    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in metrics])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Random Forest Performance: NAMCS vs. Synthetic")
    ax.legend()
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    fig.savefig(OUT_PNG, dpi=150)
    print(f"\n✅ Saved comparison plot → {OUT_PNG}")

if __name__ == "__main__":
    main()