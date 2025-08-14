import os, glob

print("📂 Current working directory:", os.getcwd(), flush=True)

namcs_files = glob.glob("models/namcs_rf*_metrics.csv") + glob.glob("models/namcs_rf_runs.csv")
synthetic_files = glob.glob("models/synthetic_rf*_metrics.csv") + glob.glob("models/synthetic_rf_runs.csv")

print("📄 NAMCS files found:", namcs_files, flush=True)
print("📄 Synthetic files found:", synthetic_files, flush=True)

if not namcs_files:
    print("❌ No NAMCS metrics files found in models/")
if not synthetic_files:
    print("❌ No Synthetic metrics files found in models/")