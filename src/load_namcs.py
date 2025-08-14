# src/load_namcs.py
import os
from pathlib import Path
import pyreadstat
import pandas as pd

# 1) Show where we are
cwd = Path.cwd()
here = Path(__file__).resolve().parent
project_root = here.parent  # assuming src/ is under the project root
print(f"📍 Current working dir: {cwd}")
print(f"📍 Script directory    : {here}")
print(f"📍 Project root        : {project_root}")

# 2) Search recursively for any .sav under the project root
candidates = list(project_root.rglob("*.sav"))
print("\n🔎 Found the following .sav files:")
for p in candidates:
    print(" -", p)

if not candidates:
    raise FileNotFoundError(
        "❌ No .sav files found anywhere under the project folder.\n"
        "• Ensure the file is named like NAMCS2019-spss.sav\n"
        "• Place it in: data/ (recommended)\n"
        "• Or anywhere under the project; this script will find it."
    )

# 3) Pick the most recently modified .sav
sav_path = max(candidates, key=lambda p: p.stat().st_mtime)
print(f"\n📄 Using file: {sav_path}")

# 4) Load with pyreadstat
try:
    df, meta = pyreadstat.read_sav(str(sav_path))
except Exception as e:
    raise RuntimeError(f"❌ Failed to load {sav_path}:\n{e}")

print("\n✅ Dataset Loaded Successfully")
print(f"📊 Shape: {df.shape}")
print("\n📌 First 5 rows:")
print(df.head())

# 5) Ensure data/ exists and save CSV for later steps
data_dir = project_root / "data"
data_dir.mkdir(exist_ok=True)
csv_out = data_dir / "namcs2019.csv"
df.to_csv(csv_out, index=False)
print(f"\n💾 Saved CSV to: {csv_out.resolve()}")