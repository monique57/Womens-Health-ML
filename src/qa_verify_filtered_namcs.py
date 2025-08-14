import pandas as pd
from pathlib import Path

IN = Path("data/namcs2019.csv")
OUT = Path("data/namcs2019_female_LFD.csv")

# Step 1: Check file existence
if not IN.exists():
    print(f"❌ File not found: {IN.resolve()}")
    exit()

df = pd.read_csv(IN, low_memory=False)
print(f"✅ Loaded {IN.name}: {df.shape[0]} rows, {df.shape[1]} cols")

# Step 2: Map ICD to disease
def map_icd_to_disease(row):
    d = [str(row.get(c, "")) for c in ["DIAG1","DIAG2","DIAG3","DIAG4","DIAG5"] if c in row]
    d = [x.strip().upper() for x in d if pd.notna(x)]
    icd_str = " ".join(d)

    if (("F32" in icd_str) or ("F33" in icd_str) or ("F34" in icd_str)
        or row.get("DEPRN", 0) == 1 or row.get("DEPRESS", 0) == 1):
        return "Depression"
    if "M32" in icd_str:
        return "Lupus"
    if "M797" in icd_str or "M79.7" in icd_str:
        return "Fibromyalgia"
    return None

df["disease"] = df.apply(map_icd_to_disease, axis=1)

# Step 3: Filter female + target diseases
female_mask = df["SEX"].astype(float) == 2.0
lfd_mask = df["disease"].isin(["Depression","Lupus","Fibromyalgia"])
subset = df.loc[female_mask & lfd_mask].copy()

print(f"✅ Filtered rows: {subset.shape[0]} out of {df.shape[0]} total")
print("\n📊 Class counts:")
print(subset["disease"].value_counts())

# Step 4: Save filtered file
subset.to_csv(OUT, index=False)
print(f"💾 Saved filtered dataset to {OUT.resolve()}")