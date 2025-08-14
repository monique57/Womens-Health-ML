import pandas as pd
import re

# === CONFIG ===
INPUT_FILE = "data/NAMCS2019-spss.sav"  # Your SPSS file
OUTPUT_FILE = "data/namcs2019_clean.csv"

# === LOAD DATA ===
print("Loading dataset...")
df = pd.read_spss(INPUT_FILE)
print(f"Original shape: {df.shape}")

# === STANDARDIZE COLUMN NAMES ===
df.columns = df.columns.str.strip().str.lower()

# === FIND GENDER COLUMN ===
gender_col = None
for col in df.columns:
    if re.search(r'gender|sex', col):
        gender_col = col
        break

if gender_col is None:
    raise ValueError("No gender/sex column found in the dataset!")

# === FILTER FOR WOMEN ===
female_values = ['female', 'f', 2, '2']
df = df[df[gender_col].astype(str).str.lower().isin([str(v).lower() for v in female_values])]
print(f"After filtering women: {df.shape}")

# === DETECT ILLNESS COLUMNS ===
illness_cols = [col for col in df.columns if re.search(r'diag|illness|disease|condition', col)]
print(f"Detected illness columns: {illness_cols}")

# === CLEAN & SAVE ===
df_clean = df.dropna(axis=1, how='all')  # remove empty columns
df_clean.to_csv(OUTPUT_FILE, index=False)

print(f"✅ Cleaned dataset saved as {OUTPUT_FILE}")
print(f"📋 Found {len(illness_cols)} possible illness columns.")
print("You can now inspect these to select target illnesses for training.")