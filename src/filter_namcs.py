import pandas as pd
from pathlib import Path

# ✅ Locate file relative to project root
BASE_DIR = Path(__file__).resolve().parent.parent
input_file = BASE_DIR / "data" / "namcs2019.csv"
output_file = BASE_DIR / "data" / "namcs2019_filtered.csv"

print(f"📂 Loading dataset from: {input_file}")

# Load dataset (keep everything as string to avoid code formatting issues)
df = pd.read_csv(input_file, dtype=str)

print(f"✅ Dataset loaded. Shape: {df.shape}")

# --------------------------------------------
# Define ICD-10 codes for each illness
# (You can expand this list if needed)
# --------------------------------------------
depression_codes = ["F32", "F33", "F34.1"]  # Major depressive disorder, dysthymia
lupus_codes = ["M32", "M32.0", "M32.1", "M32.8", "M32.9"]  # Systemic lupus erythematosus
fibromyalgia_codes = ["M79.7"]

# Column in NAMCS that stores diagnosis codes
# Sometimes they are in DIAG1, DIAG2, DIAG3
diagnosis_columns = [col for col in df.columns if "DIAG" in col.upper()]

# --------------------------------------------
# Filter rows where ANY diagnosis matches
# --------------------------------------------
df_filtered = df[
    df[diagnosis_columns].apply(
        lambda row: any(
            code.startswith(tuple(depression_codes + lupus_codes + fibromyalgia_codes))
            for code in row if pd.notna(code)
        ),
        axis=1
    )
]

print(f"✅ Filtered dataset shape: {df_filtered.shape}")

# --------------------------------------------
# Save filtered dataset
# --------------------------------------------
df_filtered.to_csv(output_file, index=False)
print(f"💾 Filtered dataset saved to: {output_file}")