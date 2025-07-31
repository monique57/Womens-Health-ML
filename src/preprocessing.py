# src/preprocessing.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

# ✅ Absolute path to dataset
DATA_PATH = r"C:/Users/Asus_/OneDrive/Desktop/WomensHealth/data/symptom_dataset_simulated.csv"

def load_and_preprocess(file_path=DATA_PATH):
    # ✅ Load dataset from absolute path
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at: {file_path}")

    df = pd.read_csv(file_path)

    # ✅ Separate features and labels
    X = df.drop("diagnosis", axis=1)
    y = df["diagnosis"]

    # ✅ Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ✅ Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test, scaler

if __name__ == "__main__":
    # Test the function
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess()
    print("✅ Preprocessing complete. Train shape:", X_train.shape)
