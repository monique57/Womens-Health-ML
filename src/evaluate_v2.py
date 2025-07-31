import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

# ✅ Paths
MODEL_PATH = r"C:/Users/Asus_/OneDrive/Desktop/WomensHealth/models/best_model_v2.joblib"
SCALER_PATH = r"C:/Users/Asus_/OneDrive/Desktop/WomensHealth/models/scaler_v2.save"
DATA_PATH = r"C:/Users/Asus_/OneDrive/Desktop/WomensHealth/data/symptom_dataset_patterned.csv"

# ✅ Load data
df = pd.read_csv(DATA_PATH)
X = df.drop("diagnosis", axis=1)
y = df["diagnosis"]

# ✅ Load scaler
scaler = joblib.load(SCALER_PATH)
X_scaled = scaler.transform(X)

# ✅ Train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# ✅ Load model
model = joblib.load(MODEL_PATH)
print("✅ Loaded Random Forest Model (v2)")

# ✅ Test 30 times
acc_list, f1_list = [], []
for i in range(30):
    y_pred = model.predict(X_test)
    acc_list.append(accuracy_score(y_test, y_pred))
    f1_list.append(f1_score(y_test, y_pred, average="weighted"))

# ✅ Print Average Results
print("\n📊 Evaluation Results (30 Runs):")
print(f"Average Accuracy: {np.mean(acc_list):.3f} ± {np.std(acc_list):.3f}")
print(f"Average F1 Score: {np.mean(f1_list):.3f} ± {np.std(f1_list):.3f}")

# ✅ Classification Report (last run)
print("\n📊 Classification Report (Last Run):")
print(classification_report(y_test, y_pred, target_names=["Depression", "Lupus", "Fibromyalgia"]))

# ✅ Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Depression","Lupus","Fibromyalgia"],
            yticklabels=["Depression","Lupus","Fibromyalgia"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix – Random Forest v2")
plt.savefig(r"C:/Users/Asus_/OneDrive/Desktop/WomensHealth/models/confusion_matrix_v2.png")
plt.show()
