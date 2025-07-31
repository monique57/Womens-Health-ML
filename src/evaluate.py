# src/evaluate.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from preprocessing import load_and_preprocess
from tensorflow.keras.models import load_model

# ✅ Absolute paths to models
MODEL_PATH_JOBLIB = r"C:/Users/Asus_/OneDrive/Desktop/WomensHealth/models/best_model.joblib"
MODEL_PATH_NN = r"C:/Users/Asus_/OneDrive/Desktop/WomensHealth/models/best_nn_model.h5"

# ✅ Load preprocessed test data
print("🔹 Loading dataset...")
X_train, X_test, y_train, y_test, scaler = load_and_preprocess()

# ✅ Load the saved model
try:
    model = joblib.load(MODEL_PATH_JOBLIB)
    is_nn = False
    print("✅ Loaded RandomForest/XGBoost/Logistic Regression model.")
except:
    model = load_model(MODEL_PATH_NN)
    is_nn = True
    print("✅ Loaded Neural Network model.")

# ✅ Run Evaluation 30 Times
acc_list, f1_list = [], []
print("🚀 Starting evaluation (30 runs)...")

for i in range(30):
    if is_nn:
        y_pred = np.argmax(model.predict(X_test), axis=1)
    else:
        y_pred = model.predict(X_test)

    acc_list.append(accuracy_score(y_test, y_pred))
    f1_list.append(f1_score(y_test, y_pred, average="weighted"))

# ✅ Print Results
avg_acc = np.mean(acc_list)
std_acc = np.std(acc_list)
avg_f1 = np.mean(f1_list)
std_f1 = np.std(f1_list)

print("\n📊 Evaluation Results (30 Runs):")
print(f"Average Accuracy: {avg_acc:.3f} ± {std_acc:.3f}")
print(f"Average F1 Score: {avg_f1:.3f} ± {std_f1:.3f}")

# ✅ Final Classification Report (last run)
print("\n📊 Classification Report (Last Run):")
print(classification_report(y_test, y_pred, target_names=["Depression", "Lupus", "Fibromyalgia"]))

# ✅ Confusion Matrix (last run)
print("📌 Saving confusion matrix to models/confusion_matrix.png")
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Depression","Lupus","Fibromyalgia"],
            yticklabels=["Depression","Lupus","Fibromyalgia"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix (Last Run)")
plt.savefig(r"C:/Users/Asus_/OneDrive/Desktop/WomensHealth/models/confusion_matrix.png")
plt.show()
