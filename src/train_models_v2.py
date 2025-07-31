# src/train_models_v2.py
import os
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

# ✅ Paths
DATA_PATH = r"C:/Users/Asus_/OneDrive/Desktop/WomensHealth/data/symptom_dataset_patterned.csv"
MODEL_DIR = r"C:/Users/Asus_/OneDrive/Desktop/WomensHealth/models"
MODEL_PATH = os.path.join(MODEL_DIR, "best_nn_model_v2.h5")
os.makedirs(MODEL_DIR, exist_ok=True)

# ✅ Load dataset
df = pd.read_csv(DATA_PATH)
X = df.drop("diagnosis", axis=1)
y = df["diagnosis"]

# ✅ Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ✅ Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# ✅ Save scaler
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler_v2.save"))

# ✅ One-hot for NN
y_train_oh = to_categorical(y_train)
y_test_oh = to_categorical(y_test)

# ✅ Compute class weights
classes = np.unique(y)
class_weights_dict = dict(zip(classes, compute_class_weight('balanced', classes=classes, y=y_train)))
print("✅ Class Weights:", class_weights_dict)

# ---- Model Functions ---- #
def train_random_forest(seed):
    model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=seed)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return model, accuracy_score(y_test, preds), f1_score(y_test, preds, average="weighted")

def train_xgboost(seed):
    model = XGBClassifier(eval_metric='mlogloss', random_state=seed)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return model, accuracy_score(y_test, preds), f1_score(y_test, preds, average="weighted")

def train_logistic_regression(seed):
    model = LogisticRegression(max_iter=1000, random_state=seed)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return model, accuracy_score(y_test, preds), f1_score(y_test, preds, average="weighted")

def train_neural_network(seed):
    np.random.seed(seed)
    model = Sequential([
        Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(3, activation='softmax')
    ])
    model.compile(optimizer=Adam(0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model.fit(X_train, y_train_oh, validation_split=0.2, epochs=50, batch_size=16,
              class_weight=class_weights_dict, verbose=0, callbacks=[es])
    _, acc = model.evaluate(X_test, y_test_oh, verbose=0)
    preds = np.argmax(model.predict(X_test), axis=1)
    f1 = f1_score(y_test, preds, average="weighted")
    return model, acc, f1

# ---- Training Loop ---- #
results = {"Model": [], "Accuracy": [], "F1": []}
best_f1 = 0
best_model = None
best_model_name = ""

for name, trainer in [("RandomForest", train_random_forest),
                      ("XGBoost", train_xgboost),
                      ("LogisticRegression", train_logistic_regression),
                      ("NeuralNetwork", train_neural_network)]:
    acc_scores, f1_scores = [], []
    print(f"\n🚀 Training {name} (30 iterations)...")
    for seed in range(1, 31):
        model, acc, f1 = trainer(seed)
        acc_scores.append(acc)
        f1_scores.append(f1)
        if f1 > best_f1:
            best_f1 = f1
            best_model = model
            best_model_name = name

    results["Model"].append(name)
    results["Accuracy"].append(np.mean(acc_scores))
    results["F1"].append(np.mean(f1_scores))

# ---- Save Best Model ---- #
if best_model_name == "NeuralNetwork":
    best_model.save(MODEL_PATH)
else:
    joblib.dump(best_model, os.path.join(MODEL_DIR, "best_model_v2.joblib"))

print(f"\n✅ Best Model: {best_model_name} (F1={best_f1:.3f}) saved successfully.")

# ---- Plot Model Comparison ---- #
df_results = pd.DataFrame(results)
plt.bar(df_results["Model"], df_results["F1"], alpha=0.7)
plt.ylabel("Average F1 Score")
plt.title("Model Comparison (30 Runs) – Patterned Data")
plt.savefig(os.path.join(MODEL_DIR, "model_comparison_v2.png"))
plt.show()

# ---- SHAP for Tree Models ---- #
if best_model_name in ["RandomForest", "XGBoost"]:
    print("\n📊 Generating SHAP Feature Importance...")
    explainer = shap.TreeExplainer(best_model)
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values, X_test, show=False)
    plt.savefig(os.path.join(MODEL_DIR, "shap_summary_v2.png"))
    print("✅ SHAP summary plot saved.")
