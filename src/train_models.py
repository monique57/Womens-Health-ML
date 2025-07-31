# src/train_models.py
import os
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from preprocessing import load_and_preprocess

# ✅ Path for saving best model
MODEL_PATH = r"C:/Users/Asus_/OneDrive/Desktop/WomensHealth/models/best_model.joblib"
MODEL_DIR = os.path.dirname(MODEL_PATH)
os.makedirs(MODEL_DIR, exist_ok=True)

# ✅ Load preprocessed data
X_train, X_test, y_train, y_test, scaler = load_and_preprocess()

# ✅ One-hot encoding for neural network
y_train_oh = to_categorical(y_train)
y_test_oh = to_categorical(y_test)

# ---- Model Training Functions ---- #
def train_random_forest(seed):
    model = RandomForestClassifier(n_estimators=150, random_state=seed)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return model, accuracy_score(y_test, preds), f1_score(y_test, preds, average="weighted")

def train_xgboost(seed):
    model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=seed)
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
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(32, activation='relu'),
        Dense(3, activation='softmax')
    ])
    model.compile(optimizer=Adam(0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train_oh, epochs=30, batch_size=16, verbose=0)
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
    best_model.save(os.path.join(MODEL_DIR, "best_nn_model.h5"))
else:
    joblib.dump(best_model, MODEL_PATH)

print(f"\n✅ Best Model: {best_model_name} (F1={best_f1:.3f}) saved successfully.")

# ---- Plot Model Comparison ---- #
df_results = pd.DataFrame(results)
plt.bar(df_results["Model"], df_results["F1"], alpha=0.7)
plt.ylabel("Average F1 Score")
plt.title("Model Comparison (30 Runs)")
plt.savefig(os.path.join(MODEL_DIR, "model_comparison.png"))
plt.show()

# ---- SHAP Feature Importance (Tree Models Only) ---- #
if best_model_name in ["RandomForest", "XGBoost"]:
    print("\n📊 Generating SHAP Feature Importance...")
    explainer = shap.TreeExplainer(best_model)
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values, X_test, show=False)
    plt.savefig(os.path.join(MODEL_DIR, "shap_summary.png"))
    print("✅ SHAP summary plot saved.")
