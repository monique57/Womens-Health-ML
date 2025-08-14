# src/app_v2.py
import streamlit as st
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import joblib
import os
from datetime import datetime

# ===== Paths =====
MODEL_PATH = r"C:/Users/Asus_/OneDrive/Desktop/WomensHealth/models/best_model_v2.joblib"
SCALER_PATH = r"C:/Users/Asus_/OneDrive/Desktop/WomensHealth/models/scaler_v2.save"
LOG_PATH = r"C:/Users/Asus_/OneDrive/Desktop/WomensHealth/data/user_submissions.csv"

# ===== Load model & scaler =====
@st.cache_resource
def load_model_and_scaler():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

model, scaler = load_model_and_scaler()

# ===== Feature names (MUST match training) =====
training_feature_names = [
    "age", "height_cm", "weight_kg", "family_autoimmune_history", "family_psych_history",
    "chronic_fatigue", "joint_pain", "muscle_pain", "skin_rash", "brain_fog",
    "depression_symptoms", "anxiety", "suicidal_thoughts", "menstrual_irregularities",
    "stress_level", "sleep_quality", "physical_activity_level", "smoking", "alcohol"
]

# ===== Labels aligned with class ids =====
CLASS_NAME = {0: "Depression", 1: "Lupus", 2: "Fibromyalgia"}

# ===== Streamlit UI =====
st.set_page_config(page_title="Women's Health (v2)", layout="centered")
st.title("🌸 Women's Health – Illness Detection with Explainability")
st.caption("This tool does not provide medical advice. It’s for educational support and must not replace professional diagnosis. Data you submit is stored anonymously for model improvement.")

with st.form("symptom_form"):
    age = st.number_input("Age", 18, 100, 30)
    height = st.number_input("Height (cm)", 140, 200, 165)
    weight = st.number_input("Weight (kg)", 40, 120, 65)
    fam_autoimmune = st.selectbox("Family history of autoimmune disease?", [0, 1])
    fam_psych = st.selectbox("Family history of mental illness?", [0, 1])
    chronic_fatigue = st.slider("Chronic Fatigue", 0, 3, 1)
    joint_pain = st.slider("Joint Pain", 0, 3, 1)
    muscle_pain = st.slider("Muscle Pain", 0, 3, 1)
    skin_rash = st.slider("Skin Rash", 0, 3, 0)
    brain_fog = st.slider("Brain Fog", 0, 3, 1)
    depression_symptoms = st.slider("Depression Symptoms", 0, 3, 1)
    anxiety = st.slider("Anxiety Symptoms", 0, 3, 1)
    suicidal_thoughts = st.selectbox("Suicidal Thoughts?", [0, 1])
    menstrual_irregularities = st.selectbox("Menstrual Irregularities?", [0, 1])
    stress = st.slider("Stress Level (0–10)", 0, 10, 5)
    sleep = st.slider("Sleep Quality (0–10)", 0, 10, 5)
    activity = st.selectbox("Physical Activity Level", [0, 1, 2])
    smoking = st.selectbox("Do you smoke?", [0, 1])
    alcohol = st.selectbox("Do you drink alcohol?", [0, 1])
    submitted = st.form_submit_button("🔍 Predict Illness")

if submitted:
    # ===== Prepare input row with training column names =====
    input_data = pd.DataFrame([[
        age, height, weight, fam_autoimmune, fam_psych, chronic_fatigue,
        joint_pain, muscle_pain, skin_rash, brain_fog, depression_symptoms,
        anxiety, suicidal_thoughts, menstrual_irregularities, stress, sleep,
        activity, smoking, alcohol
    ]], columns=training_feature_names)

    # ===== Transform =====
    features_scaled = scaler.transform(input_data)

    # ===== Predict (aligned with classes_) =====
    probs = model.predict_proba(features_scaled)[0]
    classes = model.classes_                      # e.g., array([0,1,2]); order matches probs
    labels = [CLASS_NAME[int(c)] for c in classes]

    pred_idx = int(np.argmax(probs))              # index in probs/classes
    pred_class = int(classes[pred_idx])
    pred_label = CLASS_NAME[pred_class]
    confidence = float(probs[pred_idx]) * 100

    # ===== Unsure logic =====
    top = float(np.max(probs))
    second = float(np.sort(probs)[-2]) if len(probs) > 1 else 0.0
    is_low_conf = top < 0.60
    is_too_close = (top - second) < 0.08

    # ===== Show result =====
    st.subheader("🩺 Prediction Result")
    st.success(f"**Predicted Condition:** {pred_label}")
    st.info(f"Confidence: {confidence:.2f}%")
    if is_low_conf or is_too_close:
        st.warning("⚠️ The model is not very confident. Consider consulting a clinician and repeating the questionnaire.")

    # ===== Probability chart =====
    st.subheader("📊 Illness Probability Distribution")
    fig_prob, ax_prob = plt.subplots()
    ax_prob.bar(labels, probs * 100, color=["#6FA8DC", "#E06666", "#93C47D"])
    ax_prob.set_ylabel("Probability (%)")
    ax_prob.set_title("Prediction Confidence for Each Illness")
    plt.tight_layout()
    st.pyplot(fig_prob)

    # ===== SHAP (robust to list/array) =====
    st.subheader(f"🔎 Symptoms Influencing: **{pred_label}**")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(features_scaled)

    if isinstance(shap_values, list):
        shap_single = np.array(shap_values[pred_idx][0]).flatten()
    else:
        shap_single = np.array(shap_values[0]).flatten()

    # length guard (rare SHAP shape quirks)
    if shap_single.shape[0] > len(training_feature_names):
        shap_single = shap_single[:len(training_feature_names)]
    elif shap_single.shape[0] < len(training_feature_names):
        shap_single = np.pad(shap_single, (0, len(training_feature_names) - shap_single.shape[0]))

    shap_df = pd.DataFrame({
        "Feature": training_feature_names,
        "SHAP Value": shap_single
    }).sort_values(by="SHAP Value", key=abs, ascending=False)

    fig_shap, ax_shap = plt.subplots()
    ax_shap.barh(shap_df["Feature"][:10][::-1], shap_df["SHAP Value"][:10][::-1])
    ax_shap.set_title(f"Top 10 Symptoms Influencing {pred_label} Prediction")
    plt.tight_layout()
    st.pyplot(fig_shap)

    # ===== Persist anonymous submission for future retraining =====
    record = input_data.copy()
    record["predicted_class_id"] = pred_class
    record["predicted_label"] = pred_label
    record["confidence"] = round(confidence, 2)
    record["timestamp"] = datetime.now().isoformat(timespec="seconds")
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    if not os.path.exists(LOG_PATH):
        record.to_csv(LOG_PATH, index=False)
    else:
        record.to_csv(LOG_PATH, mode="a", header=False, index=False)
    st.success("✅ Your (anonymous) responses were saved to help improve the model.")

    # ===== Download result as CSV =====
    csv_row = record.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download this result (CSV)", data=csv_row,
                       file_name="womens_health_result.csv", mime="text/csv")
