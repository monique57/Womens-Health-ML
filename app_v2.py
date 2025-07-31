# src/app_v2.py
import streamlit as st
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import joblib

# ✅ Paths
MODEL_PATH = r"C:/Users/Asus_/OneDrive/Desktop/WomensHealth/models/best_model_v2.joblib"
SCALER_PATH = r"C:/Users/Asus_/OneDrive/Desktop/WomensHealth/models/scaler_v2.save"

# ✅ Load Model & Scaler
@st.cache_resource
def load_model_and_scaler():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

model, scaler = load_model_and_scaler()

# ✅ Training Feature Names
training_feature_names = [
    "age", "height_cm", "weight_kg", "family_autoimmune_history", "family_psych_history",
    "chronic_fatigue", "joint_pain", "muscle_pain", "skin_rash", "brain_fog",
    "depression_symptoms", "anxiety", "suicidal_thoughts", "menstrual_irregularities",
    "stress_level", "sleep_quality", "physical_activity_level", "smoking", "alcohol"
]

# ✅ Illness Labels
illness_map = {0: "Depression", 1: "Lupus", 2: "Fibromyalgia"}

# ✅ UI Setup
st.set_page_config(page_title="Women's Health (v2)", layout="centered")
st.title("🌸 Women's Health – Illness Detection with SHAP Explainability")
st.write("This app uses an improved Random Forest model trained on patterned data.")

# ✅ Form for User Input
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

# ✅ Prediction
if submitted:
    input_data = pd.DataFrame([[
        age, height, weight, fam_autoimmune, fam_psych, chronic_fatigue,
        joint_pain, muscle_pain, skin_rash, brain_fog, depression_symptoms,
        anxiety, suicidal_thoughts, menstrual_irregularities, stress, sleep,
        activity, smoking, alcohol
    ]], columns=training_feature_names)

    features_scaled = scaler.transform(input_data)

    # ✅ Prediction
    probs = model.predict_proba(features_scaled)[0]
    prediction = np.argmax(probs)
    confidence = probs[prediction] * 100

    st.subheader("🩺 Prediction Result")
    st.success(f"**Predicted Condition:** {illness_map[prediction]}")
    st.info(f"Confidence: {confidence:.2f}%")

    # ✅ Probability Chart
    st.subheader("📊 Illness Probability Distribution")
    fig_prob, ax_prob = plt.subplots()
    ax_prob.bar(list(illness_map.values()), probs * 100,
                color=["#6FA8DC", "#E06666", "#93C47D"])
    ax_prob.set_ylabel("Probability (%)")
    ax_prob.set_title("Prediction Confidence for Each Illness")
    plt.tight_layout()
    st.pyplot(fig_prob)

    # ✅ SHAP Explainability (Safe for All Cases)
    st.subheader(f"🔎 Symptoms Influencing: **{illness_map[prediction]}**")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(features_scaled)

    # ✅ Handle both cases (list or single array)
    if isinstance(shap_values, list):
        shap_single = np.array(shap_values[prediction][0]).flatten()
    else:
        shap_single = np.array(shap_values[0]).flatten()

    # ✅ Ensure matching length
    if shap_single.shape[0] > len(training_feature_names):
        shap_single = shap_single[:len(training_feature_names)]
    elif shap_single.shape[0] < len(training_feature_names):
        shap_single = np.pad(shap_single, (0, len(training_feature_names) - shap_single.shape[0]))

    # ✅ Build SHAP DataFrame
    shap_df = pd.DataFrame({
        "Feature": training_feature_names,
        "SHAP Value": shap_single
    }).sort_values(by="SHAP Value", key=abs, ascending=False)

    # ✅ Plot for the predicted illness only
    fig_shap, ax_shap = plt.subplots()
    ax_shap.barh(shap_df["Feature"][:10][::-1], shap_df["SHAP Value"][:10][::-1])
    ax_shap.set_title(f"Top 10 Symptoms Influencing {illness_map[prediction]} Prediction")
    plt.tight_layout()
    st.pyplot(fig_shap)

    st.warning("⚠️ This prediction is for educational purposes only. Consult a healthcare professional.")
