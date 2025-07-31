# src/app.py
import streamlit as st
import numpy as np
import shap
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import joblib

# ✅ Paths
MODEL_PATH = r"C:/Users/Asus_/OneDrive/Desktop/WomensHealth/models/best_nn_model.h5"
SCALER_PATH = r"C:/Users/Asus_/OneDrive/Desktop/WomensHealth/models/scaler.save"

# ✅ Load Neural Network model
@st.cache_resource
def load_nn_model():
    return load_model(MODEL_PATH)

model = load_nn_model()

# ✅ Load scaler (fallback if missing)
try:
    scaler = joblib.load(SCALER_PATH)
except:
    scaler = StandardScaler()

# ✅ Feature names for SHAP
feature_names = [
    "Age", "Height", "Weight", "Family Autoimmune", "Family Psych",
    "Chronic Fatigue", "Joint Pain", "Muscle Pain", "Skin Rash",
    "Brain Fog", "Depression Symptoms", "Anxiety", "Suicidal Thoughts",
    "Menstrual Irregularities", "Stress", "Sleep", "Activity", "Smoking", "Alcohol"
]

# ✅ Illness labels
illness_map = {0: "Depression", 1: "Lupus", 2: "Fibromyalgia"}

# ✅ Streamlit UI
st.set_page_config(page_title="Women's Health", layout="centered")
st.title("🌸 Women's Health – Illness Detection with Explainability")
st.write("Fill in your symptoms to check the likelihood of **Lupus**, **Fibromyalgia**, or **Depression**.")

# ✅ Input Form
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

# ✅ Prediction + SHAP Explainability
if submitted:
    # Prepare features
    features = np.array([[age, height, weight, fam_autoimmune, fam_psych, chronic_fatigue,
                          joint_pain, muscle_pain, skin_rash, brain_fog, depression_symptoms,
                          anxiety, suicidal_thoughts, menstrual_irregularities, stress, sleep,
                          activity, smoking, alcohol]])

    # Scale
    try:
        features_scaled = scaler.transform(features)
    except:
        features_scaled = features

    # Predict
    probs = model.predict(features_scaled)
    prediction = np.argmax(probs)
    confidence = np.max(probs) * 100

    # Display Results
    st.subheader("🩺 Prediction Result")
    st.success(f"**Predicted Condition:** {illness_map[prediction]}")
    st.info(f"Confidence: {confidence:.2f}%")

    # SHAP Explainability
    st.subheader("🔎 Symptom Influence on Prediction")

    # Use KernelExplainer for Neural Networks
    explainer = shap.Explainer(model, features_scaled)
    shap_values = explainer(features_scaled)

    # Select SHAP values for the predicted class
    shap_values_single = shap_values[0][:, prediction]

    # Plot Bar Chart
    fig, ax = plt.subplots()
    shap.plots.bar(shap_values_single, show=False)
    st.pyplot(fig)

    st.warning("⚠️ This prediction is for educational purposes only. Consult a healthcare professional.")
