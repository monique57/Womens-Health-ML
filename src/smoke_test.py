import joblib, numpy as np, pandas as pd
MODEL = r"C:/Users/Asus_/OneDrive/Desktop/WomensHealth/models/best_model_v2.joblib"
SCALER = r"C:/Users/Asus_/OneDrive/Desktop/WomensHealth/models/scaler_v2.save"

cols = ["age","height_cm","weight_kg","family_autoimmune_history","family_psych_history",
        "chronic_fatigue","joint_pain","muscle_pain","skin_rash","brain_fog",
        "depression_symptoms","anxiety","suicidal_thoughts","menstrual_irregularities",
        "stress_level","sleep_quality","physical_activity_level","smoking","alcohol"]

model = joblib.load(MODEL); scaler = joblib.load(SCALER)
x = pd.DataFrame([[30,165,65,0,0,1,1,1,0,1,2,2,0,0,5,5,1,0,0]], columns=cols)
xp = scaler.transform(x)
probs = model.predict_proba(xp)[0]
print("Classes (order):", model.classes_)
print("Probs:", probs)
print("Pred idx:", np.argmax(probs))