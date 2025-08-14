import numpy as np
import pandas as pd
import joblib

MODEL_PATH = r"C:/Users/Asus_/OneDrive/Desktop/WomensHealth/models/best_model_v2.joblib"
SCALER_PATH = r"C:/Users/Asus_/OneDrive/Desktop/WomensHealth/models/scaler_v2.save"
DATA_PATH   = r"C:/Users/Asus_/OneDrive/Desktop/WomensHealth/data/symptom_dataset_patterned.csv"

FEATS = [
    "age","height_cm","weight_kg","family_autoimmune_history","family_psych_history",
    "chronic_fatigue","joint_pain","muscle_pain","skin_rash","brain_fog",
    "depression_symptoms","anxiety","suicidal_thoughts","menstrual_irregularities",
    "stress_level","sleep_quality","physical_activity_level","smoking","alcohol"
]
CLASS_NAME = {0:"Depression", 1:"Lupus", 2:"Fibromyalgia"}

def row(vals): return pd.DataFrame([vals], columns=FEATS)

def main():
    model  = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    df     = pd.read_csv(DATA_PATH)

    print("\nDataset class counts (0=Depression, 1=Lupus, 2=Fibromyalgia):")
    print(df["diagnosis"].value_counts().sort_index(), "\n")

    classes = model.classes_
    labels  = [CLASS_NAME[int(c)] for c in classes]
    print("Model classes_ order:", classes)
    print("Label names in that order:", labels, "\n")

    dep = row([28,165,62, 0,1, 1,0,0,0,1, 3,3,1,0, 9,2, 0,0,0])
    lup = row([35,165,66, 1,0, 2,3,1,2,1, 0,1,0,1, 4,6, 1,0,0])
    fib = row([38,165,68, 0,0, 3,1,3,0,3, 1,1,0,0, 6,4, 1,0,0])

    for name, X in [("Depression-like", dep), ("Lupus-like", lup), ("Fibro-like", fib)]:
        Xs = scaler.transform(X)
        probs = model.predict_proba(Xs)[0]
        pred_idx = int(np.argmax(probs))
        pred_class = int(classes[pred_idx])
        pred_name = CLASS_NAME[pred_class]
        print(f"{name} input:")
        for cls, p in zip(labels, probs):
            print(f"  {cls:14s}: {p*100:6.2f}%")
        print(f"  -> Predicted: {pred_name}\n")

if __name__ == "__main__":
    main()
