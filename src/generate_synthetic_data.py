import pandas as pd
import numpy as np

# Number of synthetic samples
n = 500  

data = []

for i in range(n):
    # Randomly assign a diagnosis (0=Depression, 1=Lupus, 2=Fibromyalgia)
    diagnosis = np.random.choice([0, 1, 2], p=[0.33, 0.33, 0.34])

    # Default symptom values
    age = np.random.randint(18, 60)
    height = np.random.randint(150, 180)
    weight = np.random.randint(45, 90)
    family_autoimmune = 0
    family_psych = 0
    chronic_fatigue = np.random.randint(0, 2)
    joint_pain = np.random.randint(0, 2)
    muscle_pain = np.random.randint(0, 2)
    skin_rash = 0
    brain_fog = np.random.randint(0, 2)
    depression_symptoms = np.random.randint(0, 2)
    anxiety = np.random.randint(0, 2)
    suicidal_thoughts = 0
    menstrual_irregularities = np.random.randint(0, 2)
    stress = np.random.randint(2, 6)
    sleep = np.random.randint(4, 7)
    activity = np.random.randint(0, 3)
    smoking = np.random.randint(0, 2)
    alcohol = np.random.randint(0, 2)

    # ✅ Apply illness-specific patterns
    if diagnosis == 0:  # Depression
        depression_symptoms = np.random.randint(2, 4)
        anxiety = np.random.randint(1, 4)
        suicidal_thoughts = np.random.randint(0, 2)
        stress = np.random.randint(6, 10)
        sleep = np.random.randint(0, 5)

    elif diagnosis == 1:  # Lupus
        family_autoimmune = 1
        joint_pain = np.random.randint(1, 4)
        skin_rash = np.random.randint(1, 3)
        chronic_fatigue = np.random.randint(1, 3)
        menstrual_irregularities = np.random.randint(0, 2)

    elif diagnosis == 2:  # Fibromyalgia
        chronic_fatigue = np.random.randint(2, 4)
        muscle_pain = np.random.randint(2, 4)
        brain_fog = np.random.randint(2, 4)
        stress = np.random.randint(4, 8)

    data.append([age, height, weight, family_autoimmune, family_psych, chronic_fatigue,
                 joint_pain, muscle_pain, skin_rash, brain_fog, depression_symptoms,
                 anxiety, suicidal_thoughts, menstrual_irregularities, stress, sleep,
                 activity, smoking, alcohol, diagnosis])

# Create DataFrame
columns = ["age", "height_cm", "weight_kg", "family_autoimmune_history", "family_psych_history",
           "chronic_fatigue", "joint_pain", "muscle_pain", "skin_rash", "brain_fog",
           "depression_symptoms", "anxiety", "suicidal_thoughts", "menstrual_irregularities",
           "stress_level", "sleep_quality", "physical_activity_level", "smoking", "alcohol", "diagnosis"]

df = pd.DataFrame(data, columns=columns)

# ✅ Save to data folder
output_path = r"C:/Users/Asus_/OneDrive/Desktop/WomensHealth/data/symptom_dataset_patterned.csv"
df.to_csv(output_path, index=False)

print(f"✅ Patterned synthetic dataset saved to: {output_path}")
