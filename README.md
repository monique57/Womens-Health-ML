"# Womens-Health-ML" 
# 🌸 Women's Health – ML-Driven Illness Prediction App

## 🧠 Project Overview
This project, "Machine Learning–Driven Differentiation of Fibromyalgia, Lupus, and Depression Using Symptom Overlap in Women", is an intelligent web application designed to:
- ✅ Detect Lupus, Fibromyalgia, or Depression based on user-provided symptoms  
- ✅ Provide probability scores for each illness  
- ✅ Explain the prediction using SHAP (feature importance)

💡 The goal is to help women and healthcare professionals identify potential conditions early and reduce misdiagnosis caused by overlapping symptoms.

---

## 🚀 Features
- 🌐 Streamlit Web App – interactive and easy to use  
- 🧠 Machine Learning – Random Forest classifier trained with synthetic data  
- 📊 Model Explainability – SHAP visualization of symptom influence  
- ✅ Supports Deployment – ready to run locally or on Streamlit Cloud  

---

## 📂 Project Structure


# Women’s Health (NAMCS 2019 + Synthetic)
## Quickstart
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# (If data not present) place namcs2019_clean.csv into data/
# Train/test (NAMCS)
python src/train_namcs2019.py
python src/test_namcs2019.py

# Stability & CV
python src/stability_eval.py
python src/cross_validate.py

# App (NAMCS)
streamlit run src/app_namcs.py

## Results bundle
See results/final_report/ (plots + CSVs ready for thesis)