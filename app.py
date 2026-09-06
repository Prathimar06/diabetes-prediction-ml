from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

MODEL_DIR = Path("models")

FEATURES = [
    "Pregnancies", "PlasmaGlucose", "DiastolicBloodPressure",
    "TricepsThickness", "SerumInsulin", "BMI", "DiabetesPedigree", "Age"
]

MODEL_FILES = {
    "Logistic Regression": MODEL_DIR / "logistic_regression.joblib",
    "Decision Tree": MODEL_DIR / "decision_tree.joblib",
    "XGBoost": MODEL_DIR / "xgboost.joblib"
}

st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 Diabetes Prediction")
st.write(
    "Educational comparison of Logistic Regression, Decision Tree and XGBoost."
)

available_models = [
    name for name, path in MODEL_FILES.items()
    if path.exists()
]

if not available_models:
    st.error("No trained models found. Run `python train.py` first.")
    st.stop()

model_name = st.selectbox("Choose a model", available_models)
model = joblib.load(MODEL_FILES[model_name])

st.subheader("Patient Information")

left, right = st.columns(2)

with left:
    pregnancies = st.number_input("Pregnancies", 0, 20, 1)
    glucose = st.number_input("Plasma Glucose", 0.0, 300.0, 120.0)
    blood_pressure = st.number_input(
        "Diastolic Blood Pressure", 0.0, 150.0, 70.0
    )
    triceps = st.number_input("Triceps Thickness", 0.0, 100.0, 20.0)

with right:
    insulin = st.number_input("Serum Insulin", 0.0, 1000.0, 80.0)
    bmi = st.number_input("BMI", 0.0, 80.0, 25.0)
    pedigree = st.number_input(
        "Diabetes Pedigree", 0.0, 3.0, 0.5
    )
    age = st.number_input("Age", 1, 120, 30)

if st.button("Predict", type="primary"):
    input_data = pd.DataFrame([[
        pregnancies,
        glucose,
        blood_pressure,
        triceps,
        insulin,
        bmi,
        pedigree,
        age
    ]], columns=FEATURES)

    prediction = int(model.predict(input_data)[0])
    probability = float(model.predict_proba(input_data)[0, 1])

    if prediction == 1:
        st.error(
            f"Higher predicted diabetes risk ({probability:.1%})"
        )
    else:
        st.success(
            f"Lower predicted diabetes risk ({probability:.1%})"
        )

    st.caption(
        "For educational purposes only — this is not a medical diagnosis."
    )
