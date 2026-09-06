# 🩺 Diabetes Prediction Using Machine Learning

An interview-friendly diabetes classification project using three complementary models:

- Logistic Regression — interpretable baseline
- Decision Tree — rule-based and explainable
- XGBoost — powerful nonlinear boosting model

## Project structure

```text
diabetes-prediction/
│
├── data/
│   └── diabetes.csv
│
├── notebooks/
│   └── diabetes_prediction.ipynb
│
├── models/
├── reports/
├── app.py
├── train.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Dataset

The project uses the 10,000-record `diabetes.csv` dataset with these columns:

`PatientID, Pregnancies, PlasmaGlucose, DiastolicBloodPressure, TricepsThickness, SerumInsulin, BMI, DiabetesPedigree, Age, Diabetic`

The dataset is stored directly inside this repository under `data/diabetes.csv`.
The application and training code do not download data from another repository at runtime.

## Setup

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Put the dataset at:

```text
data/diabetes.csv
```

Then run:

```powershell
python train.py
```

This trains all three models, tunes their hyperparameters using 5-fold stratified cross-validation, evaluates them on the held-out test set, and saves the trained pipelines.

To launch the application:

```powershell
streamlit run app.py
```

## Machine learning workflow

```text
Dataset
   ↓
Data inspection
   ↓
Remove PatientID
   ↓
Handle invalid zero measurements
   ↓
Stratified train/test split
   ↓
Preprocessing Pipeline
   ↓
Logistic Regression
Decision Tree
XGBoost
   ↓
5-fold cross-validation + GridSearchCV
   ↓
Accuracy / Precision / Recall / F1 / ROC-AUC
   ↓
Save trained models
   ↓
Streamlit application
```

## Why these three models?

### Logistic Regression

Used as the baseline because it is simple, fast and interpretable for binary classification.

### Decision Tree

Used because it learns easy-to-explain if/then decision rules and can capture nonlinear relationships without feature scaling.

### XGBoost

Used as the stronger nonlinear model. It builds an ensemble of decision trees sequentially and can capture complex relationships and feature interactions.

## Important preprocessing decisions

`PatientID` is removed because it is an identifier rather than a meaningful medical feature.

For several measurements, zero is treated as an invalid/missing value and converted to `NaN`. Median imputation is then performed inside the training pipeline.

Logistic Regression uses `StandardScaler`. Decision Tree and XGBoost do not require scaling because they are tree-based models.

## Evaluation

The project reports:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

For a medical classification problem, accuracy should not be considered alone. Recall and false negatives are particularly important.



## Medical disclaimer

This is an educational machine-learning project and should not be used as a medical diagnostic system.
