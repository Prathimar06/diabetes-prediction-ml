from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    classification_report
)
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

DATA_PATH = Path("data/diabetes.csv")
MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")
RANDOM_STATE = 42

FEATURES = [
    "Pregnancies", "PlasmaGlucose", "DiastolicBloodPressure",
    "TricepsThickness", "SerumInsulin", "BMI", "DiabetesPedigree", "Age"
]

INVALID_ZERO_FEATURES = [
    "PlasmaGlucose", "DiastolicBloodPressure",
    "TricepsThickness", "SerumInsulin", "BMI"
]


def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "data/diabetes.csv was not found. Put the dataset inside the data folder."
        )

    df = pd.read_csv(DATA_PATH)

    required = ["PatientID"] + FEATURES + ["Diabetic"]
    missing = [column for column in required if column not in df.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # PatientID is an identifier, not a predictive feature.
    df = df.drop(columns=["PatientID"])

    # Treat impossible zero measurements as missing.
    for column in INVALID_ZERO_FEATURES:
        df[column] = df[column].replace(0, np.nan)

    return df


def build_models():
    logistic_regression = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=2000,
            random_state=RANDOM_STATE
        ))
    ])

    decision_tree = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", DecisionTreeClassifier(
            random_state=RANDOM_STATE
        ))
    ])

    xgboost = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1
        ))
    ])

    return {
        "Logistic Regression": (
            logistic_regression,
            {"model__C": [0.01, 0.1, 1, 10]}
        ),
        "Decision Tree": (
            decision_tree,
            {
                "model__max_depth": [3, 5, 7, 10, None],
                "model__min_samples_split": [2, 5, 10],
                "model__min_samples_leaf": [1, 2, 5]
            }
        ),
        "XGBoost": (
            xgboost,
            {
                "model__n_estimators": [100, 200],
                "model__max_depth": [3, 5],
                "model__learning_rate": [0.05, 0.1],
                "model__subsample": [0.8, 1.0]
            }
        )
    }


def evaluate_model(name, model, X_test, y_test):
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions, zero_division=0),
        "Recall": recall_score(y_test, predictions, zero_division=0),
        "F1": f1_score(y_test, predictions, zero_division=0),
        "ROC_AUC": roc_auc_score(y_test, probabilities)
    }

    print(f"\n{'=' * 60}")
    print(name)
    print(f"{'=' * 60}")
    print(pd.Series(metrics))
    print("\nClassification Report:")
    print(classification_report(y_test, predictions, zero_division=0))

    return metrics


def main():
    MODEL_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)

    df = load_data()

    X = df[FEATURES]
    y = df["Diabetic"]

    print("Dataset shape:", df.shape)
    print("\nTarget distribution:")
    print(y.value_counts())

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=RANDOM_STATE
    )

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    results = []

    for name, (pipeline, parameter_grid) in build_models().items():
        print(f"\nTuning {name}...")

        search = GridSearchCV(
            estimator=pipeline,
            param_grid=parameter_grid,
            scoring="roc_auc",
            cv=cv,
            n_jobs=-1,
            refit=True
        )

        search.fit(X_train, y_train)

        print("Best parameters:", search.best_params_)
        print("Best CV ROC-AUC:", round(search.best_score_, 4))

        model_path = MODEL_DIR / f"{name.lower().replace(' ', '_')}.joblib"
        joblib.dump(search.best_estimator_, model_path)

        results.append(
            evaluate_model(
                name,
                search.best_estimator_,
                X_test,
                y_test
            )
        )

    results_df = pd.DataFrame(results).sort_values(
        "ROC_AUC",
        ascending=False
    )

    results_df.to_csv(
        REPORT_DIR / "model_comparison.csv",
        index=False
    )

    print("\nFINAL MODEL COMPARISON")
    print(results_df.to_string(index=False))

    print(
        "\nBest model by test ROC-AUC:",
        results_df.iloc[0]["Model"]
    )


if __name__ == "__main__":
    main()
