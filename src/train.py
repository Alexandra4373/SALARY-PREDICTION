"""
train.py - production training script for the salary prediction model.

Run from the PROJECT ROOT (the SALARY-PREDICTION folder):
    python src/train.py

What it does: load -> clean -> split -> tune Decision Tree & Random Forest
(cross-validation) -> evaluate on the test set -> log to MLflow ->
save the best model + metadata for the API/Gradio app.
"""
import json
from pathlib import Path

import joblib
import mlflow
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.tree import DecisionTreeRegressor

# ---------- Settings (change them here, not deep in the code) ----------
ROOT = Path(__file__).resolve().parent.parent      # project root folder
DATA_PATH = ROOT / "data" / "salary.csv"           # input data
MODEL_PATH = ROOT / "model" / "salary_model.pkl"   # saved model
META_PATH = ROOT / "model" / "metadata.json"       # info the app needs
RANDOM_STATE = 42                                  # makes results repeatable
TARGET = "Salary"                                  # what we predict
# Gender, Race and Country are left out on purpose (see notebook, section 5).
NUM_COLS = ["Age", "Years of Experience"]          # numeric inputs
EDU_COL = "Education Level"                        # ordered category
EDU_ORDER = ["High School", "Bachelor's Degree", "Master's Degree", "PhD"]
NOMINAL_COLS = ["Job Title"]                       # unordered category
FEATURES = NUM_COLS + [EDU_COL] + NOMINAL_COLS     # all model inputs


def load_and_clean(path):
    """Read the CSV and fix the data problems found during EDA."""
    df = pd.read_csv(path)
    df = df.drop(columns=["Unnamed: 0"], errors="ignore")  # useless index column
    df = df.dropna(subset=FEATURES + [TARGET])             # drop rows with missing values
    df[EDU_COL] = df[EDU_COL].replace({                    # make spellings consistent
        "Bachelor's": "Bachelor's Degree",
        "Master's": "Master's Degree",
        "phD": "PhD",
    })
    df["Job Title"] = (df["Job Title"].str.strip()         # fix title typos/inconsistencies
                       .str.replace("Juniour", "Junior")
                       .str.replace("Front end", "Front End"))
    df = df[df[TARGET] >= 1000]                            # 4 salaries of 350-579 are entry errors (next lowest is 25,000)
    df = df.drop_duplicates()                              # repeated rows would leak into the test set
    return df.reset_index(drop=True)


def build_pipeline(model):
    """Preprocessing + model in ONE object, so the API can take raw inputs."""
    preprocess = ColumnTransformer(
        [
            # Education: ordered numbers 0..3 (High School < Bachelor's < Master's < PhD)
            ("edu", OrdinalEncoder(categories=[EDU_ORDER],
                                   handle_unknown="use_encoded_value",
                                   unknown_value=-1), [EDU_COL]),
            # Job title: one 0/1 column per title; titles seen < 5 times share one column
            ("job", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                  min_frequency=5), NOMINAL_COLS),
        ],
        remainder="passthrough",  # Age and Years of Experience pass through unchanged
    )  # no scaling: tree models don't need it
    return Pipeline([("pre", preprocess), ("model", model)])


def main():
    # 1. Data
    df = load_and_clean(DATA_PATH)
    print(f"Clean data: {df.shape[0]} rows")
    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE)  # 80% train, 20% final test

    # 2. Models and the settings to try
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)  # 5 validation rounds
    candidates = {
        "DecisionTree": (
            DecisionTreeRegressor(random_state=RANDOM_STATE),
            {"model__max_depth": [5, 10, 15, None],
             "model__min_samples_leaf": [1, 3, 5, 10]},
        ),
        "RandomForest": (
            RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
            {"model__max_depth": [10, 20, None],
             "model__min_samples_leaf": [1, 2, 4],
             "model__max_features": [0.5, 1.0]},
        ),
    }

    # 3. Tune with cross-validation, then score once on the test set
    results = {}
    for name, (model, grid) in candidates.items():
        search = GridSearchCV(build_pipeline(model), grid, cv=cv,
                              scoring="neg_root_mean_squared_error", n_jobs=-1)
        search.fit(X_train, y_train)                 # tries every combination
        pred = search.predict(X_test)                # best combo, unseen test data
        results[name] = {
            "estimator": search.best_estimator_,
            "params": search.best_params_,
            "metrics": {
                "cv_rmse": float(-search.best_score_),
                "test_rmse": float(mean_squared_error(y_test, pred) ** 0.5),
                "test_mae": float(mean_absolute_error(y_test, pred)),
                "test_r2": float(r2_score(y_test, pred)),
            },
        }
        print(name, {k: round(v, 3) for k, v in results[name]["metrics"].items()})

    # 4. Choose the winner by CROSS-VALIDATION error (not by the test set)
    best_name = min(results, key=lambda n: results[n]["metrics"]["cv_rmse"])
    print("Best model:", best_name)

    # 5. Save the winner and the info the app needs
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(results[best_name]["estimator"], MODEL_PATH, compress=3)  # compress = smaller file
    title_counts = df["Job Title"].value_counts()
    metadata = {
        "model_name": best_name,
        "features": FEATURES,
        "education_levels": EDU_ORDER,
        "job_titles": sorted(title_counts[title_counts >= 5].index),  # for the dropdown
        "age_range": [int(df["Age"].min()), int(df["Age"].max())],
        "experience_range": [float(df["Years of Experience"].min()),
                             float(df["Years of Experience"].max())],
        "metrics": {k: round(v, 4) for k, v in results[best_name]["metrics"].items()},
    }
    META_PATH.write_text(json.dumps(metadata, indent=2))

    # 6. Log everything to MLflow (view with: mlflow ui)
    mlflow.set_experiment("salary-prediction")
    for name, res in results.items():
        with mlflow.start_run(run_name=name):
            mlflow.log_params({k.replace("model__", ""): str(v) for k, v in res["params"].items()})
            mlflow.log_metrics(res["metrics"])
            if name == best_name:
                mlflow.set_tag("selected", "true")
                mlflow.log_artifact(str(MODEL_PATH))  # attach the saved model file


if __name__ == "__main__":
    main()
