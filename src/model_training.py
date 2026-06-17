"""
Model Training Module.
Defines multiple regression models with Scikit-learn Pipelines and
provides a function to train the best model on the full training set.
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor,
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
MODELS_DIR = Path("models/trained_models")


def get_model_candidates() -> dict:
    """
    Returns a dictionary of named regression model pipelines.

    Each entry is a Scikit-learn Pipeline with optional preprocessing.
    Including a StandardScaler in the Pipeline protects distance-based
    models (SVR, KNN) even if the input was already scaled.

    Returns
    -------
    dict
        {model_name: Pipeline}
    """
    return {
        "Ridge": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0, random_state=RANDOM_STATE)),
        ]),
        "Lasso": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Lasso(alpha=0.01, max_iter=5000, random_state=RANDOM_STATE)),
        ]),
        "DecisionTree": Pipeline([
            ("model", DecisionTreeRegressor(max_depth=10, random_state=RANDOM_STATE)),
        ]),
        "RandomForest": Pipeline([
            ("model", RandomForestRegressor(
                n_estimators=100, max_depth=15,
                n_jobs=-1, random_state=RANDOM_STATE
            )),
        ]),
        "GradientBoosting": Pipeline([
            ("model", GradientBoostingRegressor(
                n_estimators=200, learning_rate=0.1,
                max_depth=5, random_state=RANDOM_STATE
            )),
        ]),
        "ExtraTrees": Pipeline([
            ("model", ExtraTreesRegressor(
                n_estimators=100, max_depth=15,
                n_jobs=-1, random_state=RANDOM_STATE
            )),
        ]),
        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsRegressor(n_neighbors=7, n_jobs=-1)),
        ]),
    }


def train_final_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_name: str,
    params: dict = None,
) -> Pipeline:
    """
    Trains the selected model on the FULL training set and serialises it.

    This function should be called AFTER hyperparameter tuning (tune.py)
    with the best params found by Optuna. Training on the full training set
    (not just the cross-validation folds) gives the model maximum data.

    Parameters
    ----------
    X_train : pd.DataFrame
        Full training feature matrix.
    y_train : pd.Series
        Full training target vector.
    model_name : str
        Key in get_model_candidates().
    params : dict, optional
        Hyperparameters to override defaults, e.g. from Optuna best_params.
        Keys must follow the Pipeline step format: 'model__n_estimators'.

    Returns
    -------
    Pipeline
        Fitted Scikit-learn Pipeline.
    """
    candidates = get_model_candidates()
    if model_name not in candidates:
        raise ValueError(f"Unknown model '{model_name}'. Valid: {list(candidates.keys())}")

    pipeline = candidates[model_name]

    if params:
        pipeline.set_params(**params)
        print(f">>  Applied custom params: {params}")

    print(f"\n>> Training {model_name} on full training set ({len(X_train):,} rows)...")
    pipeline.fit(X_train, y_train)
    print(f"OK: {model_name} trained successfully.")

    # Serialise
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / f"{model_name}_final.pkl"
    joblib.dump(pipeline, model_path)
    print(f"  Guardado: Model saved → {model_path}")

    return pipeline


def load_model(model_name: str) -> Pipeline:
    """
    Loads a serialised model from disk.

    Parameters
    ----------
    model_name : str
        Name used when saving (without '_final.pkl').

    Returns
    -------
    Pipeline
        Deserialised Scikit-learn Pipeline.
    """
    model_path = MODELS_DIR / f"{model_name}_final.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"No saved model at {model_path}")
    pipeline = joblib.load(model_path)
    print(f">> Model loaded from {model_path}")
    return pipeline


if __name__ == "__main__":
    X_train = pd.read_csv("data/processed/X_train.csv")
    y_train = pd.read_csv("data/processed/y_train.csv").squeeze()
    # Train all candidates for initial comparison
    candidates = get_model_candidates()
    for name, pipe in candidates.items():
        pipe.fit(X_train, y_train)
        print(f"OK: {name} trained.")
