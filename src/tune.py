"""
Hyperparameter Tuning Module.
Combines Optuna (Bayesian optimisation) with scikit-learn's
GridSearchCV and RandomizedSearchCV for thorough hyperparameter search.
 
Workflow
--------
1. Quick GridSearchCV / RandomizedSearchCV baselines.
2. Optuna study for each key model.
3. Train final model with best params on full training set.
4. Save best params and results.
"""

import json
import warnings
import numpy as np
import pandas as pd
import joblib
import optuna
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

RANDOM_STATE  = 42
CV_FOLDS      = 5
N_OPTUNA      = 50      # número de trials Optuna por modelo
MODELS_DIR    = Path("models/trained_models")
METRICS_DIR   = Path("results/metrics")
PLOTS_DIR     = Path("results/plots")
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Sklearn Grid / Randomized Search baselines
# ─────────────────────────────────────────────────────────────────────────────
 
def run_grid_search(X_train: pd.DataFrame, y_train: pd.Series) -> dict:
    """
    Runs GridSearchCV on Ridge and a small RandomForest grid.
 
    GridSearchCV exhaustively tests every combination → suitable for small
    parameter spaces (e.g., Ridge alpha). For large spaces, RandomizedSearchCV
    is used instead (see run_randomized_search).
 
    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training target.
 
    Returns
    -------
    dict
        {model_name: best_params}
    """
    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    results = {}
 
    # --- Ridge ---
    print("\n>> GridSearchCV – Ridge")
    ridge_pipe = Pipeline([("scaler", StandardScaler()), ("model", Ridge())])
    ridge_params = {"model__alpha": [0.001, 0.01, 0.1, 1, 10, 50, 100, 500, 1000]}
    gs_ridge = GridSearchCV(ridge_pipe, ridge_params, cv=kf, scoring="r2", n_jobs=-1)
    gs_ridge.fit(X_train, y_train)
    print(f"  Best alpha: {gs_ridge.best_params_}  |  R²={gs_ridge.best_score_:.4f}")
    results["Ridge"] = gs_ridge.best_params_
 
    # --- RandomForest (small grid) ---
    print("\n>> GridSearchCV – RandomForest (small grid)")
    rf_pipe = Pipeline([("model", RandomForestRegressor(n_jobs=-1, random_state=RANDOM_STATE))])
    rf_small_params = {
        "model__n_estimators": [100, 200],
        "model__max_depth":    [10, 15, None],
    }
    gs_rf = GridSearchCV(rf_pipe, rf_small_params, cv=kf, scoring="r2", n_jobs=-1)
    gs_rf.fit(X_train, y_train)
    print(f"  Best params: {gs_rf.best_params_}  |  R²={gs_rf.best_score_:.4f}")
    results["RandomForest_grid"] = gs_rf.best_params_
 
    return results
 
 
def run_randomized_search(X_train: pd.DataFrame, y_train: pd.Series, n_iter: int = 30) -> dict:
    """
    Runs RandomizedSearchCV on GradientBoosting (large param space).
 
    RandomizedSearchCV samples n_iter random combinations instead of
    exhaustive search, making it practical for models with many hyperparameters.
 
    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training target.
    n_iter : int
        Number of random combinations to evaluate.
 
    Returns
    -------
    dict
        {model_name: best_params}
    """
    from scipy.stats import randint, uniform
 
    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    results = {}
 
    print(f"\n>> RandomizedSearchCV – GradientBoosting ({n_iter} iterations)")
    gb_pipe = Pipeline([
        ("model", GradientBoostingRegressor(random_state=RANDOM_STATE))
    ])
    gb_params = {
        "model__n_estimators":  randint(100, 500),
        "model__learning_rate": uniform(0.01, 0.2),
        "model__max_depth":     randint(3, 8),
        "model__subsample":     uniform(0.7, 0.3),
        "model__min_samples_leaf": randint(1, 20),
    }
    rs_gb = RandomizedSearchCV(
        gb_pipe, gb_params, n_iter=n_iter,
        cv=kf, scoring="r2", n_jobs=-1, random_state=RANDOM_STATE
    )
    rs_gb.fit(X_train, y_train)
    print(f"  Best params: {rs_gb.best_params_}  |  R²={rs_gb.best_score_:.4f}")
    results["GradientBoosting_rand"] = rs_gb.best_params_
    return results
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Optuna studies
# ─────────────────────────────────────────────────────────────────────────────
 
def _objective_rf(trial, X: np.ndarray, y: np.ndarray, kf: KFold) -> float:
    """Optuna objective for RandomForestRegressor."""
    params = {
        "n_estimators":     trial.suggest_int("n_estimators", 50, 400),
        "max_depth":        trial.suggest_int("max_depth", 5, 30),
        "min_samples_split":trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features":     trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5]),
    }
    model = RandomForestRegressor(n_jobs=-1, random_state=RANDOM_STATE, **params)
    scores = cross_val_score(model, X, y, cv=kf, scoring="r2", n_jobs=-1)
    return scores.mean()
 
 
def _objective_gb(trial, X: np.ndarray, y: np.ndarray, kf: KFold) -> float:
    """Optuna objective for GradientBoostingRegressor."""
    params = {
        "n_estimators":      trial.suggest_int("n_estimators", 100, 600),
        "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth":         trial.suggest_int("max_depth", 3, 8),
        "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
        "min_samples_leaf":  trial.suggest_int("min_samples_leaf", 1, 20),
        "max_features":      trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
    }
    model = GradientBoostingRegressor(random_state=RANDOM_STATE, **params)
    scores = cross_val_score(model, X, y, cv=kf, scoring="r2", n_jobs=-1)
    return scores.mean()
 
 
def run_optuna(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_trials: int = N_OPTUNA,
) -> dict:
    """
    Runs Optuna Bayesian optimisation for RandomForest and GradientBoosting.
 
    Optuna uses Tree-structured Parzen Estimators (TPE) to intelligently
    navigate the hyperparameter space, converging faster than grid search.
 
    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training target.
    n_trials : int
        Number of Optuna trials per model.
 
    Returns
    -------
    dict
        {model_name: {'best_params': ..., 'best_value': ...}}
    """
    X = X_train.values
    y = y_train.values
    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    all_results = {}
 
    for model_tag, objective_fn in [
        ("RandomForest_optuna",      _objective_rf),
        ("GradientBoosting_optuna",  _objective_gb),
    ]:
        print(f"\n>> Optuna study: {model_tag} ({n_trials} trials)...")
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
        )
        study.optimize(
            lambda trial, fn=objective_fn: fn(trial, X, y, kf),
            n_trials=n_trials,
            show_progress_bar=False,
        )
        best = study.best_trial
        print(f"  OK: Best R² = {best.value:.4f}")
        print(f"  >> Best params: {best.params}")
 
        all_results[model_tag] = {
            "best_params": best.params,
            "best_value":  best.value,
            "study":       study,
        }
        _plot_optuna_history(study, model_tag)
        _plot_optuna_importance(study, model_tag)
 
    return all_results
 
 
def _plot_optuna_history(study, model_tag: str) -> None:
    """Plots the optimisation history (best value per trial)."""
    trials_df = study.trials_dataframe()
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(trials_df["number"], trials_df["value"], alpha=0.5, lw=1, label="Trial R²")
    best_so_far = trials_df["value"].cummax()
    ax.plot(trials_df["number"], best_so_far, color="red", lw=2, label="Best so far")
    ax.set_xlabel("Trial"); ax.set_ylabel("R² (CV mean)")
    ax.set_title(f"Optuna Optimisation History – {model_tag}")
    ax.legend()
    plt.tight_layout()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOTS_DIR / f"{model_tag}_optuna_history.png", dpi=150)
    plt.close(fig)
    print(f"  Guardado: Saved: {PLOTS_DIR}/{model_tag}_optuna_history.png")
 
 
def _plot_optuna_importance(study, model_tag: str) -> None:
    """Plots hyperparameter importance from Optuna."""
    try:
        importances = optuna.importance.get_param_importances(study)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.barh(list(importances.keys()), list(importances.values()), color="steelblue")
        ax.set_xlabel("Importance")
        ax.set_title(f"Hyperparameter Importance – {model_tag}")
        plt.tight_layout()
        fig.savefig(PLOTS_DIR / f"{model_tag}_param_importance.png", dpi=150)
        plt.close(fig)
        print(f"  Guardado: Saved: {PLOTS_DIR}/{model_tag}_param_importance.png")
    except Exception as e:
        print(f"ADVERTENCIA:  Could not plot param importance: {e}")
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Save / compare all tuning results
# ─────────────────────────────────────────────────────────────────────────────
 
def save_tuning_results(grid_results: dict, rand_results: dict, optuna_results: dict) -> pd.DataFrame:
    """
    Consolidates all tuning results into a JSON report and summary CSV.
 
    Parameters
    ----------
    grid_results, rand_results : dict
        Outputs of run_grid_search / run_randomized_search.
    optuna_results : dict
        Output of run_optuna.
 
    Returns
    -------
    pd.DataFrame
        Summary of best scores per method/model.
    """
    all_params = {}
    all_params.update(grid_results)
    all_params.update(rand_results)
    for name, res in optuna_results.items():
        all_params[name] = {"best_params": res["best_params"], "best_r2": res["best_value"]}
 
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    with open(METRICS_DIR / "best_hyperparams.json", "w") as f:
        # Convert params to JSON-serialisable types
        serialisable = {
            k: {kk: (int(vv) if isinstance(vv, (np.integer,)) else
                     float(vv) if isinstance(vv, (np.floating,)) else vv)
                for kk, vv in v.items()} if isinstance(v, dict) else v
            for k, v in all_params.items()
        }
        json.dump(serialisable, f, indent=2, default=str)
    print(f"  Guardado: Best hyperparams saved → {METRICS_DIR}/best_hyperparams.json")
 
    # Summary table for optuna
    rows = [
        {"Method": "GridSearchCV",       "Model": k, "Best R²": "see cv_results.csv"}
        for k in grid_results
    ] + [
        {"Method": "RandomizedSearchCV", "Model": k, "Best R²": "see cv_results.csv"}
        for k in rand_results
    ] + [
        {"Method": "Optuna (TPE)", "Model": k, "Best R²": round(v["best_value"], 4)}
        for k, v in optuna_results.items()
    ]
    df = pd.DataFrame(rows)
    df.to_csv(METRICS_DIR / "tuning_summary.csv", index=False)
    print(f"  Guardado: Tuning summary → {METRICS_DIR}/tuning_summary.csv")
    return df
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Entrypoint
# ─────────────────────────────────────────────────────────────────────────────
 
def run_tuning(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_optuna_trials: int = N_OPTUNA,
    n_rand_iter: int = 30,
) -> str:
    """
    Full tuning pipeline: Grid → Randomized → Optuna.
 
    Selects the best overall model by R², trains it on the full training set,
    serialises it, and returns the winning model name.
 
    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training targets.
    n_optuna_trials : int
        Trials for Optuna optimisation.
    n_rand_iter : int
        Iterations for RandomizedSearchCV.
 
    Returns
    -------
    str
        Name of the best model (key in MODELS_DIR).
    """
    print("\n" + "="*60)
    print("  HYPERPARAMETER TUNING")
    print("="*60)
 
    grid_res   = run_grid_search(X_train, y_train)
    rand_res   = run_randomized_search(X_train, y_train, n_iter=n_rand_iter)
    optuna_res = run_optuna(X_train, y_train, n_trials=n_optuna_trials)
    df_summary = save_tuning_results(grid_res, rand_res, optuna_res)
 
    # Pick best Optuna model
    best_entry = max(optuna_res.items(), key=lambda x: x[1]["best_value"])
    best_name  = best_entry[0]   # e.g. "GradientBoosting_optuna"
    best_r2    = best_entry[1]["best_value"]
    best_params= best_entry[1]["best_params"]
 
    print(f"\n>> Best model overall: {best_name}  (R²={best_r2:.4f})")
 
    # Determine sklearn class
    if "RandomForest" in best_name:
        base_class = RandomForestRegressor
    else:
        base_class = GradientBoostingRegressor
 
    _supports_n_jobs = {"RandomForest", "ExtraTrees", "KNN"}
    _n_jobs_kwarg = {"n_jobs": -1} if any(m in best_name for m in _supports_n_jobs) else {}
    final_model = Pipeline([
        ("model", base_class(random_state=RANDOM_STATE, **_n_jobs_kwarg,
                             **{k: int(v) if isinstance(v, (np.integer,)) else v
                                for k, v in best_params.items()}))
    ])
    print(f"\n>> Training best model ({best_name}) on full training set...")
    final_model.fit(X_train, y_train)
 
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / f"{best_name}_final.pkl"
    joblib.dump(final_model, model_path)
    print(f"OK: Final model saved → {model_path}")
 
    return best_name
 
 
if __name__ == "__main__":
    X_train = pd.read_csv("data/processed/X_train.csv")
    y_train = pd.read_csv("data/processed/y_train.csv").squeeze()
    best = run_tuning(X_train, y_train, n_optuna_trials=50, n_rand_iter=30)
    print(f"\nOK: Tuning complete. Best model: {best}")