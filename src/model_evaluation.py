"""
Model Evaluation Module.
Cross-validation on training set, final test-set evaluation,
classification report (via price-range bins), confusion matrix,
and comparative visualisations.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.model_selection import cross_validate, KFold
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
)

RANDOM_STATE = 42
CV_FOLDS     = 5
PLOTS_DIR    = Path("results/plots")
METRICS_DIR  = Path("results/metrics")
REPORTS_DIR  = Path("results/reports")


# ─────────────────────────────────────────────────────────────────────────────
# Cross-validation on training set
# ─────────────────────────────────────────────────────────────────────────────

def cross_validate_models(
    models: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: int = CV_FOLDS,
) -> pd.DataFrame:
    """
    Runs stratified k-fold cross-validation for each model pipeline.

    Metrics computed: R², MAE, RMSE, MAPE.
    Results are averaged across folds (mean ± std).

    Parameters
    ----------
    models : dict
        {name: fitted_or_unfitted_Pipeline}
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training targets.
    cv : int
        Number of cross-validation folds.

    Returns
    -------
    pd.DataFrame
        Summary table with mean/std for each metric per model.
    """
    print(f"\n>> Running {cv}-Fold Cross-Validation on {len(models)} models...\n")
    kf = KFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)

    rows = []
    for name, pipe in models.items():
        scoring = {
            "r2":  "r2",
            "mae": "neg_mean_absolute_error",
            "mse": "neg_mean_squared_error",
        }
        cv_results = cross_validate(
            pipe, X_train, y_train, cv=kf, scoring=scoring,
            return_train_score=False, n_jobs=-1
        )
        r2_mean  = cv_results["test_r2"].mean()
        r2_std   = cv_results["test_r2"].std()
        mae_mean = -cv_results["test_mae"].mean()
        mae_std  =  cv_results["test_mae"].std()
        rmse_mean = np.sqrt(-cv_results["test_mse"].mean())
        rmse_std  = np.sqrt( cv_results["test_mse"].std())

        print(f"  {name:<20} R²={r2_mean:.4f}±{r2_std:.4f}  "
              f"MAE={mae_mean:,.0f}±{mae_std:,.0f}  RMSE={rmse_mean:,.0f}")

        rows.append({
            "Model": name,
            "R² (CV mean)": round(r2_mean, 4),
            "R² (CV std)":  round(r2_std, 4),
            "MAE (CV mean)": round(mae_mean, 2),
            "MAE (CV std)":  round(mae_std, 2),
            "RMSE (CV mean)": round(rmse_mean, 2),
        })

    df = pd.DataFrame(rows).sort_values("R² (CV mean)", ascending=False).reset_index(drop=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(METRICS_DIR / "cv_results.csv", index=False)
    print(f"\n  Guardado: CV results saved → {METRICS_DIR}/cv_results.csv")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Final test-set evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_on_test(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = "Model",
) -> dict:
    """
    Evaluates a fitted model on the held-out test set.

    ADVERTENCIA:  This function should be called ONLY ONCE, after all modelling decisions
    have been made (model selection + hyperparameter tuning completed).

    Parameters
    ----------
    model : fitted Pipeline
        The final model to evaluate.
    X_test : pd.DataFrame
        Test features (never seen during training).
    y_test : pd.Series
        True test targets.
    model_name : str
        Label for reports and plots.

    Returns
    -------
    dict
        Dictionary with all computed metrics.
    """
    y_pred = model.predict(X_test)

    metrics = {
        "Model":  model_name,
        "R²":     round(r2_score(y_test, y_pred), 4),
        "MAE":    round(mean_absolute_error(y_test, y_pred), 2),
        "RMSE":   round(np.sqrt(mean_squared_error(y_test, y_pred)), 2),
        "MAPE %": round(mean_absolute_percentage_error(y_test, y_pred) * 100, 2),
    }

    print(f"\n{'='*55}")
    print(f"  TEST SET EVALUATION – {model_name}")
    print(f"{'='*55}")
    for k, v in metrics.items():
        if k != "Model":
            print(f"  {k:<10}: {v:>12,.4f}")
    print(f"{'='*55}")

    # Save metrics
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([metrics]).to_csv(
        METRICS_DIR / f"{model_name}_test_metrics.csv", index=False
    )

    # Plots
    _plot_predictions_vs_actual(y_test, y_pred, model_name)
    _plot_residuals(y_test, y_pred, model_name)
    _plot_price_bin_confusion(y_test, y_pred, model_name)

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

def _plot_predictions_vs_actual(y_true, y_pred, model_name: str) -> None:
    """Scatter plot of predicted vs actual values with perfect-prediction line."""
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_true, y_pred, alpha=0.3, s=8, color="steelblue")
    lim = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lim, lim, "r--", lw=1.5, label="Perfect prediction")
    ax.set_xlabel("Actual MSRP (USD)")
    ax.set_ylabel("Predicted MSRP (USD)")
    ax.set_title(f"{model_name} – Predicted vs Actual")
    ax.legend()
    plt.tight_layout()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOTS_DIR / f"{model_name}_pred_vs_actual.png", dpi=150)
    plt.close(fig)
    print(f"  Guardado: Saved: {PLOTS_DIR}/{model_name}_pred_vs_actual.png")


def _plot_residuals(y_true, y_pred, model_name: str) -> None:
    """Residual distribution + residuals vs predicted scatter."""
    residuals = np.array(y_true) - np.array(y_pred)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    sns.histplot(residuals, bins=50, kde=True, ax=ax1, color="coral")
    ax1.axvline(0, color="black", lw=1)
    ax1.set_title(f"{model_name} – Residual Distribution")
    ax1.set_xlabel("Residual (Actual − Predicted)")

    ax2.scatter(y_pred, residuals, alpha=0.3, s=8, color="coral")
    ax2.axhline(0, color="black", lw=1)
    ax2.set_title(f"{model_name} – Residuals vs Predicted")
    ax2.set_xlabel("Predicted MSRP"); ax2.set_ylabel("Residual")

    plt.tight_layout()
    fig.savefig(PLOTS_DIR / f"{model_name}_residuals.png", dpi=150)
    plt.close(fig)
    print(f"  Guardado: Saved: {PLOTS_DIR}/{model_name}_residuals.png")


def _plot_price_bin_confusion(y_true, y_pred, model_name: str, n_bins: int = 5) -> None:
    """
    Discretises MSRP into price-range bins and plots a confusion-matrix-style
    heatmap to show classification accuracy by price segment.

    Bin edges are computed from the combined true + predicted range so both
    arrays use identical thresholds.
    """
    combined = np.concatenate([y_true, y_pred])
    bin_edges = np.percentile(combined, np.linspace(0, 100, n_bins + 1))
    bin_edges = np.unique(bin_edges)  # avoid duplicate edges
    labels_str = [
        f"${bin_edges[i]:,.0f}–${bin_edges[i+1]:,.0f}"
        for i in range(len(bin_edges) - 1)
    ]
    n_bins_actual = len(labels_str)

    y_true_bin = pd.cut(y_true, bins=bin_edges, labels=labels_str, include_lowest=True)
    y_pred_bin = pd.cut(y_pred, bins=bin_edges, labels=labels_str, include_lowest=True)

    conf = pd.crosstab(y_true_bin, y_pred_bin, rownames=["Actual"], colnames=["Predicted"])
    # Reindex to ensure all labels appear
    conf = conf.reindex(index=labels_str, columns=labels_str, fill_value=0)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(conf, annot=True, fmt="d", cmap="Blues", ax=ax,
                linewidths=0.5, cbar_kws={"label": "Count"})
    ax.set_title(f"{model_name} – Price-Range Confusion Matrix")
    plt.xticks(rotation=30, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / f"{model_name}_confusion_matrix.png", dpi=150)
    plt.close(fig)
    print(f"  Guardado: Saved: {PLOTS_DIR}/{model_name}_confusion_matrix.png")


def plot_model_comparison(cv_df: pd.DataFrame) -> None:
    """
    Bar chart comparing R² across all CV-evaluated models.

    Parameters
    ----------
    cv_df : pd.DataFrame
        Output of cross_validate_models().
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(cv_df["Model"], cv_df["R² (CV mean)"],
                   xerr=cv_df["R² (CV std)"], color="steelblue",
                   edgecolor="white", capsize=4)
    ax.set_xlabel("R² (mean ± std over 5 folds)")
    ax.set_title("Model Comparison – Cross-Validation R²")
    ax.axvline(0, color="black", lw=0.8)
    for bar, val in zip(bars, cv_df["R² (CV mean)"]):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                f"{val:.4f}", va="center", fontsize=9)
    plt.tight_layout()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOTS_DIR / "model_comparison_r2.png", dpi=150)
    plt.close(fig)
    print(f"  Guardado: Saved: {PLOTS_DIR}/model_comparison_r2.png")


def classification_report_by_price(
    y_true: pd.Series,
    y_pred: np.ndarray,
    model_name: str = "Model",
    n_bins: int = 5,
) -> pd.DataFrame:
    """
    Generates a 'classification report' by discretising prices into bins.

    Computes precision-like metrics (within-bin accuracy) for each price range.

    Parameters
    ----------
    y_true, y_pred : arrays
        True and predicted MSRP values.
    model_name : str
        Label for the report file.
    n_bins : int
        Number of price segments.

    Returns
    -------
    pd.DataFrame
        Per-bin metrics.
    """
    combined = np.concatenate([np.array(y_true), np.array(y_pred)])
    bin_edges = np.percentile(combined, np.linspace(0, 100, n_bins + 1))
    bin_edges = np.unique(bin_edges)
    labels_str = [
        f"${bin_edges[i]:,.0f}–${bin_edges[i+1]:,.0f}"
        for i in range(len(bin_edges) - 1)
    ]

    y_true_bin = pd.cut(y_true, bins=bin_edges, labels=labels_str, include_lowest=True)
    y_pred_bin = pd.cut(y_pred, bins=bin_edges, labels=labels_str, include_lowest=True)

    rows = []
    for lbl in labels_str:
        mask = y_true_bin == lbl
        if mask.sum() == 0:
            continue
        n_total   = mask.sum()
        n_correct = (y_pred_bin[mask] == lbl).sum()
        mae_bin   = mean_absolute_error(
            np.array(y_true)[mask], np.array(y_pred)[mask]
        )
        rows.append({
            "Price Range":   lbl,
            "N samples":     int(n_total),
            "Correct bin %": round(100 * n_correct / n_total, 1),
            "MAE (USD)":     round(mae_bin, 0),
        })

    df_report = pd.DataFrame(rows)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    df_report.to_csv(REPORTS_DIR / f"{model_name}_price_bin_report.csv", index=False)
    print(f"\n>> Price-Bin Classification Report ({model_name}):")
    print(df_report.to_string(index=False))
    return df_report


if __name__ == "__main__":
    from src.model_training import get_model_candidates, load_model
    X_train = pd.read_csv("data/processed/X_train.csv")
    y_train = pd.read_csv("data/processed/y_train.csv").squeeze()
    X_test  = pd.read_csv("data/processed/X_test.csv")
    y_test  = pd.read_csv("data/processed/y_test.csv").squeeze()

    models = get_model_candidates()
    cv_df  = cross_validate_models(models, X_train, y_train)
    plot_model_comparison(cv_df)

    # Final evaluation with saved model (run after tune.py)
    try:
        best_model = load_model("RandomForest")
        evaluate_on_test(best_model, X_test, y_test, model_name="RandomForest")
    except FileNotFoundError:
        print("ADVERTENCIA:  No saved model found. Run tune.py first.")
