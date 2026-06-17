"""
Main Orchestration Script.
 
Runs the full machine learning pipeline end-to-end:
 
  1. SHA-256 audit + memory optimisation
  2. Preprocessing -> Train/Test split (80/20)
  3. Unsupervised learning (PCA + KMeans / DBSCAN / Agglomerative)
  4. Cross-validation comparison of supervised models
  5. Hyperparameter tuning (GridSearchCV + RandomizedSearchCV + Optuna)
  6. Train final model on complete training set
  7. Final evaluation on held-out test set (first and only time)
 
Usage
-----
  python main.py [--skip-audit] [--fast]
 
  --skip-audit  : Skip SHA-256 hash check (useful for CI)
  --fast        : Reduced trials/iterations for quick testing
"""
 
import argparse
import sys
import traceback
from pathlib import Path
 
# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).parent))
 
from src.preprocessing import run_preprocessing
from src.unsupervised import run_unsupervised
from src.model_training import get_model_candidates
from src.model_evaluation import (
    cross_validate_models,
    evaluate_on_test,
    plot_model_comparison,
    classification_report_by_price,
)
from src.tune import run_tuning
 
RANDOM_STATE = 42  # Global seed for reproducibility
 
 
def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Car Price ML Pipeline")
    parser.add_argument(
        "--skip-audit",
        action="store_true",
        help="Skip SHA-256 audit of raw data",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use fewer Optuna trials for quick testing",
    )
    return parser.parse_args()
 
 
def main():
    """Execute the full 7-step ML pipeline."""
    args = parse_args()
    n_optuna = 10 if args.fast else 50
    n_rand = 10 if args.fast else 30
 
    print("=" * 65)
    print("    CAR PRICE ML PIPELINE  -  SCY1101 Evaluacion Parcial N2")
    print("=" * 65)
 
    # STEP 1 & 2: Audit + Preprocessing
    print("\n\n>> STEP 1-2: Preprocessing")
    x_train, x_test, y_train, y_test = run_preprocessing(
        skip_audit=args.skip_audit
    )
 
    # STEP 3: Unsupervised Learning
    print("\n\n>> STEP 3: Unsupervised Learning (PCA + Clustering)")
    run_unsupervised(x_train)
 
    # STEP 4: Cross-Validation comparison
    print("\n\n>> STEP 4: Supervised Model Cross-Validation")
    models = get_model_candidates()
    cv_df = cross_validate_models(models, x_train, y_train)
    plot_model_comparison(cv_df)
    print(f"\n>> Best model by CV R2: {cv_df.iloc[0]['Model']}")
 
    # STEP 5: Hyperparameter Tuning
    print("\n\n>> STEP 5: Hyperparameter Tuning (Grid + Randomized + Optuna)")
    best_model_name = run_tuning(
        x_train, y_train,
        n_optuna_trials=n_optuna,
        n_rand_iter=n_rand,
    )
 
    # STEP 6: Final model confirmation
    print(f"\n\n>> STEP 6: Final model '{best_model_name}' trained on full training set OK")
 
    # STEP 7: Test Set Evaluation (FIRST AND ONLY TIME)
    print("\n\n>> STEP 7: Final Test-Set Evaluation (first and only time)")
    import joblib  # noqa: PLC0415
    model_path = Path("models/trained_models") / f"{best_model_name}_final.pkl"
    final_model = joblib.load(model_path)
 
    test_metrics = evaluate_on_test(
        final_model, x_test, y_test, model_name=best_model_name
    )
    classification_report_by_price(
        y_test, final_model.predict(x_test), model_name=best_model_name
    )
 
    print("\n" + "=" * 65)
    print("  OK:  PIPELINE COMPLETE")
    print(f"  Final R2  : {test_metrics['R²']}")
    print(f"  Final MAE : ${test_metrics['MAE']:,.2f}")
    print(f"  Final RMSE: ${test_metrics['RMSE']:,.2f}")
    print(f"  Final MAPE: {test_metrics['MAPE %']}%")
    print("=" * 65)
    print("\nOutputs saved in:")
    print("  results/plots/    - all visualisations")
    print("  results/metrics/  - CSV metric files")
    print("  results/reports/  - price-bin classification reports")
    print("  models/trained_models/ - serialised final model")
 
 
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nPipeline interrupted by user.")
        sys.exit(0)
    except Exception as exc:  # noqa: BLE001
        print(f"\nFATAL ERROR: {exc}")
        traceback.print_exc()
        sys.exit(1)
 