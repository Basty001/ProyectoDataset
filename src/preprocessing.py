"""
Preprocessing Module.
Loads raw data, applies the full cleaning pipeline, and splits into
Train (80%) and Test (20%) sets. The test set is saved and NEVER touched
again until final evaluation.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

from src.audit import audit_data
from src.optimization import optimize_memory
from src.pipeline import build_preprocessing_pipeline


RANDOM_STATE = 42
TEST_SIZE = 0.20
TARGET_COL = "MSRP"
COLUMNS_TO_DROP = ["Model"]


def load_raw_data(raw_dir: str = "data/raw") -> pd.DataFrame:
    """
    Loads the first CSV found in raw_dir.

    Parameters
    ----------
    raw_dir : str
        Path to the directory containing raw CSV files.

    Returns
    -------
    pd.DataFrame
        Raw DataFrame as loaded from disk.

    Raises
    ------
    FileNotFoundError
        If no CSV is found in raw_dir.
    """
    raw_path = Path(raw_dir)
    csv_files = list(raw_path.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {raw_dir}")
    df = pd.read_csv(csv_files[0])
    print(f"OK: Raw data loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


def run_preprocessing(
    raw_dir: str = "data/raw",
    processed_dir: str = "data/processed",
    skip_audit: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Full preprocessing pipeline: audit → optimize → clean → split → save.

    Steps
    -----
    1. SHA-256 audit of the raw file.
    2. Memory optimisation (downcasting numeric types).
    3. Scikit-learn pipeline: drop high-cardinality cols, replace UNKNOWN→NaN,
       drop cols with >25 % nulls, smart imputation, outlier capping,
       variance thresholding, StandardScaler + OneHotEncoder.
    4. Stratified 80/20 split on MSRP quantile bins.
    5. Save ``X_train``, ``X_test``, ``y_train``, ``y_test`` as CSVs.

    Parameters
    ----------
    raw_dir : str
        Directory containing raw CSV files.
    processed_dir : str
        Output directory for processed splits.
    skip_audit : bool
        If True, skips the hash audit (useful for CI/testing).

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (df_train, df_test) with all features + target column.
    """
    # ── 1. Audit ────────────────────────────────────────────────────────────
    if not skip_audit:
        print("\n>> Step 1: Auditing raw data...")
        if not audit_data():
            raise RuntimeError("Data audit failed. Pipeline aborted.")
    else:
        print("\nADVERTENCIA:  Step 1: Audit skipped.")

    # ── 2. Load & Optimise ──────────────────────────────────────────────────
    print("\n>> Step 2: Loading and optimising memory...")
    df_raw = load_raw_data(raw_dir)
    df_opt = optimize_memory(df_raw)

    # ── 3. Train / Test Split (BEFORE pipeline fitting to avoid leakage) ───
    print(f"\n>>  Step 3: Splitting {1-TEST_SIZE:.0%} train / {TEST_SIZE:.0%} test...")

    # Stratify on quantile bins so both splits cover the full price range
    df_opt["_msrp_bin"] = pd.qcut(df_opt[TARGET_COL], q=10, labels=False, duplicates="drop")
    df_train_raw, df_test_raw = train_test_split(
        df_opt,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df_opt["_msrp_bin"],
    )
    df_train_raw = df_train_raw.drop(columns=["_msrp_bin"]).reset_index(drop=True)
    df_test_raw  = df_test_raw.drop(columns=["_msrp_bin"]).reset_index(drop=True)
    print(f"   Train: {df_train_raw.shape[0]:,} rows  |  Test: {df_test_raw.shape[0]:,} rows")

    # ── 4. Fit pipeline on TRAIN only, transform both ──────────────────────
    print("\n>>  Step 4: Fitting preprocessing pipeline on training set...")
    pipeline = build_preprocessing_pipeline(
        df_train_raw, target_col=TARGET_COL, columns_to_drop=COLUMNS_TO_DROP
    )

    train_matrix = pipeline.fit_transform(df_train_raw)
    test_matrix  = pipeline.transform(df_test_raw)

    feature_names = pipeline.named_steps["preprocessing"].get_feature_names_out()
    clean_names   = [name.split("__")[-1] for name in feature_names]

    df_train = pd.DataFrame(train_matrix, columns=clean_names)
    df_test  = pd.DataFrame(test_matrix,  columns=clean_names)

    # ── 5. Separate target ─────────────────────────────────────────────────
    # After the ColumnTransformer the target lands in the 'remainder' block.
    # Its column name ends with 'MSRP'.
    target_col_final = [c for c in df_train.columns if TARGET_COL in c]
    if not target_col_final:
        raise ValueError(f"Target column '{TARGET_COL}' not found after pipeline.")
    target_col_final = target_col_final[0]

    X_train = df_train.drop(columns=[target_col_final])
    y_train = df_train[target_col_final].rename(TARGET_COL)
    X_test  = df_test.drop(columns=[target_col_final])
    y_test  = df_test[target_col_final].rename(TARGET_COL)

    print(f"\n>> Final feature matrix: {X_train.shape[1]} features")

    # ── 6. Save splits ─────────────────────────────────────────────────────
    print("\n  Guardado: Step 5: Saving processed splits...")
    out = Path(processed_dir)
    out.mkdir(parents=True, exist_ok=True)

    X_train.to_csv(out / "X_train.csv", index=False)
    X_test.to_csv(out / "X_test.csv",   index=False)
    y_train.to_csv(out / "y_train.csv", index=False)
    y_test.to_csv(out / "y_test.csv",   index=False)

    print(f"OK: Saved → {processed_dir}/X_train.csv  ({X_train.shape})")
    print(f"OK: Saved → {processed_dir}/X_test.csv   ({X_test.shape})")
    print(f"OK: Saved → {processed_dir}/y_train.csv  ({y_train.shape})")
    print(f"OK: Saved → {processed_dir}/y_test.csv   ({y_test.shape})")

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    run_preprocessing()
