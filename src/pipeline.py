import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_selection import VarianceThreshold
from src.transformers import (
    DropColumnsTransformer, UnknownToNaNTransformer,
    DropHighMissingTransformer, SmartImputerTransformer, OutlierCapper
)

def build_preprocessing_pipeline(df, target_col='MSRP', columns_to_drop=None):
    if columns_to_drop is None:
        columns_to_drop = []

    # TRUCO: Funciones dinámicas que se ejecutan en el Paso 5
    # Revisan qué columnas SOBREVIVIERON a los pasos anteriores, e ignoran el target
    def select_numeric(X):
        return [col for col in X.select_dtypes(include=['number']).columns if col != target_col]
        
    def select_categorical(X):
        return [col for col in X.select_dtypes(exclude=['number']).columns if col != target_col]

    num_pipe = Pipeline([
        ('capper', OutlierCapper(apply_capping=True)),
        ('zero_variance', VarianceThreshold(threshold=0.0)),
        ('scaler', StandardScaler()),
    ])

    cat_pipe = Pipeline([
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
    ])

    # Enrutador con selección dinámica
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_pipe, select_numeric),
            ('cat', cat_pipe, select_categorical),
        ], remainder='passthrough' # Salva el MSRP
    )

    full_pipeline = Pipeline([
        ('drop_cols',       DropColumnsTransformer(columns_to_drop=columns_to_drop)),
        ('clean_unknowns',  UnknownToNaNTransformer()),
        ('drop_high_nan',   DropHighMissingTransformer(threshold=0.25)),
        ('smart_imputer',   SmartImputerTransformer(low_threshold=0.10)),
        ('preprocessing',   preprocessor),
    ])

    return full_pipeline