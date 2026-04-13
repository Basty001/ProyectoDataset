"""
Pipeline Construction Module.
Builds the full Scikit-Learn preprocessing pipeline for the Car Price dataset.
"""

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from src.transformers import (
    DropColumnsTransformer,
    UnknownToNaNTransformer,
    DropHighMissingTransformer,
    SmartImputerTransformer,
    OutlierCapper,
    DropZeroVarianceTransformer,
)


def build_preprocessing_pipeline(df, columns_to_drop=None):
    """
    Builds and returns a generic scikit-learn preprocessing pipeline.
    Dynamically detects numeric and categorical columns.
    
    Por qué usar Pipeline de Scikit-Learn:
    - Reproducibilidad: el mismo código procesa entrenamiento y producción.
    - Previene Data Leakage: fit() aprende SOLO en el conjunto de entrenamiento.
    - Modularidad: cada paso es independiente y testeable.
    """
    if columns_to_drop is None:
        columns_to_drop = []

    # Ruta para variables numéricas: capping → varianza cero → escalado
    num_pipe = Pipeline([
        ('capper', OutlierCapper(apply_capping=True)),
        ('zero_variance', DropZeroVarianceTransformer()),
        ('scaler', StandardScaler()),
    ])

    # Ruta para variables categóricas: encoding con manejo de categorías nuevas
    cat_pipe = Pipeline([
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
    ])

    # ColumnTransformer: enruta cada columna a su pipeline según tipo de dato
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_pipe, make_column_selector(dtype_include='number')),
            ('cat', cat_pipe, make_column_selector(dtype_exclude='number')),
        ],
        remainder='drop'
    )

    # Pipeline completo: limpieza estructural → preprocesamiento para ML
    full_pipeline = Pipeline([
        # Paso 1: eliminar columnas de alta cardinalidad (Model: 915 valores únicos)
        ('drop_cols',       DropColumnsTransformer(columns_to_drop=columns_to_drop)),
        # Paso 2: convertir 'UNKNOWN' string a NaN verdadero
        ('clean_unknowns',  UnknownToNaNTransformer()),
        # Paso 3: eliminar columnas con >25% de nulos (Market Category: 31.4%)
        ('drop_high_nan',   DropHighMissingTransformer(threshold=0.25)),
        # Paso 4: imputar nulos restantes con mediana/moda según tipo y porcentaje
        ('smart_imputer',   SmartImputerTransformer(low_threshold=0.10)),
        # Paso 5: escalar numéricos + codificar categóricas
        ('preprocessing',   preprocessor),
    ])

    return full_pipeline
