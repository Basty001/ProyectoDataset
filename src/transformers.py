"""
Custom Scikit-Learn Transformers.
Contains classes for structural cleaning, outlier capping, and smart imputation.
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class DropColumnsTransformer(BaseEstimator, TransformerMixin):
    """
    Drops specified columns from the DataFrame.
    
    Uso en este proyecto: elimina 'Model' (915 valores únicos = alta cardinalidad).
    OneHotEncoding de Model generaría 915 columnas nuevas, causando overfitting.
    """
    def __init__(self, columns_to_drop):
        self.columns_to_drop = columns_to_drop

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        # Solo elimina si la columna realmente existe en el dataset
        cols = [col for col in self.columns_to_drop if col in X_copy.columns]
        return X_copy.drop(columns=cols)


class UnknownToNaNTransformer(BaseEstimator, TransformerMixin):
    """
    Converts masked null values (strings like 'UNKNOWN') into true numpy NaN.
    
    Problema: 'Transmission Type' tiene 19 registros con el string 'UNKNOWN'.
    Pandas no los detecta como nulos, entonces isnull() los ignora.
    Este transformer los convierte a NaN para que SmartImputer los trate.
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Reemplaza el texto enmascarado por nulos matemáticos reales
        return X.replace('UNKNOWN', np.nan)


class DropHighMissingTransformer(BaseEstimator, TransformerMixin):
    """
    Drops columns that exceed a specific threshold of missing values.
    
    Uso en este proyecto: 'Market Category' tiene 31.4% de nulos.
    Además sus valores son multi-etiqueta ('Luxury,Performance'), lo que
    complica el encoding. Se descarta con threshold=0.25.
    """
    def __init__(self, threshold=0.8):
        self.threshold = threshold
        self.cols_to_drop_ = []

    def fit(self, X, y=None):
        # Aprende qué columnas superan el límite sobre el conjunto de entrenamiento
        pct_nulos = X.isnull().mean()
        self.cols_to_drop_ = pct_nulos[pct_nulos > self.threshold].index.tolist()
        return self

    def transform(self, X):
        X_copy = X.copy()
        cols = [c for c in self.cols_to_drop_ if c in X_copy.columns]
        return X_copy.drop(columns=cols)


class OutlierCapper(BaseEstimator, TransformerMixin):
    """
    Caps numerical outliers using the Interquartile Range (IQR) method.
    Can be bypassed using apply_capping=False.
    
    Por qué Winsorización y no eliminar filas:
    MSRP tiene outliers extremos (Bugatti: $2.065.902). Eliminar esos registros
    pierde información valiosa. El capping limita el valor al umbral IQR pero
    mantiene el registro completo en el dataset.
    """
    def __init__(self, apply_capping=True):
        self.apply_capping = apply_capping
        self.bounds_ = {}

    def fit(self, X, y=None):
        if not self.apply_capping:
            return self
        # Calcula y guarda los límites IQR para cada columna numérica
        for col in X.select_dtypes(include=['number']).columns:
            Q1 = X[col].quantile(0.25)
            Q3 = X[col].quantile(0.75)
            IQR = Q3 - Q1
            self.bounds_[col] = (Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)
        return self

    def transform(self, X):
        X_copy = X.copy()
        if not self.apply_capping:
            return X_copy
        # np.clip recorta al rango [lower, upper] sin eliminar filas
        for col, (lower, upper) in self.bounds_.items():
            if col in X_copy.columns:
                X_copy[col] = np.clip(X_copy[col], lower, upper)
        return X_copy

    def get_feature_names_out(self, input_features=None):
        return input_features


class DropZeroVarianceTransformer(BaseEstimator, TransformerMixin):
    """
    Drops numerical columns that have zero variance (constant values).
    
    Una columna constante no aporta información predictiva y causa errores
    en StandardScaler (división por desviación estándar = 0).
    """
    def __init__(self):
        self.cols_to_drop_ = []

    def fit(self, X, y=None):
        # Buscamos columnas numéricas cuya desviación estándar sea exactamente 0
        num_cols = X.select_dtypes(include=['number']).columns
        self.cols_to_drop_ = [col for col in num_cols if X[col].std() == 0]
        return self

    def transform(self, X):
        X_copy = X.copy()
        cols = [c for c in self.cols_to_drop_ if c in X_copy.columns]
        return X_copy.drop(columns=cols)

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return None
        # Devuelve solo las columnas que NO fueron eliminadas
        return np.array([f for f in input_features if f not in self.cols_to_drop_])


class SmartImputerTransformer(BaseEstimator, TransformerMixin):
    """
    Decides the imputation strategy based on the percentage of missing values:
    - < 10%: Simple imputation (Median for numeric / Mode for categorical)
    - 10% - threshold: Complex imputation (fallback to simple, future: KNNImputer)
    - > threshold: Ignored (handled by DropHighMissingTransformer)
    
    Por qué mediana y no media: la mediana es robusta a outliers.
    Engine HP tiene un máximo de 1001 HP (Bugatti); la media queda inflada.
    La mediana de 227 HP representa mejor el valor central real del mercado.
    """
    def __init__(self, low_threshold=0.10):
        self.low_threshold = low_threshold
        self.cols_simples_ = []
        self.cols_complejas_ = []

    def fit(self, X, y=None):
        porcentaje_nulos = X.isnull().mean()

        for col in X.columns:
            pct = porcentaje_nulos[col]
            if 0 < pct <= self.low_threshold:
                self.cols_simples_.append(col)
            elif pct > self.low_threshold:
                self.cols_complejas_.append(col)

        print(f"🧠 SmartImputer - Simple (<10%): {self.cols_simples_}")
        print(f"🚧 SmartImputer - Complex (>10%): {self.cols_complejas_} (PENDIENTE)")
        return self

    def transform(self, X):
        X_copy = X.copy()

        # Imputación simple: mediana para números, moda para texto
        for col in self.cols_simples_:
            if pd.api.types.is_numeric_dtype(X_copy[col]):
                X_copy[col] = X_copy[col].fillna(X_copy[col].median())
            else:
                X_copy[col] = X_copy[col].fillna(X_copy[col].mode()[0])

        # Imputación compleja (fallback temporal a simple)
        # MEJORA FUTURA: reemplazar con KNNImputer
        for col in self.cols_complejas_:
            if pd.api.types.is_numeric_dtype(X_copy[col]):
                X_copy[col] = X_copy[col].fillna(X_copy[col].median())
            else:
                if not X_copy[col].mode().empty:
                    X_copy[col] = X_copy[col].fillna(X_copy[col].mode()[0])

        return X_copy

    def get_feature_names_out(self, input_features=None):
        return input_features
