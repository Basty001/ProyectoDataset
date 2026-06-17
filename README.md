# Proyecto Modelado – Car Price Prediction
**SCY1101 – Programación para la Ciencia de Datos | Evaluación Parcial N°2**

## Descripción
Pipeline completo de Machine Learning para predecir el precio (MSRP) de automóviles usando el dataset de Kaggle "Car Features and MSRP" (11.914 registros, 16 columnas).

## Estructura del Proyecto
```
ProyectoDataset/
├── data/
│   ├── raw/               # Dataset original (data.csv + metadata.json)
│   └── processed/         # Splits limpios (X_train, X_test, y_train, y_test)
├── notebooks/
│   ├── 01_exploratory_analysis.ipynb
│   ├── 02_supervised_modeling.ipynb
│   ├── 03_model_evaluation.ipynb
│   ├── 04_hyperparameter_optimization.ipynb
│   └── 05_final_analysis.ipynb
├── src/
│   ├── audit.py              # SHA-256 integrity check
│   ├── optimization.py       # Memory optimisation + chunk processing
│   ├── transformers.py       # Custom Scikit-learn transformers
│   ├── pipeline.py           # Full cleaning pipeline
│   ├── preprocessing.py      # Train/Test split (80/20)
│   ├── unsupervised.py       # PCA + KMeans + DBSCAN + Agglomerative
│   ├── model_training.py     # Model definitions + final training
│   ├── model_evaluation.py   # CV, test evaluation, plots
│   └── tune.py               # GridSearchCV + RandomizedSearchCV + Optuna
├── models/trained_models/    # Serialised .pkl models
├── results/
│   ├── metrics/              # CSV metric files
│   ├── plots/                # All visualisations
│   └── reports/              # Price-bin classification reports
├── main.py                   # Full pipeline orchestrator
├── requirements.txt
└── README.md
```

## Pipeline de 7 pasos
| Paso | Módulo | Descripción |
|------|--------|-------------|
| 1 | `audit.py` | Verificación de integridad SHA-256 |
| 2 | `preprocessing.py` | Limpieza + split 80/20 (sin data leakage) |
| 3 | `unsupervised.py` | PCA (95% var) + KMeans / DBSCAN / Agglomerative |
| 4 | `model_evaluation.py` | Cross-validation 5-fold de 7 modelos |
| 5 | `tune.py` | GridSearchCV + RandomizedSearchCV + Optuna (TPE) |
| 6 | `model_training.py` | Entrenamiento final en todo el training set |
| 7 | `model_evaluation.py` | Evaluación única en test set + classification report |

## Instalación
```bash
pip install -r requirements.txt
```

## Uso
```bash
# Pipeline completo
python main.py

# Modo rápido (menos trials Optuna, útil para CI)
python main.py --fast

# Saltarse la verificación de hash
python main.py --skip-audit --fast
```

## Dependencias principales
- Python ≥ 3.10
- scikit-learn ≥ 1.3
- pandas ≥ 2.0
- optuna ≥ 3.0
- numpy, matplotlib, seaborn, scipy, joblib

## Reproducibilidad
Todos los componentes estocásticos usan `random_state=42`. El test set se guarda **antes** de ajustar el pipeline para evitar data leakage y solo se evalúa una vez al final.
