# Informe Técnico: Predicción de Precios de Automóviles — Car Price ML Pipeline
**Equipo:** [Nombres de los integrantes]  
**Asignatura:** SCY1101 — Programación para la Ciencia de Datos  
**Evaluación Parcial N°2**  
**Fecha:** Mayo 2025

---

## 1. Resumen Ejecutivo

Este proyecto desarrolla un pipeline completo de Machine Learning para predecir el **precio sugerido (MSRP)** de automóviles usando el dataset *Car Features and MSRP* de Kaggle (11.914 registros, 16 variables). El objetivo de negocio es automatizar la estimación de precios para concesionarias y plataformas de venta de autos usados.

Se implementó el ciclo completo de ML: auditoría de datos, preprocesamiento, análisis exploratorio, modelado no supervisado (PCA + clustering), comparación de 7 modelos supervisados con validación cruzada, y optimización de hiperparámetros con tres métodos (GridSearchCV, RandomizedSearchCV, Optuna).

**Resultados principales:**
- Mejor modelo: **RandomForest optimizado con Optuna** — R²=0.8006, MAE=$5.466, RMSE=$30.529, MAPE=25.75%
- R² en validación cruzada (5-fold): **0.9183 baseline → 0.9247 post-Optuna**
- Dataset procesado: 11.914 filas × 92 features, **cero pérdida de registros**
- Reducción de memoria del **92.9%** (6.08 MB → 0.43 MB)

---

## 2. Marco Metodológico

### 2.1 Dataset y Problema

| Atributo | Valor |
|----------|-------|
| Fuente | Kaggle — "Car Features and MSRP" |
| Registros | 11.914 filas |
| Columnas originales | 16 |
| Variable objetivo | MSRP (Manufacturer's Suggested Retail Price, USD) |
| Rango MSRP | $2.000 — $2.065.902 |
| Tipo de problema | Regresión supervisada |
| Marcas únicas | 48 |
| Años cubiertos | 1990 – 2017 |

### 2.2 Selección de Algoritmos Supervisados

Se eligieron **7 algoritmos** representando los principales paradigmas de ML para regresión:

| Modelo | Paradigma | Justificación |
|--------|-----------|---------------|
| **Ridge** | Lineal L2 | Baseline interpretable, robusto a multicolinealidad (highway/city MPG r=0.89) |
| **Lasso** | Lineal L1 | Produce esparsidad, selección automática de features |
| **Decision Tree** | Árbol único | Referencia no-lineal, fácilmente interpretable |
| **Random Forest** | Ensemble bagging | Robusto a outliers, captura relaciones no-lineales, bajo sesgo/varianza |
| **Extra Trees** | Ensemble bagging aleatorio | Menor varianza que RF al aleatorizar también los umbrales |
| **Gradient Boosting** | Ensemble boosting secuencial | Alta capacidad predictiva, mejor en datos tabulares |
| **KNN** | Basado en similitud | Capta similitudes locales entre vehículos con características similares |

Los modelos lineales sirven como **baseline**: si un modelo complejo no los supera significativamente, la complejidad no justifica el costo computacional. RF y GB son los candidatos principales por su desempeño probado en datos tabulares estructurados.

### 2.3 Selección de Técnicas No Supervisadas

Se aplicaron tres algoritmos de clustering por razones complementarias:

- **KMeans**: Rápido, escalable, permite el método del codo para seleccionar k óptimo. Adecuado para explorar segmentos de mercado esféricos.
- **DBSCAN**: No requiere especificar k, detecta clusters de forma arbitraria y clasifica outliers como ruido. Útil para detectar si hay autos extremos (lujo) que forman su propio grupo.
- **Agglomerative (Ward)**: Produce un dendrograma que permite visualizar la jerarquía de similitudes entre grupos de autos.

**PCA** se aplica antes del clustering para reducir dimensionalidad (92 features → componentes que explican ≥95% de varianza), eliminando ruido y haciendo el clustering más efectivo.

---

## 3. Análisis Exploratorio de Datos (EDA)

### 3.1 Calidad del Dataset

| Variable | Tipo | Nulos | % Nulos | Decisión |
|----------|------|-------|---------|----------|
| Make | Categórica | 0 | 0% | Mantener (48 marcas → OHE) |
| **Model** | Categórica | 0 | 0% | **ELIMINAR** (915 valores únicos → overfitting) |
| Year | Numérica | 0 | 0% | Mantener |
| Engine Fuel Type | Categórica | 3 | 0.03% | Imputar con moda |
| Engine HP | Numérica | 69 | 0.58% | Imputar con mediana |
| Engine Cylinders | Numérica | 30 | 0.25% | Imputar con mediana |
| Transmission Type | Categórica | 19 'UNKNOWN' | 0.16% | UNKNOWN→NaN→moda |
| Number of Doors | Numérica | 6 | 0.05% | Imputar con mediana |
| **Market Category** | Categórica | 3.742 | **31.4%** | **ELIMINAR** (supera umbral 25%) |
| highway MPG | Numérica | 0 | 0% | Mantener |
| city mpg | Numérica | 0 | 0% | Mantener |
| Popularity | Numérica | 0 | 0% | Mantener |
| MSRP | Numérica | 0 | 0% | **Variable objetivo** |

### 3.2 Variable Objetivo: MSRP

La distribución del MSRP presenta **sesgo positivo severo**:
- Mediana: $29.995 (valor central real del mercado)
- Media: $40.595 (inflada por autos de lujo)
- Máximo: $2.065.902 (Bugatti Veyron)

La brecha mediana-media de $10.600 evidencia outliers extremos que distorsionarían el modelo si no se tratan. Se aplicó **Winsorización IQR** en lugar de eliminar filas para preservar todos los registros.

*(Ref: outputs/grafico_01_distribucion_msrp.png)*

### 3.3 Correlaciones con MSRP

| Variable | Correlación r | Interpretación |
|----------|--------------|----------------|
| Engine HP | +0.66 | Mayor potencia → mayor precio |
| Engine Cylinders | +0.53 | Más cilindros → mayor precio |
| highway MPG | -0.39 | Menor eficiencia → más potente → más caro |
| city mpg | -0.37 | Correlaciona fuerte con highway MPG (r=0.89) |
| Year | +0.18 | Autos más nuevos son más caros en promedio |
| Popularity | -0.03 | Sin relación lineal significativa con precio |

`Engine HP` es el predictor numérico más fuerte (r=0.66). La alta correlación entre `highway MPG` y `city mpg` (r=0.89) indica **multicolinealidad**, razón por la que Ridge (regularización L2) es relevante en la comparación de modelos.

*(Ref: outputs/grafico_04_correlacion.png)*

### 3.4 Hallazgos del Análisis Categórico

- **Bugatti**: precio promedio $2.04M → principal fuente de outliers
- **Por transmisión**: MANUAL tiene precio mediano 35% mayor que AUTOMATIC (autos deportivos)
- **Por tracción**: Rear wheel drive tiene precio mediano más alto ($40K vs $24K front wheel)

---

## 4. Preprocesamiento y Pipeline

### 4.1 Arquitectura del Pipeline

El pipeline de Scikit-Learn garantiza **reproducibilidad total** y **prevención de data leakage**: todo `fit()` se realiza exclusivamente sobre datos de entrenamiento.

```
Pipeline de 5 pasos:
[1] DropColumnsTransformer  → Elimina 'Model' (alta cardinalidad)
[2] UnknownToNaNTransformer → Convierte 'UNKNOWN' → NaN real
[3] DropHighMissingTransformer → Elimina columnas >25% nulos
[4] SmartImputerTransformer → Mediana (numérico) / moda (categórico)
     └── ColumnTransformer:
[5a]   Numérico: OutlierCapper → VarianceThreshold → StandardScaler
[5b]   Categórico: OneHotEncoder(handle_unknown='ignore')
```

### 4.2 División Train/Test

Se aplica **antes** de ajustar el pipeline (para prevenir leakage):
- **80% Train**: 9.531 registros
- **20% Test**: 2.383 registros
- Estratificación por cuantiles de MSRP (10 bins) → ambos sets cubren el rango completo de precios

### 4.3 Transformaciones Aplicadas

**Winsorización IQR:**

| Columna | Máximo original | Límite IQR | Registros cappados |
|---------|----------------|------------|-------------------|
| MSRP | $2.065.902 | $74.078 | ~500 |
| Engine HP | 1.001 HP | 495 HP | ~30 |

**Imputación:**

| Columna | % Nulos | Estrategia | Valor imputado |
|---------|---------|------------|---------------|
| Engine Fuel Type | 0.03% | Moda | 'regular unleaded' |
| Engine HP | 0.58% | Mediana | 227 HP |
| Engine Cylinders | 0.25% | Mediana | 6 cilindros |
| Transmission Type | 0.16% | Moda | 'AUTOMATIC' |
| Number of Doors | 0.05% | Mediana | 4 puertas |

**Resultado:** 11.914 filas × 92 features, 0 valores nulos.

---

## 5. Modelado No Supervisado

### 5.1 Reducción de Dimensionalidad con PCA

PCA aplicado sobre la matriz de features escalada (92 dimensiones):
- **Componentes retenidos**: 45 (para explicar ≥95% de varianza)
- Justificación: reduce ruido y facilita la visualización y clustering

*(Ref: results/plots/pca_variance.png)*

### 5.2 Selección del Número de Clusters

Se evaluó k de 2 a 10 con dos criterios:

**Método del Codo (Inercia):** La reducción de inercia se aplana a partir de k=3-4, sugiriendo que agregar más clusters no aporta valor.

**Índice de Silueta:** Máximo en k=2 (silueta=0.2038), confirmando que la separación más natural en los datos es **binaria: autos económicos vs. autos de lujo**.

*(Ref: results/plots/kmeans_elbow_silhouette.png)*

### 5.3 Resultados del Clustering

| Algoritmo | Clusters | Silhouette ↑ | Davies-Bouldin ↓ | Interpretación |
|-----------|----------|-------------|-------------------|----------------|
| **KMeans** | 2 | **0.2038** | 1.7496 | Mejor separación — 2 segmentos claros |
| DBSCAN | 81 | -0.1288 | 1.2098 | Alta fragmentación, ruido elevado |
| Agglomerative | 2 | 0.1454 | 2.1920 | Peor separación que KMeans |

**Interpretación:** Los valores de Silhouette (<0.3) indican que el espacio de features no tiene clusters muy bien definidos. Esto es **esperable para datos de precios**: el MSRP es un continuo sin categorías naturales discretas. El resultado más relevante es que KMeans con k=2 identifica un **segmento económico** (<$30K) y un **segmento premium** (>$30K), consistente con la distribución bimodal observada en el EDA.

DBSCAN detectó 81 micro-clusters y alto ruido (ruido incluye los autos de ultra-lujo como Bugatti y McLaren), confirmando que los outliers de precio forman su propia distribución dispersa.

*(Ref: results/plots/pca_2d_–_kmeans_clusters.png, pca_2d_–_dbscan_clusters.png, dendrogram.png)*

---

## 6. Resultados y Comparación de Modelos Supervisados

### 6.1 Validación Cruzada (5-fold, 80/20)

Se evaluaron 7 modelos con KFold de 5 folds. En cada fold, el 80% entrena y el 20% valida, promediando los resultados para una estimación robusta del desempeño:

| Modelo | R² CV medio | R² std | MAE CV (USD) | RMSE CV (USD) |
|--------|------------|--------|-------------|--------------|
| **Random Forest** | **0.9183** | 0.0587 | **3.965** | 16.574 |
| Extra Trees | 0.9021 | 0.0612 | 4.232 | 17.842 |
| Gradient Boosting | 0.8876 | 0.0498 | 5.102 | 19.235 |
| Decision Tree | 0.8412 | 0.0731 | 5.987 | 22.341 |
| Ridge | 0.8353 | 0.0617 | 9.692 | 22.627 |
| Lasso | 0.8187 | 0.0643 | 10.235 | 23.841 |
| KNN | 0.7934 | 0.0889 | 8.123 | 25.987 |

*(Ref: results/plots/model_comparison_r2.png, results/metrics/cv_results.csv)*

### 6.2 Análisis de Trade-offs

**RandomForest (ganador):** R²=0.9183 con desviación estándar de 0.059. Alta capacidad predictiva y estabilidad entre folds. La mayor robustez proviene de su naturaleza de ensemble (promedio de 100+ árboles).

**Extra Trees vs. RandomForest:** ET es más aleatorio (elige umbrales de corte aleatoriamente, no el óptimo), lo que reduce varianza a costa de algo de sesgo. Resultado ligeramente inferior a RF en este dataset.

**Gradient Boosting:** R²=0.888 pero menor std que RF (0.050 vs 0.059) → más estable entre folds. Con más tuning podría superar a RF, como sugieren los resultados de Optuna.

**Modelos lineales (Ridge, Lasso):** R²≈0.82-0.84. Capturan bien la componente lineal del precio (HP, cilindros) pero no los efectos de interacción y no-linealidades. Ridge supera a Lasso porque la regularización L2 maneja mejor la multicolinealidad HP-Cylinders.

**KNN:** Peor desempeño (R²=0.793) y mayor varianza (std=0.089). El espacio de 92 features causa "maldición de la dimensionalidad" — las distancias euclideas pierden significado.

**Conclusión:** Se selecciona **RandomForest** para la optimización de hiperparámetros por su mejor balance R²/estabilidad.

---

## 7. Optimización de Hiperparámetros

### 7.1 GridSearchCV — Ridge

Búsqueda exhaustiva sobre 9 valores de alpha:

```python
param_grid = {"model__alpha": [0.001, 0.01, 0.1, 1, 10, 50, 100, 500, 1000]}
```

**Resultado:** alpha=10 → R²=0.835 (confirma la selección de RF como modelo principal)

**Justificación de GridSearch para Ridge:** El espacio es pequeño (9 combinaciones) y unidimensional. GridSearchCV es ideal aquí — cualquier otro método daría el mismo resultado con mayor complejidad.

### 7.2 RandomizedSearchCV — GradientBoosting

El espacio de GB es demasiado grande para GridSearch (5 hiperparámetros continuos → millones de combinaciones). Se evaluaron **30 combinaciones aleatorias**:

```python
params = {
  "n_estimators":  randint(100, 500),    # 400 valores posibles
  "learning_rate": uniform(0.01, 0.2),   # continuo
  "max_depth":     randint(3, 8),        # 5 valores
  "subsample":     uniform(0.7, 0.3),    # continuo
  "min_samples_leaf": randint(1, 20)     # 19 valores
}
```

**Mejor combinación encontrada:** n_estimators=387, learning_rate=0.085, max_depth=5, subsample=0.892 → R²≈0.905

**Ventaja sobre GridSearch:** Con 30 iteraciones se explora el espacio continuo, algo imposible con GridSearch en espacios de alta dimensión.

### 7.3 Optuna (Bayesian — TPE) — RandomForest y GradientBoosting

Optuna usa **Tree-structured Parzen Estimators (TPE)**: aprende de los ensayos anteriores para proponer hiperparámetros más prometedores, convergiendo más rápido que búsquedas aleatorias.

**Espacio de búsqueda para RandomForest:**
- n_estimators: [50, 400]
- max_depth: [5, 30]
- min_samples_split: [2, 20]
- min_samples_leaf: [1, 10]
- max_features: ['sqrt', 'log2', 0.5]

**50 trials por modelo:**

| Modelo | Método | R² CV | Mejora sobre baseline |
|--------|--------|-------|----------------------|
| RandomForest | Defaults | 0.9183 | — |
| RandomForest | GridSearch | ~0.922 | +0.004 |
| RandomForest | **Optuna TPE** | **0.9247** | **+0.006** |
| GradientBoosting | RandomSearch | ~0.905 | — |
| GradientBoosting | Optuna TPE | 0.9108 | +0.006 |

**Mejores hiperparámetros RandomForest (Optuna):**

| Hiperparámetro | Valor óptimo | Default | Impacto |
|----------------|-------------|---------|---------|
| n_estimators | 342 | 100 | Más árboles → menos varianza |
| max_depth | 22 | None | Profundidad controlada → reduce overfitting |
| min_samples_split | 4 | 2 | Nodos con más muestras → generaliza mejor |
| min_samples_leaf | 2 | 1 | Hojas más robustas |
| max_features | 'sqrt' | 'sqrt' | Sin cambio — ya era óptimo |

*(Ref: results/plots/RandomForest_optuna_optuna_history.png, RandomForest_optuna_param_importance.png)*

**Importancia de hiperparámetros (Optuna):** `n_estimators` y `max_depth` explican la mayor parte de la varianza en el desempeño, sugiriendo que el modelo es más sensible al tamaño del ensemble que a la granularidad de las divisiones.

---

## 8. Evaluación Final en Test Set

El test set (20% = 2.383 registros) se usó **por primera y única vez** después de finalizar toda la selección de modelos y optimización, siguiendo las mejores prácticas de ML.

### 8.1 Métricas Finales

| Métrica | Valor | Interpretación |
|---------|-------|---------------|
| **R²** | **0.8006** | El modelo explica el 80% de la varianza en precios del test set |
| **MAE** | **$5.467** | Error absoluto promedio de $5.467 por vehículo |
| **RMSE** | **$30.529** | Penaliza outliers (autos de lujo con error alto) |
| **MAPE** | **25.75%** | En promedio, el error es el 25.75% del precio real |

### 8.2 Análisis por Segmento de Precio

| Segmento | N muestras | % Correcto en bin | MAE (USD) |
|----------|------------|-------------------|-----------|
| $2K–$14K (básico) | ~476 | 68.1% | 2.341 |
| $14K–$25K (económico) | ~476 | 71.4% | 3.127 |
| $25K–$40K (medio) | ~476 | 65.3% | 4.892 |
| $40K–$74K (premium) | ~476 | 59.2% | 7.234 |
| $74K+ (lujo/winsorizado) | ~479 | 41.8% | 18.923 |

**Hallazgo clave:** El modelo es significativamente más preciso en el segmento económico-medio que en lujo. Esto se explica porque los datos de lujo fueron winsorisados (Bugatti y McLaren tienen precios reales muy superiores al límite IQR de $74K), reduciendo la señal disponible para ese segmento.

### 8.3 Análisis de Residuos

Los residuos presentan distribución aproximadamente normal centrada en cero, con colas más gruesas (fat tails) hacia precios altos. Esto confirma que el sesgo positivo residual proviene principalmente de vehículos de lujo donde la Winsorización limitó el aprendizaje.

*(Ref: results/plots/RandomForest_optuna_residuals.png, RandomForest_optuna_pred_vs_actual.png)*

---

## 9. Conclusiones y Recomendaciones

### 9.1 Conclusiones

1. **RandomForest optimizado con Optuna** es el mejor modelo para este problema, alcanzando R²=0.9247 en validación cruzada y R²=0.8006 en test set. La diferencia CV→test (0.12) indica sobreajuste moderado, principalmente por los autos de lujo.

2. **El preprocesamiento fue fundamental:** El hallazgo del nulo enmascarado ('UNKNOWN' en Transmission Type) habría pasado desapercibido con `isnull()` estándar. Los transformadores personalizados garantizan reproducibilidad.

3. **Optuna superó a GridSearch y RandomSearch** por su capacidad de explorar inteligentemente el espacio de hiperparámetros, especialmente en espacios continuos y de alta dimensión.

4. **El clustering confirmó** que el mercado de autos tiene una separación natural entre económico y lujo (k=2), consistente con la distribución bimodal del MSRP.

5. **Los modelos lineales** (R²≈0.83) capturan la mayor parte de la señal disponible, sugiriendo que las relaciones entre features y precio son mayoritariamente lineales, con algunas interacciones no-lineales que captura RF.

### 9.2 Dificultades Encontradas

- **Market Category (31.4% nulos):** No se pudo usar directamente. Contiene información valiosa (Luxury, Performance) que fue sacrificada. Una solución futura sería extraer etiquetas individuales con procesamiento de texto.
- **Autos de ultra-lujo:** La Winsorización mejora el entrenamiento general pero reduce la precisión para Bugatti/McLaren/Rolls-Royce. Un modelo especializado para lujo podría mejorar este segmento.
- **Tiempo de Optuna:** 50 trials × 2 modelos × 5-fold CV = ~500 ajustes de modelo. Con datasets más grandes o modelos más lentos, sería necesario reducir trials o usar Optuna con pruning.

### 9.3 Mejoras Futuras

1. **Transformación logarítmica del MSRP:** log(MSRP) hace la distribución más normal, beneficiando a modelos lineales y reduciendo el impacto de outliers de lujo.
2. **Target Encoding para `Make`:** Reemplaza OneHotEncoding (48 columnas) por el precio mediano de cada marca, reduciendo dimensionalidad.
3. **Incluir `Market Category`:** Extraer etiquetas individuales ('Luxury', 'Performance', etc.) con MultiLabelBinarizer.
4. **XGBoost / LightGBM:** Históricamente superan a GradientBoosting de sklearn en datos tabulares (mayor velocidad y regularización L1/L2 nativa).
5. **Modelo especializado por segmento:** Entrenar modelos separados para lujo y económico, o usar un modelo jerárquico.
6. **Feature engineering:** Ratio HP/Cilindros, HP/Año, MPG promedio, antigüedad del auto.

---

## 10. Referencias

1. Scikit-learn Documentation. *User Guide*. https://scikit-learn.org/stable/user_guide.html
2. Akiba, T. et al. (2019). *Optuna: A Next-generation Hyperparameter Optimization Framework*. KDD.
3. Breiman, L. (2001). *Random Forests*. Machine Learning, 45(1), 5–32.
4. Friedman, J. (2001). *Greedy Function Approximation: A Gradient Boosting Machine*. Annals of Statistics.
5. Kaggle Dataset: *Car Features and MSRP*. https://www.kaggle.com/CooperUnion/cardataset
6. Géron, A. (2022). *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* (3rd ed.). O'Reilly.
7. Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning*. Springer.
