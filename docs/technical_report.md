# Informe Técnico: Preparación de Datos — Car Price Dataset 🚗

**Equipo:** [Nombres de los integrantes]  
**Asignatura:** SCY1101 — Programación para la Ciencia de Datos  
**Fecha:** [Fecha de entrega]  

---

## 1. Resumen Ejecutivo

Este proyecto tiene como objetivo procesar el dataset *Car Features and MSRP* (Kaggle) para preparar los datos para un futuro modelo de Machine Learning de **regresión supervisada**: predecir el precio de venta sugerido (MSRP) de un automóvil a partir de sus características técnicas y de mercado.

Se implementó un flujo de trabajo reproducible que incluye la verificación de integridad SHA-256, optimización de memoria con downcasting, tratamiento de valores nulos reales y enmascarados, Winsorización de outliers, y Pipelines de Scikit-Learn para garantizar reproducibilidad.

**Resultados clave:**
- Dataset de 11.914 registros procesado con **cero pérdida de filas**
- Reducción de memoria del **92.9%** (6.08 MB → 0.43 MB)
- Integridad verificada con firma digital SHA-256
- Matriz final de 11.914 × 93 features lista para modelado

---

## 2. Análisis Exploratorio Inicial (EDA)

*(Ver: outputs/grafico_01_distribucion_msrp.png, outputs/grafico_04_correlacion.png)*

### Resumen del Dataset

| Atributo | Valor |
|---|---|
| Filas | 11.914 |
| Columnas | 16 |
| Variable objetivo | MSRP (precio en USD) |
| Rango MSRP | $2.000 — $2.065.902 |
| Marcas únicas | 48 |
| Modelos únicos | 915 |

### Tabla de Calidad de Datos

| Variable | Tipo | Nulos | % Nulos | Decisión |
|---|---|---|---|---|
| Make | Categórica | 0 | 0% | Mantener |
| Model | Categórica | 0 | 0% | **ELIMINAR** (915 valores únicos) |
| Year | Numérica | 0 | 0% | Mantener |
| Engine Fuel Type | Categórica | 3 | 0.03% | Imputar con moda |
| Engine HP | Numérica | 69 | 0.58% | Imputar con mediana |
| Engine Cylinders | Numérica | 30 | 0.25% | Imputar con mediana |
| Transmission Type | Categórica | 19 'UNKNOWN' | 0.16% | UNKNOWN → NaN → moda |
| Number of Doors | Numérica | 6 | 0.05% | Imputar con mediana |
| Market Category | Categórica | 3.742 | **31.4%** | **ELIMINAR** (supera umbral) |
| highway MPG | Numérica | 0 | 0% | Mantener |
| city mpg | Numérica | 0 | 0% | Mantener |
| Popularity | Numérica | 0 | 0% | Mantener |
| MSRP | Numérica | 0 | 0% | **Variable objetivo** |

### Patrones Identificados

**Desbalance de la variable objetivo:** El MSRP presenta distribución sesgada a la derecha. Mediana ($29.995) y media ($40.595) difieren en más de $10.000, evidenciando outliers extremos como el Bugatti Veyron ($2.065.902). *(Ver: outputs/grafico_01_distribucion_msrp.png)*

**Nulos enmascarados:** `Transmission Type` contiene 19 registros con el string literal `'UNKNOWN'`. Pandas no los detecta con `isnull()`. *(Ver: outputs/grafico_00_mapa_nulos.png)*

**Correlaciones numéricas:** `Engine HP` (r=0.66) y `Engine Cylinders` (r=0.53) son los predictores más fuertes. Multicolinealidad entre `highway MPG` y `city mpg` (r=0.89). *(Ver: outputs/grafico_04_correlacion.png)*

---

## 3. Metodología de Transformación

### 3.1 Eliminación de `Model` (Alta Cardinalidad)

915 valores únicos → OneHotEncoding generaría 915 columnas nuevas → maldición de la dimensionalidad y overfitting. La información del fabricante se conserva en `Make` (48 marcas).

### 3.2 Tratamiento de Valores UNKNOWN Enmascarados

`UnknownToNaNTransformer` convierte los 19 registros `'UNKNOWN'` de Transmission Type a `np.nan` real, permitiendo que `SmartImputerTransformer` los rellene con la moda (`AUTOMATIC`).

### 3.3 Eliminación de `Market Category` (31.4% Nulos)

Supera el umbral del 25% y sus valores son multi-etiqueta (`'Luxury,Performance,High-Performance'`), haciendo inviable cualquier estrategia de encoding estándar.

### 3.4 Winsorización (Capping IQR)

En lugar de eliminar registros con precios extremos, se recortan al límite estadístico IQR. *(Ver: outputs/grafico_02_winsorizacion_msrp.png)*

| Columna | Máximo original | Límite IQR | Registros cappados |
|---|---|---|---|
| MSRP | $2.065.902 | $74.078 | ~500 |
| Engine HP | 1.001 HP | 495 HP | ~30 |

### 3.5 Imputación Inteligente

| Columna | % Nulos | Estrategia | Valor |
|---|---|---|---|
| Engine Fuel Type | 0.03% | Moda | 'regular unleaded' |
| Engine HP | 0.58% | Mediana | 227 HP |
| Engine Cylinders | 0.25% | Mediana | 6 cilindros |
| Transmission Type | 0.16% | Moda | 'AUTOMATIC' |
| Number of Doors | 0.05% | Mediana | 4 puertas |

**¿Por qué mediana y no media?** La mediana es robusta a outliers. Para Engine HP, la media queda inflada por los 1.001 HP del Bugatti; la mediana de 227 HP es más representativa.

---

## 4. Resultados y Validación Técnica

**Integridad (Checksum):** El archivo original fue validado exitosamente generando su firma SHA-256 (`26e39d3e902246d01a93ae390f51129a288079aefad2cb3292751a262ffd62d8`), confirmando la ausencia de corrupción de datos.

**Optimización de Memoria:** Downcasting numérico redujo el peso en memoria un **92.9%** (6.08 MB → 0.43 MB).

| Tipo original | Tipo optimizado | Ahorro |
|---|---|---|
| int64 | int16 (Year, MPG) | 75% por columna |
| int64 | int32 (MSRP) | 50% por columna |
| float64 | float32 (Engine HP) | 50% por columna |

**Pipelines de Scikit-Learn:** Pipeline de 5 pasos secuenciales. *(Ver diagrama: notebooks/02_Pipelines.ipynb)*

| Métrica | Valor |
|---|---|
| Filas originales | 11.914 |
| Filas finales | 11.914 (sin pérdida) |
| Columnas originales | 16 |
| Features para ML | 93 |
| Nulos en resultado | 0 |

---

## 5. Conclusiones y Recomendaciones

**Conclusiones:** Se conservaron los 11.914 registros originales. El hallazgo más relevante fue el nulo enmascarado en `Transmission Type`: el string `'UNKNOWN'` habría pasado desapercibido con solo usar `isnull()`, contaminando el modelo con una categoría semánticamente inválida.

**Lecciones Aprendidas:** El uso de entornos virtuales (`.venv`) es fundamental para reproducibilidad. La separación `fit()` / `transform()` garantiza que ninguna información del test contamine el entrenamiento, previniendo Data Leakage.

**Mejoras Futuras:**
- `KNNImputer` para `Engine HP` (usa similitud entre vehículos)
- Transformación logarítmica al MSRP antes del modelado
- Target Encoding para `Make` (reduce dimensionalidad vs OneHotEncoding)
