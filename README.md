# Laboratorio 01 - Métodos de Ensamble
## Predicción del Deterioro Cognitivo

Comparación de tres métodos de ensamble —**Bagging**, **Boosting** y **Stacking**— sobre un dataset pequeño y desbalanceado de evaluaciones cognitivas.

**Variable objetivo**: `GDS_R2` (Global Deterioration Scale Rating 2)  
**Dataset**: 15 atributos binarios  
**Clases**: 3 clases  

---

### Estructura del Proyecto

```
laboratorio_01_DL-6/
│
├── data/
│   ├── raw/                          # Datos originales (.sav)
│   ├── processed/
│   └── outputs/
│
├── reports/
│   ├── figuras/                      # Gráficos generados
│   └── tablas/                       # Tablas de resultados
│
├── src/
│   ├── config.py                     # Parámetros del experimento
│   ├── data_loader.py                # Carga y verificación del dataset
│   ├── preprocessing.py              # Selección de características
│   ├── base_estimators_balancing.py  # Clasificadores base con manejo de desbalance
│   ├── bagging_model.py              # Modelo Bagging
│   ├── boosting_model.py             # Modelo Boosting
│   ├── stacking_model.py             # Modelo Stacking
│   ├── evaluation.py                 # Métricas y evaluación LOOCV
│   ├── visualization.py              # Visualizaciones y exportación
│   └── main.py                       # Orquestación principal
│
├── requirements.txt
└── README.md
```

---

### Instalación

```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

pip install -r requirements.txt
```

Coloque el archivo `15 atributos R0-R5.sav` en `data/raw/`.

---

### Uso

```bash
python src/main.py
```

---

### Configuración

Parámetros ajustables en `src/config.py`:

```python
TARGET_COL = "GDS_R2"

RANDOM_STATE = 42
IMBALANCE_STRATEGY = "class_weight"

FEATURE_SELECTION = "chi2"
CHI2_K_BEST = 12

BAGGING_N_ESTIMATORS = 100
BOOSTING_N_ESTIMATORS = 50
BOOSTING_LEARNING_RATE = 0.3
DT_MAX_DEPTH = 4
```

---

### Resultados

**Métricas calculadas por modelo:**
- Accuracy, Precision macro, Recall macro, F1-score macro
- Reporte por clase y matriz de confusión

**Archivos generados:**
- `reports/figuras/dist_clases.png` — Distribución de la variable objetivo
- `reports/figuras/chi2_features.png` — Importancia de características (Chi²)
- `reports/figuras/cm_bagging.png`, `cm_boosting.png`, `cm_stacking.png` — Matrices de confusión
- `reports/figuras/comparacion_modelos.png` — Comparación de métricas
- `reports/tablas/tabla_comparativa.csv` — Resumen de métricas por modelo

---

### Metodología

#### Protocolo de Validación
- **Leave-One-Out Cross-Validation (LOOCV)**: En cada iteración se deja una muestra fuera para test y se entrena con el resto. Maximiza el uso de los datos disponibles, relevante dado el tamaño reducido del dataset.

#### Manejo del Desbalance
- **Class weights** (`class_weight="balanced"`): Pesos inversamente proporcionales a la frecuencia de cada clase, aplicados en DecisionTree y LogisticRegression.
- **GaussianNB**: No soporta `class_weight`; usa `priors=None`, por lo que estima los priors directamente desde los datos de entrenamiento de cada fold.
- No se generan datos sintéticos, evitando fuga entre folds.

#### Selección de Características
- **Chi-cuadrado**: Mide dependencia estadística entre cada atributo y la variable objetivo. Adecuado para datos binarios/categóricos. Se seleccionan las 12 mejores de 15.

#### Modelos de Ensamble

**Bagging:**
- `BaggingClassifier` con `DecisionTreeClassifier` como base, 100 estimadores con bootstrap.

**Boosting:**
- `AdaBoostClassifier` con decision stumps (max_depth=1), 50 estimadores, learning_rate=0.3.

**Stacking:**
- Nivel 0: GaussianNB + DecisionTree + LogisticRegression
- Nivel 1: LogisticRegression (meta-modelo)
- CV interno de 5 folds para generar meta-features, independiente del LOOCV externo.

---

### Métricas de Evaluación

Se usa **F1-score macro** como métrica principal porque:
- Con 3 clases desbalanceadas el accuracy puede ser engañoso.
- El promedio macro trata todas las clases por igual, exponiendo el rendimiento en clases minoritarias.

---

### Referencias

- Géron, A. *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow*
- Hastie, Tibshirani y Friedman. *The Elements of Statistical Learning*
- Documentación oficial de scikit-learn: https://scikit-learn.org

---

### Autor

Felipe Parga Meza
Stevens Alday
Diego Parga Meza
Universidad Católica del Norte  
