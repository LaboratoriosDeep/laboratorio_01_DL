# Laboratorio 01 - Métodos de Ensamble
## Predicción del Deterioro Cognitivo mediante Machine Learning

### Descripción del Proyecto

Proyecto de Machine Learning enfocado en la predicción del deterioro cognitivo utilizando tres métodos de ensamble: **Bagging**, **Boosting** y **Stacking**. El objetivo es comparar el rendimiento de estos métodos en un dataset pequeño y desbalanceado de evaluaciones cognitivas.

**Variable objetivo**: `GDS_R3` (Global Deterioration Scale Rating 3)  
**Dataset**: 22 instancias con 15 atributos binarios  
**Clases**: 2 clases (1-3 vs 4-7) - Clasificación binaria del deterioro cognitivo

---

### Objetivos

**Objetivo General:**  
Desarrollar y comparar modelos de clasificación para la predicción del deterioro cognitivo utilizando técnicas de ensamble tipo Bagging, Boosting y Stacking.

**Objetivos Específicos:**
- Comprender el problema y la estructura del dataset
- Diseñar un protocolo experimental adecuado para pocas muestras
- Implementar los tres enfoques de ensamble
- Incorporar una estrategia explícita frente al desbalance de clases
- Evaluar con métricas apropiadas e interpretar críticamente los resultados

---

### Estructura del Proyecto

```
laboratorio_01_DL-6/
│
├── data/
│   ├── raw/                          # Datos originales (.sav)
│   ├── processed/                    # Datos procesados
│   └── outputs/                      # Resultados intermedios
│
├── reports/
│   ├── figuras/                      # Gráficos generados
│   └── tablas/                       # Tablas de resultados
│
├── src/                              # Código fuente
│   ├── config.py                     # Configuración central del experimento
│   ├── data_loader.py                # Carga y verificación del dataset
│   ├── preprocessing.py              # Selección de características
│   ├── base_estimators_balancing.py  # Clasificadores base con manejo de desbalance
│   ├── bagging_model.py              # Modelo Bagging
│   ├── boosting_model.py             # Modelo Boosting
│   ├── stacking_model.py             # Modelo Stacking
│   ├── evaluation.py                 # Métricas y protocolo de evaluación
│   ├── visualization.py              # Visualizaciones y exportación
│   └── main.py                       # Orquestación principal
│
├── requirements.txt                  # Dependencias
└── README.md                         # Este archivo
```

---

### Instalación y Configuración

#### 1. Clonar el repositorio
```bash
git clone <url-del-repositorio>
cd laboratorio_01_DL-6
```

#### 2. Crear entorno virtual (recomendado)
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

#### 3. Instalar dependencias
```bash
pip install -r requirements.txt
```

#### 4. Ubicar el dataset
Coloque el archivo `15 atributos R0-R5.sav` en `data/raw/`

---

### Uso

#### Ejecución del experimento completo:
```bash
python src/main.py
```

---

### Configuración

Todos los parámetros del experimento se gestionan desde `src/config.py`:

```python
# Variable objetivo
TARGET_COL = "GDS_R3"

# Protocolo experimental
CV_N_FOLDS = 10          # Stratified K-Fold Cross-Validation
RANDOM_STATE = 42

# Desbalance
IMBALANCE_STRATEGY = "class_weight"   # Pesos inversos por clase

# Selección de características
FEATURE_SELECTION = "chi2"            # chi2 | variance | all
CHI2_K_BEST = 12

# Hiperparámetros de modelos
BAGGING_N_ESTIMATORS = 100
BOOSTING_N_ESTIMATORS = 50
BOOSTING_LEARNING_RATE = 0.3
DT_MAX_DEPTH = 4
```

---

### Resultados

El experimento genera automáticamente:

**Métricas calculadas:**
- Accuracy global
- Precision macro
- Recall macro
- F1-score macro
- AUC-ROC (Stacking)
- Matrices de confusión
- Reportes por clase

**Visualizaciones generadas (`reports/figuras/`):**
- `dist_clases.png` — Distribución de la variable objetivo
- `chi2_features.png` — Importancia de características (Chi²)
- `cm_bagging.png`, `cm_boosting.png`, `cm_stacking.png` — Matrices de confusión
- `comparacion_modelos.png` — Comparación de métricas entre modelos
- `roc_curve_stacking.png` — Curva ROC del modelo Stacking

**Tablas exportadas (`reports/tablas/`):**
- `tabla_comparativa.csv` — Resumen de métricas por modelo

---

### Metodología

#### 1. Protocolo de Validación
- **Stratified 10-Fold Cross-Validation**: Mantiene la proporción de clases en cada fold
- **Predicciones out-of-fold**: Las métricas se calculan sobre las predicciones acumuladas de todos los folds
- **Reproducibilidad**: Random state fijo (42)

#### 2. Manejo del Desbalance
- **Class weights** (`class_weight="balanced"`): Asigna pesos inversamente proporcionales a la frecuencia de cada clase
- Aplicado directamente en DecisionTree y LogisticRegression
- GaussianNB maneja el desbalance mediante priors iguales (`[0.5, 0.5]`)
- No se generan datos sintéticos, evitando riesgo de fuga entre folds

#### 3. Selección de Características
- **Chi-cuadrado**: Mide dependencia estadística entre cada atributo y la variable objetivo
- Selecciona las 12 mejores de 15 totales
- Adecuado para datos binarios/categóricos

#### 4. Modelos de Ensamble

**Bagging:**
- `BaggingClassifier` con `DecisionTreeClassifier` como base
- 100 estimadores con bootstrap (remuestreo con reemplazo)
- Reduce la varianza promediando múltiples modelos entrenados en subconjuntos distintos

**Boosting:**
- `AdaBoostClassifier` con decision stumps (max_depth=1)
- 50 estimadores, learning_rate=0.3
- Aprendizaje secuencial: cada árbol se enfoca en los errores del anterior

**Stacking:**
- Nivel 0: GaussianNB + DecisionTree + LogisticRegression
- Nivel 1: LogisticRegression (meta-modelo)
- CV interno de 5 folds para generar meta-features (independiente del CV externo de 10 folds)

---

### Métricas de Evaluación

**Por qué F1-score macro es la métrica principal:**
- Con desbalance de clases, el accuracy puede ser engañoso
- El promedio macro trata todas las clases por igual, revelando el rendimiento en la clase minoritaria
- Recall en clase minoritaria es especialmente crítico para diagnóstico clínico

---

### Consideraciones Técnicas

#### Desafíos del Dataset:
1. **Tamaño pequeño (n=22)**:
   - Alta varianza en las estimaciones
   - Stratified K-Fold asegura representación de ambas clases en cada fold

2. **Desbalance de clases**:
   - Class weights aplicados en entrenamiento sin generar datos nuevos
   - Priors iguales en GaussianNB como mecanismo equivalente

3. **Atributos binarios**:
   - Chi-cuadrado es el método de selección adecuado para datos categóricos
   - No se requiere normalización
   - Alta interpretabilidad clínica

#### Prevención de Fuga de Datos:
- Los pesos de clase se calculan dentro de cada fold de entrenamiento
- La selección de características se realiza sobre el conjunto completo (sin información de test)
- El Stacking usa CV interno para generar meta-features

---

### Referencias

- Géron, A. *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow*
- Hastie, Tibshirani y Friedman. *The Elements of Statistical Learning*
- Documentación oficial de scikit-learn: [https://scikit-learn.org](https://scikit-learn.org)

---

### Autor

Laboratorio 01 - Deep Learning  
Universidad Católica del Norte  
Dr. Juan Bekios Calfa

---

### Licencia

Este proyecto es material educativo para el curso de Deep Learning.
