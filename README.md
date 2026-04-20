# Laboratorio 01 - Métodos de Ensamble
## Predicción del Deterioro Cognitivo mediante Machine Learning

### 📋 Descripción del Proyecto

Proyecto de Machine Learning enfocado en la predicción multiclase del deterioro cognitivo utilizando tres métodos de ensamble: **Bagging**, **Boosting** y **Stacking**. El objetivo es comparar el rendimiento de estos métodos en un dataset pequeño y desbalanceado de evaluaciones cognitivas.

**Variable objetivo**: `GDS_R3` (Global Deterioration Scale Rating 3)  
**Dataset**: 22 instancias con 15 atributos binarios  
**Clases**: 2 clases (1-3 vs 4-7) - Clasificación binaria del deterioro cognitivo

---

### 🎯 Objetivos

**Objetivo General:**
Desarrollar y comparar modelos de clasificación multiclase para la predicción del deterioro cognitivo utilizando técnicas de ensamble tipo Bagging, Boosting y Stacking.

**Objetivos Específicos:**
- Comprender el problema y la estructura del dataset
- Diseñar un protocolo experimental adecuado para pocas muestras
- Implementar los tres enfoques de ensamble
- Incorporar una estrategia explícita frente al desbalance
- Evaluar con métricas apropiadas e interpretar críticamente los resultados

---

### 📂 Estructura del Proyecto

```
lab01-ensambles/
│
├── data/
│   ├── raw/                      # Datos originales (.sav)
│   ├── processed/                # Datos procesados
│   └── outputs/                  # Resultados intermedios
│
├── reports/
│   ├── figuras/                  # Gráficos generados
│   └── tablas/                   # Tablas de resultados
│
├── src/                          # Código fuente
│   ├── config.py                 # Configuración central
│   ├── data_loader.py            # Carga de datos
│   ├── preprocessing.py          # Preprocesamiento y selección
│   ├── balancing.py              # Manejo de desbalance
│   ├── bagging_model.py          # Modelo Bagging
│   ├── boosting_model.py         # Modelo Boosting
│   ├── stacking_model.py         # Modelo Stacking
│   ├── evaluation.py             # Métricas y evaluación
│   ├── visualization.py          # Visualizaciones
│   └── main.py                   # Orquestación principal
│
├── requirements.txt              # Dependencias
└── README.md                     # Este archivo
```

---

### 🔧 Instalación y Configuración

#### 1. Clonar el repositorio (si aplica)
```bash
git clone <url-del-repositorio>
cd lab01-ensambles
```

#### 2. Crear entorno virtual (recomendado)
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# o
venv\Scripts\activate  # Windows
```

#### 3. Instalar dependencias
```bash
pip install -r requirements.txt
```

#### 4. Configurar rutas de datos
Coloque el archivo `15 atributos R0-R5.sav` en `data/raw/`

---

### 🚀 Uso

#### Ejecución del experimento completo:
```bash
python src/main.py
```

#### Ejecución de módulos individuales:
```bash
# Probar carga de datos
python src/data_loader.py

# Probar modelo Bagging
python src/bagging_model.py

# Probar evaluación
python src/evaluation.py
```

---

### ⚙️ Configuración

Todos los parámetros del experimento se gestionan desde `config.py`:

```python
# Variable objetivo
TARGET_COL = "GDS_R3"

# Protocolo experimental
CV_STRATEGY = "loocv"              # Leave-One-Out Cross-Validation
RANDOM_STATE = 42

# Desbalance
IMBALANCE_STRATEGY = "smote"       # smote | class_weight | none

# Selección de características
FEATURE_SELECTION = "chi2"         # chi2 | variance | all
CHI2_K_BEST = 12

# Hiperparámetros de modelos
BAGGING_N_ESTIMATORS = 100
BOOSTING_N_ESTIMATORS = 50
BOOSTING_LEARNING_RATE = 0.3
DT_MAX_DEPTH = 4
```

---

### 📊 Resultados

El experimento genera automáticamente:

**Métricas calculadas:**
- Accuracy global
- Precision macro
- Recall macro
- F1-score macro
- AUC-ROC (para clasificación binaria)
- Matrices de confusión
- Reportes por clase

**Visualizaciones generadas:**
- `dist_clases.png` - Distribución de la variable objetivo
- `cm_*.png` - Matrices de confusión por modelo
- `comparacion_modelos.png` - Comparación de métricas
- `chi2_features.png` - Importancia de características

**Tablas exportadas:**
- `tabla_comparativa.csv` - Resumen de métricas

---

### 🔬 Metodología

#### 1. **Protocolo de Validación**
- **Leave-One-Out Cross-Validation (LOOCV)**: Maximiza el uso de datos con n=22
- **Estratificación**: Mantiene proporciones de clase en cada fold
- **Reproducibilidad**: Random state fijo (42)

#### 2. **Manejo del Desbalance**
- **SMOTE**: Sobremuestreo sintético (k_neighbors=1 para clase minoritaria)
- **Class weights**: Pesos inversamente proporcionales a frecuencia
- Aplicado SOLO en datos de entrenamiento (sin fuga)

#### 3. **Selección de Características**
- **Chi-cuadrado**: Dependencia estadística con variable objetivo
- Selecciona las 12 mejores de 15 totales
- Evaluación de todas las características para el informe

#### 4. **Modelos de Ensamble**

**Bagging:**
- `BaggingClassifier` con `DecisionTree` como base
- 100 estimadores con bootstrap
- Max depth=4 para evitar sobreajuste

**Boosting:**
- `AdaBoostClassifier` con decision stumps
- 50 estimadores, learning rate=0.3
- Enfoque secuencial en errores

**Stacking:**
- Nivel 0: GaussianNB + DecisionTree + LogisticRegression
- Nivel 1: LogisticRegression (meta-modelo)
- CV interno con LOOCV para evitar fuga

---

### 📈 Métricas de Evaluación

**Por qué F1-score macro es la métrica principal:**
- Con desbalance de clases, accuracy puede ser engañoso
- Precision/Recall weighted favorecen clases mayoritarias
- **Macro**: Trata todas las clases por igual → revela rendimiento en minoritarias

**Interpretación de resultados:**
- Diferencias < 5% entre modelos pueden ser varianza muestral
- Recall en clase minoritaria es crítico para diagnóstico
- Matrices de confusión revelan patrones de error específicos

---

### 🧪 Consideraciones Técnicas

#### Desafíos del Dataset:
1. **Tamaño pequeño (n=22)**: 
   - Alta varianza en estimaciones
   - LOOCV maximiza datos de entrenamiento
   - Resultados con intervalos de confianza amplios

2. **Desbalance de clases**:
   - SMOTE genera muestras sintéticas
   - k_neighbors=1 para clases con pocas muestras
   - Class weights como alternativa más conservadora

3. **Atributos binarios**:
   - Chi-cuadrado adecuado para datos categóricos
   - No requiere normalización
   - Interpretabilidad clínica alta

#### Prevención de Fuga de Datos:
- Balanceo SOLO en fold de entrenamiento
- Selección de características antes del loop CV
- Stacking usa CV interno para meta-features

---

### 📚 Referencias

- Géron, A. *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow*
- Hastie, Tibshirani y Friedman. *The Elements of Statistical Learning*
- Documentación oficial de scikit-learn: [https://scikit-learn.org](https://scikit-learn.org)

---

### 👥 Autor

Laboratorio 01 - Deep Learning  
Universidad Católica del Norte  
Dr. Juan Bekios Calfa

---

### 📝 Licencia

Este proyecto es material educativo para el curso de Deep Learning.