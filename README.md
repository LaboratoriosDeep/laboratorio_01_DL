# Laboratorio 01 — Predicción Multiclase del Deterioro Cognitivo

**Asignatura:** Deep Learning  
**Profesor:** Dr. Juan Bekios Calfa  
**Fecha de entrega:** 15 de marzo de 2026

---

## Descripción

Problema de clasificación multiclase supervisada para predecir el nivel de deterioro
cognitivo (escala GDS) a partir de 15 respuestas binarias del test ACE-III.

Se implementan y comparan tres estrategias de ensamble — **Bagging**, **Boosting** y
**Stacking** — bajo un protocolo de **validación Leave-One-Out (LOOCV)**, justificado
por el tamaño reducido del dataset (n = 22 pacientes).

---

## Estructura del proyecto

```
laboratorio_01/
├── data/
│   ├── raw/                  # Datos originales (.sav)
│   ├── processed/            # Datos preprocesados (si se guardan)
│   └── outputs/              # Salidas del experimento
├── notebooks/
│   └── exploracion.ipynb     # EDA interactivo
├── src/
│   ├── config.py             # Parámetros centrales del experimento
│   ├── data_loader.py        # Carga del archivo SPSS (.sav)
│   ├── preprocessing.py      # Selección de características y escalamiento
│   ├── balancing.py          # Tratamiento del desbalance de clases
│   ├── bagging_model.py      # Modelo Bagging
│   ├── boosting_model.py     # Modelo Boosting (AdaBoost)
│   ├── stacking_model.py     # Modelo Stacking
│   ├── evaluation.py         # Métricas y comparación
│   ├── visualization.py      # Gráficos e informes
│   └── main.py               # Orquestación del experimento completo
├── reports/
│   ├── figuras/              # PNG generados automáticamente
│   └── tablas/               # CSV con métricas comparativas
├── requirements.txt
└── README.md
```

---

## Dataset

| Campo | Valor |
|-------|-------|
| Archivo | `15_atributos_R0-R5.sav` |
| Instancias | 22 pacientes |
| Atributos predictores | 15 (binarios: 0 = error, 1 = acierto) |
| Variable objetivo | `GDS` (3 clases: 1, 2, 3) |
| Distribución | GDS=1: 10, GDS=2: 9, GDS=3: 3 |

Los atributos corresponden a ítems de orientación (DIA, MES, ANO, ESTACION, PAIS,
CIUDAD, CALLELUG, NUMEROPI) y memoria (MIGUEL2, GONZALEZ, AVENIDA2, IMPERIAL,
A682, CALDERA2, COPIAPO2) del ACE-III.

---

## Instalación

```bash
# 1. Crear entorno virtual con Conda
conda create -n lab01_dl python=3.10 -y
conda activate lab01_dl

# 2. Instalar dependencias
pip install -r requirements.txt
```

---

## Ejecución

```bash
cd src/
python main.py
```

Los resultados (figuras y tabla comparativa) se guardan en `reports/`.

---

## Configuración

Todos los parámetros se centralizan en `src/config.py`:

| Parámetro | Valor por defecto | Descripción |
|-----------|-------------------|-------------|
| `TARGET_COL` | `"GDS"` | Variable objetivo |
| `CV_STRATEGY` | `"loocv"` | Protocolo de validación |
| `IMBALANCE_STRATEGY` | `"class_weight"` | Tratamiento del desbalance |
| `FEATURE_SELECTION` | `"all"` | Estrategia de selección (`"all"`, `"variance"`, `"chi2"`) |
| `BAGGING_N_ESTIMATORS` | `50` | Número de estimadores en Bagging |
| `BOOSTING_N_ESTIMATORS` | `50` | Número de estimadores en Boosting |
| `BOOSTING_LEARNING_RATE` | `0.5` | Tasa de aprendizaje de AdaBoost |

---

## Decisiones metodológicas

### ¿Por qué LOOCV?
Con n = 22, un split estándar 80/20 dejaría solo 4 muestras para prueba
(insuficiente). LOOCV usa las 21 muestras restantes para entrenamiento en cada
iteración, maximizando la información disponible y dando una estimación
más estable del rendimiento real.

### ¿Por qué pesos por clase?
La clase GDS = 3 tiene solo 3 muestras. SMOTE requiere al menos
k_neighbors + 1 muestras por clase; con k = 1 generaría solo interpolaciones
sobre 3 puntos, con alto riesgo de sobreajuste. Los pesos por clase (`balanced`)
penalizan más los errores en la clase minoritaria sin generar datos artificiales.

### ¿Por qué métricas macro?
El accuracy favorece a las clases mayoritarias. El promedio macro da igual peso
a cada clase, revelando el comportamiento real en GDS = 3.

### Modelos base en Stacking
- **GaussianNB**: perspectiva probabilística, rápido con pocos datos.
- **DecisionTree**: captura relaciones no lineales; alta varianza → diversidad.
- **KNN (k=3)**: perspectiva por similitud geométrica, complementaria a los anteriores.

La diversidad entre los modelos base es requisito para que el meta-modelo
aporte valor real.

---

## Métricas reportadas

- Accuracy
- Precision macro
- Recall macro  
- F1-score macro ← métrica principal de comparación
- Matriz de confusión por modelo
- Reporte por clase (con énfasis en GDS = 3)

---

## Dependencias

Ver `requirements.txt`. Principales: `scikit-learn`, `pandas`, `numpy`,
`matplotlib`, `seaborn`.
