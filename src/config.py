"""
config.py
---------
Configuración central del experimento.
Todos los parámetros ajustables del laboratorio se definen aquí.
"""

import os

# ── Rutas ─────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW_DIR      = os.path.join(BASE_DIR, "data", "raw")
DATA_PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
DATA_OUTPUTS_DIR   = os.path.join(BASE_DIR, "data", "outputs")
REPORTS_FIG_DIR    = os.path.join(BASE_DIR, "reports", "figuras")
REPORTS_TAB_DIR    = os.path.join(BASE_DIR, "reports", "tablas")

DATA_FILE = os.path.join(DATA_RAW_DIR, "15_atributos_R0-R5.sav")

# ── Datos ─────────────────────────────────────────────────────────────────────
# Columnas que identifican al paciente o son etiquetas de clase; se excluyen de X
NON_FEATURE_COLS = ["ID", "GDS", "GDS_R1", "GDS_R2", "GDS_R3", "GDS_R4", "GDS_R5"]

# Variable objetivo (multiclase con 3 clases: 1, 2, 3)
TARGET_COL = "GDS_R3"

# ── Protocolo experimental ────────────────────────────────────────────────────
# Leave-One-Out: óptimo para datasets muy pequeños (n=22)
CV_STRATEGY      = "loocv"       # "loocv" | "stratified_kfold"
CV_N_FOLDS       = 5             # usado solo si CV_STRATEGY = "stratified_kfold"
RANDOM_STATE     = 42

# ── Tratamiento del desbalance ────────────────────────────────────────────────
# "class_weight"  → pesos inversos de clase (sin generar datos nuevos)
# "smote"         → sobremuestreo sintético (requiere imblearn)
# "none"          → sin tratamiento
IMBALANCE_STRATEGY = "smote"

# ── Modelos base para Stacking ────────────────────────────────────────────────
# Clasificadores que conforman el nivel 0 del stacking
STACKING_BASE_ESTIMATORS = ["gnb", "dt", "lr"]
# Meta-clasificador del nivel 1
STACKING_META_ESTIMATOR   = "lr"

# ── Parámetros de modelos ─────────────────────────────────────────────────────
BAGGING_N_ESTIMATORS   = 50
BOOSTING_N_ESTIMATORS  = 50
BOOSTING_LEARNING_RATE = 0.5
KNN_N_NEIGHBORS        = 3        # mínimo razonable para n=22
DT_MAX_DEPTH           = 3        # árbol poco profundo para evitar sobreajuste
LR_MAX_ITER            = 1000

N_JOBS                 = 6
# ── Selección de características ──────────────────────────────────────────────
# "all" → usar las 15 características
# "variance" → filtro por varianza
# "chi2"     → filtro chi-cuadrado
FEATURE_SELECTION = "chi2"
VARIANCE_THRESHOLD = 0.02
CHI2_K_BEST        = 12
