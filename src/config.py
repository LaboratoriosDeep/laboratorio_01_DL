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

DATA_FILE = os.path.join(DATA_RAW_DIR, "15 atributos R0-R5.sav")

# ── Datos ─────────────────────────────────────────────────────────────────────
# Columnas que identifican al paciente o son etiquetas de clase; se excluyen de X
NON_FEATURE_COLS = ["ID", "GDS", "GDS_R1", "GDS_R2", "GDS_R3", "GDS_R4", "GDS_R5"]

TARGET_COL = "GDS_R2"

# ── Protocolo experimental ────────────────────────────────────────────────────
RANDOM_STATE     = 42

# ── Tratamiento del desbalance ────────────────────────────────────────────────
# "class_weight" → pesos inversos de clase en DT y LR
IMBALANCE_STRATEGY = "class_weight"

# ── Modelos base para Stacking ────────────────────────────────────────────────
STACKING_BASE_ESTIMATORS = ["gnb", "dt", "lr"]
STACKING_META_ESTIMATOR   = "lr"

# ── Parámetros de modelos ─────────────────────────────────────────────────────
BAGGING_N_ESTIMATORS   = 100
BOOSTING_N_ESTIMATORS  = 50
BOOSTING_LEARNING_RATE = 0.3
DT_MAX_DEPTH           = 4
LR_MAX_ITER            = 2000

N_JOBS                 = -1       # -1 usa todos los cores disponibles

# ── Selección de características ──────────────────────────────────────────────
# "all" → usar las 15 características
# "variance" → filtro por varianza
# "chi2"     → filtro chi-cuadrado
FEATURE_SELECTION = "chi2"
VARIANCE_THRESHOLD = 0.02
CHI2_K_BEST        = 12