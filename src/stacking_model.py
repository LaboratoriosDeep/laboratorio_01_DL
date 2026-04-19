"""
stacking_model.py
-----------------
Modelo de ensamble basado en Stacking.

Arquitectura de dos niveles:
  Nivel 0 (modelos base):
    - GaussianNB   → perspectiva probabilística (Bayes)
    - DecisionTree → perspectiva no lineal y local
    - KNN          → perspectiva por similitud

  Nivel 1 (meta-clasificador):
    - LogisticRegression → aprende cómo combinar las predicciones base

Diseño metodológico correcto:
  Para evitar fuga de información, las predicciones del nivel 0 que se usan
  para entrenar el meta-modelo se obtienen mediante cross_val_predict con LOOCV
  sobre los datos de entrenamiento.  Esto garantiza que el meta-modelo aprenda
  de predicciones "out-of-fold" (nunca vio ese dato durante el entrenamiento
  del modelo base).

Justificación de la selección de modelos base:
  - Naive Bayes: simple, probabilístico, funciona bien con binarios y poco datos.
  - Árbol: captura patrones no lineales; su alta varianza beneficia al stacking.
  - KNN: diferente inductive bias (geometría), complementa a los anteriores.
  La diversidad entre modelos base es clave para que el stacking aporte valor.
"""

import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from config import (
    DT_MAX_DEPTH, KNN_N_NEIGHBORS, LR_MAX_ITER,
    RANDOM_STATE, IMBALANCE_STRATEGY
)


def build_base_estimators() -> list:
    """
    Construye los clasificadores base del nivel 0.

    Returns
    -------
    list de (nombre, estimador)
    """
    cw   = "balanced" if IMBALANCE_STRATEGY == "class_weight" else None
    gnb  = GaussianNB()
    dt   = DecisionTreeClassifier(
        max_depth=DT_MAX_DEPTH,
        class_weight=cw,
        random_state=RANDOM_STATE
    )
    lr  = LogisticRegression(
        max_iter=LR_MAX_ITER,
        class_weight=cw,
        random_state=RANDOM_STATE,
        solver="lbfgs"
    )
    return [("gnb", gnb), ("dt", dt), ("lr", lr)]


def build_meta_estimator() -> LogisticRegression:
    """
    Construye el meta-clasificador del nivel 1.

    Returns
    -------
    LogisticRegression
    """
    cw = "balanced" if IMBALANCE_STRATEGY == "class_weight" else None
    return LogisticRegression(
        max_iter=LR_MAX_ITER,
        class_weight=cw,
        random_state=RANDOM_STATE,
        solver="lbfgs"
    )


def build_stacking_model() -> StackingClassifier:
    """
    Construye el StackingClassifier completo.

    Usa cv=LeaveOneOut() internamente para generar predicciones out-of-fold
    durante el entrenamiento del meta-modelo.

    Returns
    -------
    StackingClassifier
    """
    base  = build_base_estimators()
    meta  = build_meta_estimator()
    model = StackingClassifier(
        estimators=base,
        final_estimator=meta,
        cv=5,
        passthrough=False,
        n_jobs= 6
    )
    return model


def train_stacking(X_train: np.ndarray, y_train: np.ndarray) -> StackingClassifier:
    """
    Entrena el StackingClassifier completo.

    Returns
    -------
    model : StackingClassifier ajustado
    """
    model = build_stacking_model()
    model.fit(X_train, y_train)
    return model


def predict_stacking(model: StackingClassifier, X_test: np.ndarray) -> np.ndarray:
    """Genera predicciones finales del StackingClassifier."""
    return model.predict(X_test)


def predict_proba_stacking(model: StackingClassifier,
                           X_test: np.ndarray) -> np.ndarray:
    """Retorna probabilidades del meta-modelo para cada clase."""
    return model.predict_proba(X_test)


# ── Uso directo ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from data_loader import load_data, get_X_y
    from sklearn.model_selection import LeaveOneOut
    from sklearn.metrics import accuracy_score

    df = load_data()
    X, y = get_X_y(df)
    X_arr, y_arr = X.values, y.values

    loo = LeaveOneOut()
    preds = []
    for train_idx, test_idx in loo.split(X_arr):
        model = train_stacking(X_arr[train_idx], y_arr[train_idx])
        pred  = predict_stacking(model, X_arr[test_idx])
        preds.append(pred[0])

    print(f"Stacking LOOCV Accuracy: {accuracy_score(y_arr, preds):.4f}")
