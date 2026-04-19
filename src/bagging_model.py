"""
bagging_model.py
----------------
Modelo de ensamble basado en Bagging.

Estrategia: BaggingClassifier con árbol de decisión como clasificador base.
  - Reduce la varianza entrenando múltiples modelos sobre subconjuntos
    remuestreados con reemplazo (bootstrap).
  - El árbol de decisión se eligió como base porque tiene alta varianza
    (sensible a pequeñas perturbaciones en datos), siendo el tipo de modelo
    que más se beneficia del bagging.
  - La profundidad máxima se limita (DT_MAX_DEPTH=3) para evitar árboles que
    memoricen el conjunto de entrenamiento con tan pocos datos.

Justificación con este dataset:
  Con n=22 y clases desbalanceadas, el bagging aprovecha el remuestreo para
  generar diversidad interna y produce estimaciones más estables que un único
  árbol.
"""

import numpy as np
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from config import (
    BAGGING_N_ESTIMATORS, DT_MAX_DEPTH, RANDOM_STATE,
    IMBALANCE_STRATEGY
)


def build_bagging_model() -> BaggingClassifier:
    """
    Construye el clasificador Bagging con los hiperparámetros del config.

    Se usa class_weight='balanced' como string (no dict) en el árbol base para
    que sklearn lo recalcule dinámicamente en cada subconjunto bootstrap, evitando
    el error "class not in class_weight" cuando un fold no contiene todas las clases.

    Returns
    -------
    BaggingClassifier listo para entrenar.
    """
    # "balanced" como string: sklearn recalcula los pesos en cada bootstrap sample
    cw = "balanced" if IMBALANCE_STRATEGY == "class_weight" else None
    base_estimator = DecisionTreeClassifier(
        max_depth=DT_MAX_DEPTH,
        class_weight=cw,
        random_state=RANDOM_STATE
    )
    model = BaggingClassifier(
        estimator=base_estimator,
        n_estimators=BAGGING_N_ESTIMATORS,
        bootstrap=True,
        random_state=RANDOM_STATE,
        n_jobs= 1   # n_jobs=1 para evitar conflictos en contextos restringidos
    )
    return model


def train_bagging(X_train: np.ndarray, y_train: np.ndarray) -> BaggingClassifier:
    """
    Entrena el modelo Bagging.

    El peso de clase se delega al estimador base con 'balanced' (string),
    lo que permite que sklearn lo recalcule correctamente en cada muestra bootstrap.

    Returns
    -------
    model : BaggingClassifier ajustado
    """
    model = build_bagging_model()
    model.fit(X_train, y_train)
    return model


def predict_bagging(model: BaggingClassifier, X_test: np.ndarray) -> np.ndarray:
    """Genera predicciones del modelo Bagging."""
    return model.predict(X_test)


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
        model = train_bagging(X_arr[train_idx], y_arr[train_idx])
        pred  = predict_bagging(model, X_arr[test_idx])
        preds.append(pred[0])

    print(f"Bagging LOOCV Accuracy: {accuracy_score(y_arr, preds):.4f}")
