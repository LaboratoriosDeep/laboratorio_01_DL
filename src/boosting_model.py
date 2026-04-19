"""
boosting_model.py
-----------------
Modelo de ensamble basado en Boosting.

Estrategia: AdaBoostClassifier con árbol de decisión de profundidad 1 (stump).
  - Aprende secuencialmente: cada modelo se enfoca en los errores del anterior.
  - Los stumps son la base estándar de AdaBoost (mayor boosting, menor varianza
    propia del clasificador base).

Discusión con este dataset:
  - Ventaja: puede superar a modelos base en clases difíciles de clasificar.
  - Riesgo: con n=22 y clase GDS=3 (solo 3 muestras), AdaBoost puede
    sobreajustar al concentrar "demasiado" peso en esas pocas instancias.
    El parámetro learning_rate controla este efecto; valores más bajos reducen
    el riesgo de sobreajuste a costa de necesitar más iteraciones.
  - La estrategia class_weight no es directamente compatible con AdaBoostClassifier,
    por lo que se pasa como parámetro del estimador base cuando corresponde.
"""

import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from config import (
    BOOSTING_N_ESTIMATORS, BOOSTING_LEARNING_RATE,
    RANDOM_STATE, IMBALANCE_STRATEGY
)


def build_boosting_model() -> AdaBoostClassifier:
    """
    Construye el clasificador AdaBoost.
    Se usa class_weight='balanced' en el stump base para manejar el desbalance.

    Returns
    -------
    AdaBoostClassifier listo para entrenar.
    """
    cw = "balanced" if IMBALANCE_STRATEGY == "class_weight" else None
    base_stump = DecisionTreeClassifier(
        max_depth=1,
        class_weight=cw,
        random_state=RANDOM_STATE
    )
    model = AdaBoostClassifier(
        estimator=base_stump,
        n_estimators=BOOSTING_N_ESTIMATORS,
        learning_rate=BOOSTING_LEARNING_RATE,
        random_state=RANDOM_STATE
    )
    return model


def train_boosting(X_train: np.ndarray, y_train: np.ndarray) -> AdaBoostClassifier:
    """
    Entrena el modelo AdaBoost.

    Returns
    -------
    model : AdaBoostClassifier ajustado
    """
    model = build_boosting_model()
    model.fit(X_train, y_train)
    return model


def predict_boosting(model: AdaBoostClassifier, X_test: np.ndarray) -> np.ndarray:
    """Genera predicciones del modelo AdaBoost."""
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
        model = train_boosting(X_arr[train_idx], y_arr[train_idx])
        pred  = predict_boosting(model, X_arr[test_idx])
        preds.append(pred[0])

    print(f"Boosting LOOCV Accuracy: {accuracy_score(y_arr, preds):.4f}")
