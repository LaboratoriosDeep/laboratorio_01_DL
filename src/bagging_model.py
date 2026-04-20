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

"""

import numpy as np
from sklearn.ensemble import BaggingClassifier
from config import BAGGING_N_ESTIMATORS, RANDOM_STATE
from base_estimators_balancing import make_decision_tree


def build_bagging_model() -> BaggingClassifier:
    """
    Construye el clasificador Bagging con los hiperparámetros del config.

    Returns
    -------
    BaggingClassifier listo para entrenar.
    """
    model = BaggingClassifier(
        estimator=make_decision_tree(),
        n_estimators=BAGGING_N_ESTIMATORS,
        bootstrap=True,
        random_state=RANDOM_STATE,
        n_jobs= 1   # n_jobs=1 para evitar conflictos en contextos restringidos
    )
    return model


def train_bagging(X_train: np.ndarray, y_train: np.ndarray) -> BaggingClassifier:
    """
    Entrena el modelo Bagging.

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
