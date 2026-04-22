"""
boosting_model.py
-----------------
Modelo de ensamble basado en Boosting.

Estrategia: AdaBoostClassifier con árbol de decisión de profundidad 1 (stump).
  - Aprende secuencialmente: cada modelo se enfoca en los errores del anterior.
  - Los stumps son la base estándar de AdaBoost (mayor boosting, menor varianza
    propia del clasificador base).

  class_weight no es compatible con AdaBoostClassifier directamente;
  se aplica en el estimador base (DecisionTree).
"""

import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from config import BOOSTING_N_ESTIMATORS, BOOSTING_LEARNING_RATE, RANDOM_STATE
from base_estimators_balancing import make_decision_tree


def build_boosting_model() -> AdaBoostClassifier:
    model = AdaBoostClassifier(
        estimator=make_decision_tree(max_depth=1),
        n_estimators=BOOSTING_N_ESTIMATORS,
        learning_rate=BOOSTING_LEARNING_RATE,
        random_state=RANDOM_STATE
    )
    return model


def train_boosting(X_train: np.ndarray, y_train: np.ndarray) -> AdaBoostClassifier:
    model = build_boosting_model()
    model.fit(X_train, y_train)
    return model


def predict_boosting(model: AdaBoostClassifier, X_test: np.ndarray) -> np.ndarray:
    """Genera predicciones del modelo AdaBoost."""
    return model.predict(X_test)
