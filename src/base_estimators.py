"""
base_estimators.py
------------------
Funciones factory para los clasificadores base reutilizados en los ensambles.
Centraliza parámetros y lógica de class_weight en un único lugar.
"""

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from config import DT_MAX_DEPTH, LR_MAX_ITER, RANDOM_STATE, IMBALANCE_STRATEGY


def get_class_weight():
    """Retorna 'balanced' si la estrategia activa usa pesos por clase, si no None."""
    return "balanced" if IMBALANCE_STRATEGY == "class_weight" else None


def make_decision_tree(max_depth: int = DT_MAX_DEPTH) -> DecisionTreeClassifier:
    """
    Árbol de decisión con los parámetros estándar del proyecto.
    max_depth es parametrizable para que Boosting pueda pedir max_depth=1 (stump).
    """
    return DecisionTreeClassifier(
        max_depth=max_depth,
        class_weight=get_class_weight(),
        random_state=RANDOM_STATE,
    )


def make_logistic_regression() -> LogisticRegression:
    """Regresión logística con los parámetros estándar del proyecto."""
    return LogisticRegression(
        max_iter=LR_MAX_ITER,
        class_weight=get_class_weight(),
        random_state=RANDOM_STATE,
        solver="lbfgs",
    )
