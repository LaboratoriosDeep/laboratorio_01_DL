"""
stacking_model.py
-----------------
Modelo de ensamble basado en Stacking.

Arquitectura de dos niveles:
  Nivel 0 (modelos base):
    - GaussianNB   → perspectiva probabilística (Bayes)
    - DecisionTree → perspectiva no lineal y local
    - LogisticRegression → perspectiva lineal y regularizada

  Nivel 1 (meta-clasificador):
    - LogisticRegression → aprende cómo combinar las predicciones base

Justificación de la selección de modelos base:
  - Naive Bayes: simple, probabilístico, funciona bien con binarios y poco datos.
  - Árbol: captura patrones no lineales; su alta varianza beneficia al stacking.
  - LogisticRegression: modelo lineal estable, complementa a los anteriores.
"""

import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from config import N_JOBS
from base_estimators_balancing import make_decision_tree, make_logistic_regression, make_gnb


def build_base_estimators() -> list:
    return [("gnb", make_gnb()), ("dt", make_decision_tree(max_depth=10)), ("lr", make_logistic_regression())]


def build_meta_estimator() -> LogisticRegression:
    return make_logistic_regression()


def build_stacking_model() -> StackingClassifier:
    """Construye el StackingClassifier con los estimadores base y el meta-modelo."""
    base  = build_base_estimators()
    meta  = build_meta_estimator()
    model = StackingClassifier(
        estimators=base,
        final_estimator=meta,
        cv=5,  # CV interno para generar meta-features; independiente del LOOCV externo
        passthrough=False,
        n_jobs=N_JOBS
    )
    return model


def train_stacking(X_train: np.ndarray, y_train: np.ndarray) -> StackingClassifier:
    model = build_stacking_model()
    model.fit(X_train, y_train)
    return model


def predict_stacking(model: StackingClassifier, X_test: np.ndarray) -> np.ndarray:
    """Genera predicciones finales del StackingClassifier."""
    return model.predict(X_test)
