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
    """
    Construye los clasificadores base del nivel 0.

    Returns
    -------
    list de (nombre, estimador)
    """
    return [("gnb", make_gnb()), ("dt", make_decision_tree(max_depth=10)), ("lr", make_logistic_regression())]


def build_meta_estimator() -> LogisticRegression:
    """
    Construye el meta-clasificador del nivel 1.

    Returns
    -------
    LogisticRegression
    """
    return make_logistic_regression()


def build_stacking_model() -> StackingClassifier:
    """
    Construye el StackingClassifier completo.

    Returns
    -------
    StackingClassifier
    """
    base  = build_base_estimators()
    meta  = build_meta_estimator()
    model = StackingClassifier(
        estimators=base,
        final_estimator=meta,
        cv=5,  # CV interno del Stacking (genera meta-features); independiente del Stratified 10-Fold externo
        passthrough=False,
        n_jobs=N_JOBS
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
