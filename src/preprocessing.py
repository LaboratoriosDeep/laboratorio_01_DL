"""
preprocessing.py
----------------
Limpieza, selección de características y escalamiento.

IMPORTANTE: ningún ajuste (fit) se realiza sobre datos de prueba.
Toda transformación que dependa de estadísticas del conjunto debe
ocurrir solo sobre datos de entrenamiento dentro del esquema de
validación.
"""

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2, RFE
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from config import (
    FEATURE_SELECTION, VARIANCE_THRESHOLD, CHI2_K_BEST,
    RANDOM_STATE, LR_MAX_ITER
)


def remove_low_variance(X: pd.DataFrame, threshold: float = VARIANCE_THRESHOLD):
    """
    Elimina columnas con varianza menor al umbral indicado.
    Útil para datos binarios donde casi todos los valores son iguales.

    Returns
    -------
    X_filtered : pd.DataFrame
    selector   : VarianceThreshold (ajustado)
    """
    selector = VarianceThreshold(threshold=threshold)
    X_arr = selector.fit_transform(X)
    selected = X.columns[selector.get_support()]
    X_filtered = pd.DataFrame(X_arr, columns=selected, index=X.index)
    print(f"[preprocessing] Varianza: {len(selected)}/{len(X.columns)} características seleccionadas.")
    return X_filtered, selector


def select_chi2(X: pd.DataFrame, y: pd.Series, k: int = CHI2_K_BEST):
    """
    Selecciona las k mejores características según la prueba chi-cuadrado.
    Adecuada para datos binarios/categóricos.

    Returns
    -------
    X_selected : pd.DataFrame
    selector   : SelectKBest
    scores     : pd.DataFrame con puntajes ordenados
    """
    k = min(k, X.shape[1])
    selector = SelectKBest(score_func=chi2, k=k)
    selector.fit(X, y)
    selected = X.columns[selector.get_support()]
    X_selected = X[selected].copy()

    scores = pd.DataFrame({
        "Feature":    X.columns,
        "Chi2 Score": selector.scores_,
        "p-value":    selector.pvalues_,
    }).sort_values("Chi2 Score", ascending=False)

    return X_selected, selector, scores


def select_rfe(X: pd.DataFrame, y: pd.Series, n_features: int = 10):
    """
    Recursive Feature Elimination con Regresión Logística como estimador base.

    Returns
    -------
    X_selected : pd.DataFrame
    selector   : RFE
    ranking    : pd.DataFrame con ranking de características
    """
    n_features = min(n_features, X.shape[1])
    model = LogisticRegression(
        max_iter=LR_MAX_ITER, class_weight="balanced",
        random_state=RANDOM_STATE, solver="lbfgs"
    )
    selector = RFE(estimator=model, n_features_to_select=n_features)
    selector.fit(X, y)
    selected = X.columns[selector.get_support()]
    X_selected = X[selected].copy()

    ranking = pd.DataFrame({
        "Feature":  X.columns,
        "Ranking":  selector.ranking_,
        "Selected": selector.support_,
    }).sort_values("Ranking")

    print(f"[preprocessing] RFE: {len(selected)}/{len(X.columns)} características seleccionadas.")
    return X_selected, selector, ranking


def select_features(X: pd.DataFrame, y: pd.Series, strategy: str = FEATURE_SELECTION):
    """
    Aplica la estrategia de selección de características configurada.

    Parámetros
    ----------
    strategy : "all" | "variance" | "chi2"

    Returns
    -------
    X_sel : pd.DataFrame  con las características seleccionadas
    info  : dict          con metadata de la selección
    """
    if strategy == "all":
        print(f"[preprocessing] Sin filtro: usando las {X.shape[1]} características.")
        return X.copy(), {"strategy": "all", "n_features": X.shape[1]}

    elif strategy == "variance":
        X_sel, selector = remove_low_variance(X)
        return X_sel, {"strategy": "variance", "selector": selector,
                       "n_features": X_sel.shape[1]}

    elif strategy == "chi2":
        X_sel, selector, scores = select_chi2(X, y)
        return X_sel, {"strategy": "chi2", "selector": selector,
                       "scores": scores, "n_features": X_sel.shape[1]}

    else:
        raise ValueError(f"Estrategia desconocida: '{strategy}'")


