"""
balancing.py
------------
Estrategias para enfrentar el desbalance de clases.

Dado el tamaño muy reducido del dataset (n=22) con distribución desigual
(GDS 1: 10, GDS 2: 9, GDS 3: 3), se adopta como estrategia principal el uso
de pesos por clase (class_weight='balanced').  Esta opción no genera
observaciones sintéticas y es segura con muestras tan pequeñas.

El sobremuestreo con SMOTE se proporciona como alternativa, pero requiere
que la clase minoritaria tenga al menos k_neighbors+1 muestras; dado que
GDS=3 solo tiene 3 muestras, se configura k_neighbors=1.
"""

import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight
from config import IMBALANCE_STRATEGY, RANDOM_STATE


def get_class_weights(y: np.ndarray) -> dict:
    """
    Calcula pesos inversamente proporcionales a la frecuencia de cada clase.
    Retorna un diccionario {clase: peso} listo para pasar a sklearn.

    Ejemplo con GDS: clase 3 (n=3) recibirá ~3.7× más peso que clase 1 (n=10).
    """
    classes = np.unique(y)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    class_weight_dict = dict(zip(classes, weights))
    print(f"[balancing] Pesos de clase: {class_weight_dict}")
    return class_weight_dict


def apply_smote(X_train: np.ndarray, y_train: np.ndarray,
                k_neighbors: int = 1) -> tuple:
    """
    Aplica SMOTE (Synthetic Minority Over-sampling Technique) solo a los
    datos de entrenamiento, dentro del loop de validación.

    Requiere la librería imbalanced-learn (imblearn).
    Si no está disponible, lanza ImportError con instrucciones claras.

    ADVERTENCIA: con n_minority=3, usar k_neighbors=1 (mínimo válido).

    Returns
    -------
    X_res : np.ndarray  rebalanceado
    y_res : np.ndarray  rebalanceado
    """
    try:
        from imblearn.over_sampling import SMOTE
    except ImportError:
        raise ImportError(
            "La librería imbalanced-learn no está instalada.\n"
            "Instale con: pip install imbalanced-learn\n"
            "O use la estrategia 'class_weight' en config.py."
        )

    smote = SMOTE(k_neighbors=k_neighbors, random_state=RANDOM_STATE)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    # print(f"[balancing] SMOTE: {len(y_train)} → {len(y_res)} muestras.")
    return X_res, y_res


def apply_balancing(X_train: np.ndarray, y_train: np.ndarray,
                    strategy: str = IMBALANCE_STRATEGY) -> tuple:
    """
    Punto de entrada principal para la estrategia de balanceo.

    Parámetros
    ----------
    strategy : "class_weight" | "smote" | "none"
        - "class_weight" → retorna X, y sin modificar (los pesos se pasan
          al estimador vía get_class_weights)
        - "smote"        → genera muestras sintéticas en X_train
        - "none"         → sin tratamiento

    Returns
    -------
    X_out : np.ndarray
    y_out : np.ndarray
    cw    : dict | None  — diccionario de pesos (solo para "class_weight")
    """
    if strategy == "class_weight":
        cw = get_class_weights(y_train)
        return X_train, y_train, cw

    elif strategy == "smote":
        X_res, y_res = apply_smote(X_train, y_train)
        return X_res, y_res, None

    elif strategy == "none":
        print("[balancing] Sin tratamiento de desbalance.")
        return X_train, y_train, None

    else:
        raise ValueError(f"Estrategia de balanceo desconocida: '{strategy}'")


# ── Uso directo ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from data_loader import load_data, get_X_y

    df = load_data()
    X, y = get_X_y(df)

    print("Distribución original:", dict(zip(*np.unique(y, return_counts=True))))
    _, _, cw = apply_balancing(X.values, y.values, strategy="class_weight")
    print("Pesos calculados:", cw)
