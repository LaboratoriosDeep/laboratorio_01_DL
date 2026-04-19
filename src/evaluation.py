"""
evaluation.py
-------------
Cálculo de métricas de evaluación y resúmenes comparativos.

Métricas implementadas:
  - Accuracy global
  - Precision macro (ignora soporte de cada clase → trata todas por igual)
  - Recall macro
  - F1-score macro
  - Reporte por clase (classification_report)
  - Matriz de confusión

Por qué métricas macro en vez de weighted:
  Con desbalance de clases, las métricas ponderadas favorecen a las mayoritarias
  y pueden enmascarar el mal desempeño en GDS=3 (n=3).  El promedio macro trata
  a cada clase con igual peso, revelando el comportamiento en clases minoritarias.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from sklearn.model_selection import LeaveOneOut

from balancing import apply_balancing
from config import IMBALANCE_STRATEGY

from config import RANDOM_STATE
from tqdm import tqdm

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    model_name: str = "Modelo") -> dict:
    """
    Calcula el conjunto completo de métricas de clasificación multiclase.

    Returns
    -------
    dict con accuracy, precision_macro, recall_macro, f1_macro, report, conf_matrix
    """
    acc   = accuracy_score(y_true, y_pred)
    prec  = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec   = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1    = f1_score(y_true, y_pred, average="macro", zero_division=0)
    report = classification_report(y_true, y_pred, zero_division=0)
    cm    = confusion_matrix(y_true, y_pred)

    print(f"\n{'='*55}")
    print(f"  {model_name}")
    print(f"{'='*55}")
    print(f"  Accuracy        : {acc:.4f}")
    print(f"  Precision macro : {prec:.4f}")
    print(f"  Recall macro    : {rec:.4f}")
    print(f"  F1-score macro  : {f1:.4f}")
    print(f"\nReporte por clase:\n{report}")
    print(f"Matriz de confusión:\n{cm}\n")

    return {
        "model":            model_name,
        "accuracy":         acc,
        "precision_macro":  prec,
        "recall_macro":     rec,
        "f1_macro":         f1,
        "report":           report,
        "conf_matrix":      cm,
        "y_pred":           y_pred,
    }


def loocv_evaluate(X: np.ndarray, y: np.ndarray,
                   train_fn, predict_fn,
                   model_name: str = "Modelo") -> dict:
    """
    Evalúa un modelo mediante Leave-One-Out Cross-Validation.

    Parámetros
    ----------
    X           : array de predictores
    y           : array de etiquetas
    train_fn    : función(X_train, y_train) → modelo ajustado
    predict_fn  : función(modelo, X_test) → array de predicciones
    model_name  : nombre del modelo para el reporte

    Returns
    -------
    dict con métricas agregadas sobre todos los folds
    """
    loo   = LeaveOneOut()
    preds = np.empty(len(y), dtype=y.dtype)

    for fold, (train_idx, test_idx) in enumerate(tqdm(loo.split(X), total=len(X), desc=f"Evaluando {model_name}")):        
        X_tr, y_tr = X[train_idx], y[train_idx]
    
         # Aplicar balanceo SOLO sobre datos de entrenamiento
        X_tr, y_tr, _ = apply_balancing(X_tr, y_tr, strategy=IMBALANCE_STRATEGY)
    
        model = train_fn(X_tr, y_tr)
        pred  = predict_fn(model, X[test_idx])
        preds[test_idx] = pred

    return compute_metrics(y, preds, model_name=model_name)


def compare_models(results: list) -> pd.DataFrame:
    """
    Genera una tabla comparativa de métricas para todos los modelos.

    Parámetros
    ----------
    results : lista de dicts retornados por compute_metrics / loocv_evaluate

    Returns
    -------
    pd.DataFrame ordenado por F1-score macro descendente
    """
    rows = []
    for r in results:
        rows.append({
            "Modelo":           r["model"],
            "Accuracy":         r["accuracy"],
            "Precision macro":  r["precision_macro"],
            "Recall macro":     r["recall_macro"],
            "F1-score macro":   r["f1_macro"],
        })
    df = pd.DataFrame(rows).sort_values("F1-score macro", ascending=False)
    df = df.reset_index(drop=True)
    return df


# ── Uso directo ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from data_loader import load_data, get_X_y
    from bagging_model  import train_bagging,  predict_bagging
    from boosting_model import train_boosting, predict_boosting
    from stacking_model import train_stacking, predict_stacking

    df = load_data()
    X, y = get_X_y(df)
    X_arr, y_arr = X.values, y.values

    r_bag = loocv_evaluate(X_arr, y_arr, train_bagging,  predict_bagging,  "Bagging")
    r_boo = loocv_evaluate(X_arr, y_arr, train_boosting, predict_boosting, "Boosting")
    r_sta = loocv_evaluate(X_arr, y_arr, train_stacking, predict_stacking, "Stacking")

    tabla = compare_models([r_bag, r_boo, r_sta])
    print("\nTabla comparativa:\n")
    print(tabla.to_string(index=False))
