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
  - AUC-ROC (cuando aplique para clasificación binaria)

Por qué métricas macro en vez de weighted:
  Con desbalance de clases, las métricas ponderadas favorecen a las mayoritarias
  y pueden enmascarar el mal desempeño en clases minoritarias. El promedio macro
  trata a cada clase con igual peso, revelando el comportamiento real del modelo.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score
)
from sklearn.model_selection import LeaveOneOut

from balancing import apply_balancing
from config import IMBALANCE_STRATEGY, RANDOM_STATE
from tqdm import tqdm


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    y_pred_proba: np.ndarray = None,
                    model_name: str = "Modelo") -> dict:
    """
    Calcula el conjunto completo de métricas de clasificación multiclase.

    Parameters
    ----------
    y_true : array de etiquetas verdaderas
    y_pred : array de predicciones
    y_pred_proba : array de probabilidades (opcional, para AUC)
    model_name : nombre del modelo

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
    
    # Calcular AUC si hay probabilidades y es clasificación binaria
    auc = None
    if y_pred_proba is not None and len(np.unique(y_true)) == 2:
        try:
            auc = roc_auc_score(y_true, y_pred_proba[:, 1])
        except:
            pass

    print(f"\n{'='*55}")
    print(f"  {model_name}")
    print(f"{'='*55}")
    print(f"  Accuracy        : {acc:.4f}")
    print(f"  Precision macro : {prec:.4f}")
    print(f"  Recall macro    : {rec:.4f}")
    print(f"  F1-score macro  : {f1:.4f}")
    if auc is not None:
        print(f"  AUC-ROC         : {auc:.4f}")
    print(f"\nReporte por clase:\n{report}")
    print(f"Matriz de confusión:\n{cm}\n")

    return {
        "model":            model_name,
        "accuracy":         acc,
        "precision_macro":  prec,
        "recall_macro":     rec,
        "f1_macro":         f1,
        "auc":              auc,
        "report":           report,
        "conf_matrix":      cm,
        "y_true":           y_true,
        "y_pred":           y_pred,
        "y_pred_proba":     y_pred_proba,
    }


def loocv_evaluate(X: np.ndarray, y: np.ndarray,
                   train_fn, predict_fn, predict_proba_fn=None,
                   model_name: str = "Modelo") -> dict:
    """
    Evalúa un modelo mediante Leave-One-Out Cross-Validation.

    Parámetros
    ----------
    X               : array de predictores
    y               : array de etiquetas
    train_fn        : función(X_train, y_train) → modelo ajustado
    predict_fn      : función(modelo, X_test) → array de predicciones
    predict_proba_fn: función(modelo, X_test) → array de probabilidades (opcional)
    model_name      : nombre del modelo para el reporte

    Returns
    -------
    dict con métricas agregadas sobre todos los folds
    """
    loo   = LeaveOneOut()
    preds = np.empty(len(y), dtype=y.dtype)
    
    # Preparar array para probabilidades si aplica
    if predict_proba_fn is not None:
        n_classes = len(np.unique(y))
        proba = np.empty((len(y), n_classes), dtype=float)
    else:
        proba = None

    for fold, (train_idx, test_idx) in enumerate(tqdm(loo.split(X), total=len(X), desc=f"Evaluando {model_name}")):        
        X_tr, y_tr = X[train_idx], y[train_idx]
    
        # Aplicar balanceo SOLO sobre datos de entrenamiento
        X_tr, y_tr, _ = apply_balancing(X_tr, y_tr, strategy=IMBALANCE_STRATEGY)
    
        model = train_fn(X_tr, y_tr)
        pred  = predict_fn(model, X[test_idx])
        preds[test_idx] = pred
        
        # Obtener probabilidades si se proporciona la función
        if predict_proba_fn is not None:
            try:
                proba[test_idx] = predict_proba_fn(model, X[test_idx])
            except:
                proba = None

    return compute_metrics(y, preds, proba, model_name=model_name)


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
        row = {
            "Modelo":           r["model"],
            "Accuracy":         r["accuracy"],
            "Precision macro":  r["precision_macro"],
            "Recall macro":     r["recall_macro"],
            "F1-score macro":   r["f1_macro"],
        }
        if r.get("auc") is not None:
            row["AUC-ROC"] = r["auc"]
        rows.append(row)
    
    df = pd.DataFrame(rows).sort_values("F1-score macro", ascending=False)
    df = df.reset_index(drop=True)
    return df


def get_classification_report_df(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    """
    Convierte el classification_report a DataFrame para mejor visualización.
    
    Returns
    -------
    pd.DataFrame con métricas por clase
    """
    from sklearn.metrics import classification_report
    report_dict = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    
    # Filtrar solo las clases numéricas
    classes_data = {k: v for k, v in report_dict.items() 
                   if k not in ['accuracy', 'macro avg', 'weighted avg']}
    
    df = pd.DataFrame(classes_data).T
    df.index.name = 'Clase'
    return df
