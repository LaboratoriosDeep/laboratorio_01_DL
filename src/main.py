"""
main.py
-------
Orquestación general del flujo experimental del Laboratorio 01.

Secuencia:
  1. Carga y verificación del dataset
  2. Exploración y distribución de la variable objetivo
  3. Selección de características
  4. Evaluación LOOCV de los tres modelos de ensamble
  5. Comparación e interpretación de resultados
  6. Generación de figuras y tabla comparativa
"""

import sys
import os
import warnings
import numpy as np

# Agregar src/ al path para imports locales
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

warnings.filterwarnings("ignore")  # suprimir warnings de convergencia en logs

from config import (
    DATA_FILE, TARGET_COL, FEATURE_SELECTION,
    IMBALANCE_STRATEGY, CV_STRATEGY
)
from data_loader    import load_data, verify_integrity, get_X_y
from preprocessing  import select_features, select_chi2
from evaluation     import loocv_evaluate, compare_models
from visualization  import (
    plot_class_distribution, plot_confusion_matrix,
    plot_metrics_comparison, plot_feature_importance,
    save_metrics_table
)
from bagging_model  import train_bagging,  predict_bagging
from boosting_model import train_boosting, predict_boosting
from stacking_model import train_stacking, predict_stacking


# ─────────────────────────────────────────────────────────────────────────────

def print_banner(text: str, width: int = 60) -> None:
    print("\n" + "=" * width)
    print(f"  {text}")
    print("=" * width)


def run_experiment() -> None:

# ── 1. Carga de datos ─────────────────────────────────────────────────────
    print_banner("1. CARGA Y VERIFICACIÓN DEL DATASET")
    
    # Usamos la ruta oficial y absoluta que definiste en config.py
    df = load_data(DATA_FILE)
    
    X_full, y_full = get_X_y(df, target_col=TARGET_COL)
    verify_integrity(X_full)
    
    print(f"\n  Instancias : {len(df)}")
    print(f"  Atributos  : {X_full.shape[1]}")
    print(f"  Clases ({TARGET_COL}): {sorted(y_full.unique())}")
    print(f"\n  Distribución de clases:")
    for cls, cnt in y_full.value_counts().sort_index().items():
        pct = cnt / len(y_full) * 100
        print(f"    GDS={int(cls)} → {cnt} instancias ({pct:.1f}%)")

    # SOLUCIÓN: Le pasamos 'y_full' (la serie) como espera visualization.py
    plot_class_distribution(y_full)

    # ── 2. Selección de características ───────────────────────────────────────
    print_banner("2. SELECCIÓN DE CARACTERÍSTICAS")
    X_sel, feat_info = select_features(X_full, y_full, strategy=FEATURE_SELECTION)
    print(f"  Estrategia : {FEATURE_SELECTION}")
    print(f"  Features   : {feat_info['n_features']} seleccionadas")
    print(f"  Columnas   : {list(X_sel.columns)}")

    # Gráfico Chi2 (siempre se calcula para el informe)
    _, _, chi2_scores = select_chi2(X_full, y_full, k=X_full.shape[1])
    plot_feature_importance(chi2_scores)
    print("\n  Top 5 características por Chi²:")
    print(chi2_scores.head(5).to_string(index=False))

    # ── 3. Protocolo de validación ────────────────────────────────────────────
    print_banner("3. PROTOCOLO EXPERIMENTAL")
    print(f"  Validación       : Leave-One-Out CV (LOOCV)")
    print(f"  Razón            : Maximizar rendimiento")
    print(f"  Desbalance       : estrategia '{IMBALANCE_STRATEGY}'")
    print(f"  Variable objetivo: {TARGET_COL} ({len(y_full.unique())} clases)")

    X_arr = X_sel.values
    y_arr = y_full.values

    # ── 4. Evaluación de modelos ──────────────────────────────────────────────

    print_banner("4a. MODELO BAGGING  (BaggingClassifier + DecisionTree)")
    r_bag = loocv_evaluate(
        X_arr, y_arr,
        train_fn=train_bagging,
        predict_fn=predict_bagging,
        model_name="Bagging"
    )

    print_banner("4b. MODELO BOOSTING  (AdaBoost + DecisionStump)")
    r_boo = loocv_evaluate(
        X_arr, y_arr,
        train_fn=train_boosting,
        predict_fn=predict_boosting,
        model_name="Boosting"
    )

    print_banner("4c. MODELO STACKING  (GNB + DT + KNN → LR)")
    r_sta = loocv_evaluate(
        X_arr, y_arr,
        train_fn=train_stacking,
        predict_fn=predict_stacking,
        model_name="Stacking"
    )

    # ── 5. Comparación y visualizaciones ──────────────────────────────────────
    print_banner("5. COMPARACIÓN E INTERPRETACIÓN")
    all_results = [r_bag, r_boo, r_sta]
    tabla = compare_models(all_results)
    print("\nTabla comparativa (LOOCV):\n")
    print(tabla.to_string(index=False))

    # Determinar mejor modelo por F1 macro
    best = tabla.iloc[0]
    print(f"\n  ► Mejor modelo por F1-score macro: {best['Modelo']} "
          f"(F1={best['F1-score macro']:.4f})")

    # Matrices de confusión
    for r in all_results:
        plot_confusion_matrix(y_arr, r["y_pred"], model_name=r["model"])

    # Gráfico comparativo de métricas
    plot_metrics_comparison(tabla)

    # Exportar tabla
    save_metrics_table(tabla)

    # ── 6. Interpretación textual ─────────────────────────────────────────────
    print_banner("6. INTERPRETACIÓN DE RESULTADOS")
    _print_interpretation(r_bag, r_boo, r_sta, y_arr)

    print_banner("EXPERIMENTO COMPLETADO")
    print("  Figuras guardadas en : reports/figuras/")
    print("  Tablas guardadas en  : reports/tablas/")


def _print_interpretation(r_bag, r_boo, r_sta, y_true) -> None:
    """Imprime una interpretación automática de los resultados."""

    models = {
        "Bagging":  r_bag,
        "Boosting": r_boo,
        "Stacking": r_sta,
    }

    # Qué modelo predijo mejor la clase minoritaria (GDS=3)
    classes = np.unique(y_true)
    minority_cls = classes[np.argmin([np.sum(y_true == c) for c in classes])]

    print(f"\n  Clase minoritaria: GDS={minority_cls} "
          f"(n={np.sum(y_true == minority_cls)})\n")

    from sklearn.metrics import recall_score
    for name, r in models.items():
        rec_minority = recall_score(
            y_true, r["y_pred"],
            labels=[minority_cls], average="macro", zero_division=0
        )
        print(f"  {name:<10} — Recall en GDS={minority_cls}: {rec_minority:.2f}  "
              f"| F1 macro: {r['f1_macro']:.4f}")

    print(
        "\n  Observaciones metodológicas:"
        "\n  · El accuracy no es suficiente por el desbalance; "
        "se priorizó F1 macro."
        "\n  · Con n=22 los resultados tienen alta varianza; "
        "las diferencias pequeñas entre modelos no son concluyentes."
        "\n  · Boosting puede sobreajustar en datasets tan pequeños "
        "(risk de memorizar GDS=3)."
        "\n  · Bagging mitiga la varianza de un árbol individual "
        "mediante el remuestreo bootstrap."
        "\n  · Stacking integra perspectivas complementarias (probabilístico, "
        "geométrico, local),\n    siendo el más robusto conceptualmente "
        "aunque no siempre el mejor numéricamente."
    )


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_experiment()
