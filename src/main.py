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

# Agregar directorio actual al path para imports locales
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

warnings.filterwarnings("ignore")  # suprimir warnings de convergencia en logs

from config import (
    DATA_FILE, TARGET_COL, FEATURE_SELECTION,
    IMBALANCE_STRATEGY, CV_STRATEGY
)
from data_loader    import load_data, verify_integrity, get_X_y
from preprocessing  import select_features, select_chi2
from evaluation     import loocv_evaluate, compare_models, get_classification_report_df
from visualization  import (
    plot_class_distribution, plot_confusion_matrix,
    plot_metrics_comparison, plot_feature_importance,
    save_metrics_table
)
from bagging_model  import train_bagging,  predict_bagging
from boosting_model import train_boosting, predict_boosting
from stacking_model import train_stacking, predict_stacking, predict_proba_stacking


# ─────────────────────────────────────────────────────────────────────────────

def print_banner(text: str, width: int = 70, char: str = "=") -> None:
    """Imprime un banner decorativo para organizar la salida."""
    print("\n" + char * width)
    print(f"  {text}")
    print(char * width)


def run_experiment() -> None:
    """Ejecuta el flujo completo del experimento."""

    # ── 1. Carga de datos ─────────────────────────────────────────────────────
    print_banner("1. CARGA Y VERIFICACIÓN DEL DATASET")
    
    df = load_data(DATA_FILE)
    
    if df is None:
        print("\n[ERROR] No se pudo cargar el dataset. Verifique la ruta del archivo.")
        print(f"Ruta esperada: {DATA_FILE}")
        return
    
    X_full, y_full = get_X_y(df, target_col=TARGET_COL)
    verify_integrity(X_full)
    
    print(f"\n  Instancias : {len(df)}")
    print(f"  Atributos  : {X_full.shape[1]}")
    print(f"  Clases ({TARGET_COL}): {sorted(y_full.unique())}")
    print(f"\n  Distribución de clases:")
    for cls, cnt in y_full.value_counts().sort_index().items():
        pct = cnt / len(y_full) * 100
        print(f"    Clase {int(cls)} → {cnt:2d} instancias ({pct:5.1f}%)")

    plot_class_distribution(y_full)

    # ── 2. Selección de características ───────────────────────────────────────
    print_banner("2. SELECCIÓN DE CARACTERÍSTICAS")
    X_sel, feat_info = select_features(X_full, y_full, strategy=FEATURE_SELECTION)
    print(f"  Estrategia : {FEATURE_SELECTION}")
    print(f"  Features   : {feat_info['n_features']} seleccionadas de {X_full.shape[1]} totales")
    print(f"  Columnas   : {list(X_sel.columns)}")

    # Gráfico Chi2 (siempre se calcula para el informe)
    _, _, chi2_scores = select_chi2(X_full, y_full, k=X_full.shape[1])
    plot_feature_importance(chi2_scores)
    print("\n  Top 5 características por Chi²:")
    print(chi2_scores.head(5).to_string(index=False))

    # ── 3. Protocolo de validación ────────────────────────────────────────────
    print_banner("3. PROTOCOLO EXPERIMENTAL")
    print(f"  Validación cruzada : Leave-One-Out (LOOCV)")
    print(f"  Justificación      : Maximizar uso de datos con n={len(y_full)}")
    print(f"  Desbalance         : Estrategia '{IMBALANCE_STRATEGY}'")
    print(f"  Variable objetivo  : {TARGET_COL} ({len(y_full.unique())} clases)")
    print(f"  Random state       : 42 (reproducibilidad)")

    X_arr = X_sel.values
    y_arr = y_full.values

    # ── 4. Evaluación de modelos ──────────────────────────────────────────────
    print_banner("4. EVALUACIÓN DE MODELOS DE ENSAMBLE")

    print("\n[4a] BAGGING (BaggingClassifier + DecisionTree)")
    r_bag = loocv_evaluate(
        X_arr, y_arr,
        train_fn=train_bagging,
        predict_fn=predict_bagging,
        model_name="Bagging"
    )

    print_banner("", width=70, char="-")
    print("[4b] BOOSTING (AdaBoost + DecisionStump)")
    r_boo = loocv_evaluate(
        X_arr, y_arr,
        train_fn=train_boosting,
        predict_fn=predict_boosting,
        model_name="Boosting"
    )

    print_banner("", width=70, char="-")
    print("[4c] STACKING (GNB + DT + LR → LR meta-modelo)")
    r_sta = loocv_evaluate(
        X_arr, y_arr,
        train_fn=train_stacking,
        predict_fn=predict_stacking,
        predict_proba_fn=predict_proba_stacking,
        model_name="Stacking"
    )

    # ── 5. Comparación y visualizaciones ──────────────────────────────────────
    print_banner("5. COMPARACIÓN E INTERPRETACIÓN")
    all_results = [r_bag, r_boo, r_sta]
    tabla = compare_models(all_results)
    
    print("\n╔" + "═" * 68 + "╗")
    print("║" + " " * 20 + "TABLA COMPARATIVA (LOOCV)" + " " * 23 + "║")
    print("╚" + "═" * 68 + "╝\n")
    print(tabla.to_string(index=False))

    # Determinar mejor modelo por F1 macro
    best = tabla.iloc[0]
    print(f"\n  ► MEJOR MODELO por F1-score macro:")
    print(f"    {best['Modelo']:15s} → F1 = {best['F1-score macro']:.4f}")

    # Análisis por clase
    print("\n  Análisis de rendimiento por clase:")
    for r in all_results:
        print(f"\n  {r['model']}:")
        class_report = get_classification_report_df(y_arr, r["y_pred"])
        print(class_report.to_string())

    # Matrices de confusión
    print_banner("6. GENERACIÓN DE VISUALIZACIONES")
    for r in all_results:
        plot_confusion_matrix(y_arr, r["y_pred"], model_name=r["model"])

    # Gráfico comparativo de métricas
    plot_metrics_comparison(tabla)

    # Exportar tabla
    save_metrics_table(tabla)

    # ── 7. Interpretación textual ─────────────────────────────────────────────
    print_banner("7. INTERPRETACIÓN DE RESULTADOS")
    _print_interpretation(r_bag, r_boo, r_sta, y_arr, tabla)

    print_banner("EXPERIMENTO COMPLETADO ✓", char="═")
    print(f"  📊 Figuras guardadas en : {os.path.relpath('reports/figuras')}")
    print(f"  📄 Tablas guardadas en  : {os.path.relpath('reports/tablas')}")
    print("\n  Para visualizar los resultados:")
    print("    - Revise las matrices de confusión en reports/figuras/")
    print("    - Analice la tabla comparativa en reports/tablas/")
    print("    - Compare las métricas en el gráfico de comparación\n")


def _print_interpretation(r_bag, r_boo, r_sta, y_true, tabla: 'pd.DataFrame') -> None:
    """Imprime una interpretación automática de los resultados."""

    models = {
        "Bagging":  r_bag,
        "Boosting": r_boo,
        "Stacking": r_sta,
    }

    # Identificar clase minoritaria
    classes = np.unique(y_true)
    class_counts = [np.sum(y_true == c) for c in classes]
    minority_cls = classes[np.argmin(class_counts)]

    print(f"\n  ANÁLISIS DE CLASE MINORITARIA")
    print(f"  {'─' * 50}")
    print(f"  Clase minoritaria: {minority_cls} (n={np.sum(y_true == minority_cls)})\n")

    from sklearn.metrics import recall_score
    for name, r in models.items():
        rec_minority = recall_score(
            y_true, r["y_pred"],
            labels=[minority_cls], average="macro", zero_division=0
        )
        print(f"  {name:<12} → Recall clase {minority_cls}: {rec_minority:.3f}  "
              f"| F1 macro: {r['f1_macro']:.4f}")

    # Análisis de diferencias entre modelos
    print(f"\n  ANÁLISIS COMPARATIVO")
    print(f"  {'─' * 50}")
    f1_scores = {r['model']: r['f1_macro'] for r in [r_bag, r_boo, r_sta]}
    best_model = max(f1_scores, key=f1_scores.get)
    worst_model = min(f1_scores, key=f1_scores.get)
    diff = f1_scores[best_model] - f1_scores[worst_model]
    
    print(f"  Mejor modelo : {best_model} (F1={f1_scores[best_model]:.4f})")
    print(f"  Peor modelo  : {worst_model} (F1={f1_scores[worst_model]:.4f})")
    print(f"  Diferencia   : {diff:.4f} ({diff*100:.2f}%)")
    
    if diff < 0.05:
        print(f"\n  ⚠ Las diferencias son pequeñas (< 5%) - Los modelos tienen")
        print(f"     rendimiento similar en este dataset pequeño.")

    print(f"\n  OBSERVACIONES METODOLÓGICAS")
    print(f"  {'─' * 50}")
    print(
        "\n  · El F1-score macro es la métrica principal por el desbalance de clases."
        "\n  · Con n={} los intervalos de confianza son amplios; pequeñas diferencias".format(len(y_true))
        + "\n    entre modelos pueden deberse a varianza muestral."
        "\n  · Boosting puede sobreajustar con pocos datos si la clase minoritaria"
        "\n    tiene muy pocas muestras (riesgo de memorización)."
        "\n  · Bagging reduce la varianza mediante bootstrap, estabilizando"
        "\n    las predicciones de árboles individuales."
        "\n  · Stacking combina perspectivas complementarias (probabilístico,"
        "\n    no lineal, lineal), siendo robusto conceptualmente aunque no"
        "\n    siempre superior numéricamente en datasets tan pequeños."
    )


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Crear directorios si no existen
    os.makedirs("reports/figuras", exist_ok=True)
    os.makedirs("reports/tablas", exist_ok=True)
    os.makedirs("data/raw", exist_ok=True)
    
    print("\n" + "="*70)
    print("  LABORATORIO 01 - MÉTODOS DE ENSAMBLE")
    print("  Predicción del Deterioro Cognitivo")
    print("  Variable objetivo: {}".format("GDS_R3"))
    print("="*70)
    
    run_experiment()