import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.metrics import confusion_matrix
from config import REPORTS_FIG_DIR, REPORTS_TAB_DIR, TARGET_COL

# Estilo global
plt.rcParams.update({
    "figure.dpi": 120,
    "font.size":  11,
    "axes.titlesize": 13,
})
PALETTE = ["#4C72B0", "#DD8452", "#55A868"]


# ── 1. Distribución de clases ─────────────────────────────────────────────────

def plot_class_distribution(y: pd.Series, title: str = None,
                             filename: str = "dist_clases.png") -> None:
    """
    Barplot con frecuencia absoluta de cada clase.
    Anota el porcentaje sobre cada barra para evidenciar el desbalance.
    """
    counts = y.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(counts.index.astype(str), counts.values, color=PALETTE[:len(counts)])

    total = counts.sum()
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                f"{val} ({val/total*100:.1f}%)",
                ha="center", va="bottom", fontsize=9)

    ax.set_xlabel(f"Clase ({TARGET_COL})")
    ax.set_ylabel("Frecuencia")
    ax.set_title(title or f"Distribución de la variable objetivo ({TARGET_COL})")
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    plt.tight_layout()

    path = os.path.join(REPORTS_FIG_DIR, filename)
    fig.savefig(path)
    plt.close(fig)
    print(f"[viz] Guardado: {path}")


# ── 2. Heatmap de matriz de confusión ─────────────────────────────────────────

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                          model_name: str = "Modelo",
                          filename: str = None) -> None:
    """
    Heatmap de la matriz de confusión con anotaciones absolutas.
    """
    cm     = confusion_matrix(y_true, y_pred)
    labels = sorted(np.unique(y_true))
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax,
                linewidths=0.5, linecolor="gray")
    ax.set_xlabel("Clase predicha")
    ax.set_ylabel("Clase real")
    ax.set_title(f"Matriz de confusión – {model_name}")
    plt.tight_layout()

    fname = filename or f"cm_{model_name.lower().replace(' ', '_')}.png"
    path  = os.path.join(REPORTS_FIG_DIR, fname)
    fig.savefig(path)
    plt.close(fig)
    print(f"[viz] Guardado: {path}")


# ── 3. Comparativa de métricas ────────────────────────────────────────────────

def plot_metrics_comparison(df_metrics: pd.DataFrame,
                            filename: str = "comparacion_modelos.png") -> None:
    """
    Barplot agrupado comparando Accuracy, Precision, Recall y F1 entre modelos.
    """
    metrics = ["Accuracy", "Precision macro", "Recall macro", "F1-score macro"]
    x      = np.arange(len(df_metrics))
    width  = 0.18
    fig, ax = plt.subplots(figsize=(9, 5))

    for i, metric in enumerate(metrics):
        offset = (i - 1.5) * width
        rects  = ax.bar(x + offset, df_metrics[metric], width,
                        label=metric, color=PALETTE[i % len(PALETTE)],
                        alpha=0.85)
        for rect in rects:
            ax.text(rect.get_x() + rect.get_width() / 2,
                    rect.get_height() + 0.01,
                    f"{rect.get_height():.2f}",
                    ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(df_metrics["Modelo"])
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Valor")
    ax.set_title("Comparación de métricas entre modelos de ensamble (LOOCV)")
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()

    path = os.path.join(REPORTS_FIG_DIR, filename)
    fig.savefig(path)
    plt.close(fig)
    print(f"[viz] Guardado: {path}")


# ── 4. Feature importance (Chi2 scores) ───────────────────────────────────────

def plot_feature_importance(scores: pd.DataFrame,
                            filename: str = "chi2_features.png") -> None:
    """
    Barplot horizontal de importancia de características según Chi2.
    """
    top = scores.head(15).sort_values("Chi2 Score", ascending=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(top["Feature"], top["Chi2 Score"], color="#4C72B0", alpha=0.85)
    ax.set_xlabel("Chi² Score")
    ax.set_title("Importancia de características (prueba Chi²)")
    plt.tight_layout()

    path = os.path.join(REPORTS_FIG_DIR, filename)
    fig.savefig(path)
    plt.close(fig)
    print(f"[viz] Guardado: {path}")


# ── 5. Exportar tabla a CSV ───────────────────────────────────────────────────

def save_metrics_table(df_metrics: pd.DataFrame,
                       filename: str = "tabla_comparativa.csv") -> None:
    """Guarda la tabla comparativa de métricas en CSV."""
    path = os.path.join(REPORTS_TAB_DIR, filename)
    df_metrics.to_csv(path, index=False, float_format="%.4f")
    print(f"[viz] Tabla guardada: {path}")


# ── Uso directo ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from data_loader import load_data, get_X_y

    df = load_data()
    X, y = get_X_y(df)
    plot_class_distribution(y)
    print("Gráfico de distribución generado.")
