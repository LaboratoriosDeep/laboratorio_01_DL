import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from config import REPORTS_FIG_DIR, REPORTS_TAB_DIR, TARGET_COL

# Estilo global
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams.update({
    "figure.dpi": 120,
    "font.size":  11,
    "axes.titlesize": 13,
    "figure.figsize": (8, 6)
})
PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]


# ── 1. Distribución de clases ─────────────────────────────────────────────────

def plot_class_distribution(y: pd.Series, title: str = None,
                             filename: str = "dist_clases.png") -> None:
    """
    Barplot con frecuencia absoluta de cada clase.
    Anota el porcentaje sobre cada barra para evidenciar el desbalance.
    """
    os.makedirs(REPORTS_FIG_DIR, exist_ok=True)
    
    counts = y.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(counts.index.astype(str), counts.values, 
                  color=PALETTE[:len(counts)], edgecolor='black', linewidth=1.2)

    total = counts.sum()
    for bar, val in zip(bars, counts.values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2,
                height + 0.5,
                f"{val}\n({val/total*100:.1f}%)",
                ha="center", va="bottom", fontsize=10, fontweight='bold')

    ax.set_xlabel(f"Clase ({TARGET_COL})", fontsize=12, fontweight='bold')
    ax.set_ylabel("Frecuencia", fontsize=12, fontweight='bold')
    ax.set_title(title or f"Distribución de la variable objetivo ({TARGET_COL})",
                 fontsize=14, fontweight='bold', pad=20)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()

    path = os.path.join(REPORTS_FIG_DIR, filename)
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[viz] Guardado: {path}")


# ── 2. Heatmap de matriz de confusión mejorado ────────────────────────────────

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                          model_name: str = "Modelo",
                          filename: str = None) -> None:
    """
    Heatmap de la matriz de confusión con anotaciones y porcentajes.
    """
    os.makedirs(REPORTS_FIG_DIR, exist_ok=True)
    
    cm = confusion_matrix(y_true, y_pred)
    labels = sorted(np.unique(y_true))
    
    # Crear matriz de porcentajes
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    fig, ax = plt.subplots(figsize=(6, 5))
    
    # Anotaciones combinadas: número absoluto y porcentaje
    annot = np.empty_like(cm).astype(str)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f"{cm[i, j]}\n({cm_percent[i, j]:.1f}%)"
    
    sns.heatmap(cm, annot=annot, fmt='', cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax,
                linewidths=2, linecolor="white", cbar_kws={'label': 'Frecuencia'})
    
    ax.set_xlabel("Clase predicha", fontsize=12, fontweight='bold')
    ax.set_ylabel("Clase real", fontsize=12, fontweight='bold')
    ax.set_title(f"Matriz de confusión – {model_name}",
                 fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()

    fname = filename or f"cm_{model_name.lower().replace(' ', '_')}.png"
    path  = os.path.join(REPORTS_FIG_DIR, fname)
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[viz] Guardado: {path}")


# ── 3. Comparativa de métricas mejorada ───────────────────────────────────────

def plot_metrics_comparison(df_metrics: pd.DataFrame,
                            filename: str = "comparacion_modelos.png") -> None:
    """
    Barplot agrupado comparando Accuracy, Precision, Recall y F1 entre modelos.
    """
    os.makedirs(REPORTS_FIG_DIR, exist_ok=True)
    
    metrics = ["Accuracy", "Precision macro", "Recall macro", "F1-score macro"]
    x      = np.arange(len(df_metrics))
    width  = 0.18
    fig, ax = plt.subplots(figsize=(11, 6))

    for i, metric in enumerate(metrics):
        offset = (i - 1.5) * width
        rects  = ax.bar(x + offset, df_metrics[metric], width,
                        label=metric, color=PALETTE[i % len(PALETTE)],
                        alpha=0.85, edgecolor='black', linewidth=1)
        
        # Anotaciones con valores
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2,
                    height + 0.01,
                    f"{height:.3f}",
                    ha="center", va="bottom", fontsize=9, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(df_metrics["Modelo"], fontsize=11, fontweight='bold')
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Valor de la métrica", fontsize=12, fontweight='bold')
    ax.set_title("Comparación de métricas entre modelos de ensamble (Stratified 10-Fold CV)",
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc="upper right", fontsize=10, framealpha=0.9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()

    path = os.path.join(REPORTS_FIG_DIR, filename)
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[viz] Guardado: {path}")


# ── 4. Feature importance mejorado ────────────────────────────────────────────

def plot_feature_importance(scores: pd.DataFrame,
                            filename: str = "chi2_features.png") -> None:
    """
    Barplot horizontal de importancia de características según Chi2.
    """
    os.makedirs(REPORTS_FIG_DIR, exist_ok=True)
    
    top = scores.head(15).sort_values("Chi2 Score", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 6))
    
    bars = ax.barh(top["Feature"], top["Chi2 Score"], 
                   color="#4C72B0", alpha=0.85, edgecolor='black', linewidth=1)
    
    # Añadir valores al final de cada barra
    for i, (bar, val) in enumerate(zip(bars, top["Chi2 Score"])):
        ax.text(val + 0.5, i, f'{val:.2f}',
                va='center', ha='left', fontsize=9, fontweight='bold')
    
    ax.set_xlabel("Chi² Score", fontsize=12, fontweight='bold')
    ax.set_ylabel("Característica", fontsize=12, fontweight='bold')
    ax.set_title("Importancia de características (prueba Chi²)",
                 fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    plt.tight_layout()

    path = os.path.join(REPORTS_FIG_DIR, filename)
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[viz] Guardado: {path}")


# ── 5. Gráfico de comparación por clase ──────────────────────────────────────

def plot_per_class_metrics(results: list, 
                           filename: str = "metricas_por_clase.png") -> None:
    """
    Muestra precision, recall y f1-score por clase para cada modelo.
    """
    os.makedirs(REPORTS_FIG_DIR, exist_ok=True)
    
    from sklearn.metrics import precision_recall_fscore_support
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    metric_names = ['Precision', 'Recall', 'F1-Score']
    
    for idx, metric_name in enumerate(metric_names):
        ax = axes[idx]
        x = np.arange(len(results))
        width = 0.25
        
        # Obtener clases únicas desde las etiquetas reales
        classes = sorted(np.unique(results[0]['y_true']))

        for i, cls in enumerate(classes):
            values = []
            for r in results:
                p, rec, f1, _ = precision_recall_fscore_support(
                    r['y_true'], r['y_pred'], labels=[cls], zero_division=0
                )
                if idx == 0:
                    values.append(p[0])
                elif idx == 1:
                    values.append(rec[0])
                else:
                    values.append(f1[0])
            
            offset = (i - len(classes)/2 + 0.5) * width
            ax.bar(x + offset, values, width, label=f'Clase {cls}',
                   alpha=0.85, edgecolor='black', linewidth=1)
        
        ax.set_xlabel('Modelo', fontsize=11, fontweight='bold')
        ax.set_ylabel(metric_name, fontsize=11, fontweight='bold')
        ax.set_title(f'{metric_name} por clase', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([r['model'] for r in results])
        ax.legend()
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_ylim(0, 1.1)
    
    plt.tight_layout()
    path = os.path.join(REPORTS_FIG_DIR, filename)
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[viz] Guardado: {path}")


# ── 6. Curva ROC (Stacking) ──────────────────────────────────────────────────

def plot_roc_curve_stacking(result: dict,
                            filename: str = "roc_curve_stacking.png") -> None:
    """
    Curva ROC del modelo Stacking usando las probabilidades out-of-fold
    del Stratified 10-Fold CV. Solo aplica a clasificación binaria.
    """
    os.makedirs(REPORTS_FIG_DIR, exist_ok=True)

    pos_label = int(np.unique(result["y_true"]).max())
    fpr, tpr, _ = roc_curve(result["y_true"], result["y_pred_proba"][:, 1], pos_label=pos_label)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color=PALETTE[2], linewidth=2,
            label=f"Stacking (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Clasificador aleatorio")
    ax.set_xlabel("Tasa de falsos positivos (FPR)", fontsize=12, fontweight='bold')
    ax.set_ylabel("Tasa de verdaderos positivos (TPR)", fontsize=12, fontweight='bold')
    ax.set_title("Curva ROC – Stacking\n(Stratified 10-Fold CV, predicciones out-of-fold)",
                 fontsize=13, fontweight='bold', pad=15)
    ax.legend(loc="lower right", fontsize=10, framealpha=0.9)
    ax.grid(alpha=0.3, linestyle='--')
    plt.tight_layout()

    path = os.path.join(REPORTS_FIG_DIR, filename)
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[viz] Guardado: {path}")


# ── 7. Exportar tabla a CSV ───────────────────────────────────────────────────

def save_metrics_table(df_metrics: pd.DataFrame,
                       filename: str = "tabla_comparativa.csv") -> None:
    """Guarda la tabla comparativa de métricas en CSV."""
    os.makedirs(REPORTS_TAB_DIR, exist_ok=True)
    path = os.path.join(REPORTS_TAB_DIR, filename)
    df_metrics.to_csv(path, index=False, float_format="%.4f")
    print(f"[viz] Tabla guardada: {path}")
