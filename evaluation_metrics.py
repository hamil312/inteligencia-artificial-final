"""
Módulo de Métricas para Clasificación de Lesiones Cutáneas
===========================================================
Proporciona evaluación completa: matriz de confusión, curvas ROC,
sensibilidad/especificidad, y gráficas de entrenamiento.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    roc_auc_score, precision_score, recall_score, f1_score,
    accuracy_score
)

# =============================================================================
# CONFIGURACIÓN VISUAL
# =============================================================================
CLASS_NAMES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
COLORS = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c', '#e67e22']

# =============================================================================
# 1. MÉTRICAS POR CLASE (sensibilidad, especificidad, precisión, F1)
# =============================================================================
def per_class_metrics(y_true, y_pred, class_names=CLASS_NAMES):
    """
    Calcula sensibilidad (recall), especificidad, precisión y F1 por clase.

    Sensibilidad = VP / (VP + FN)  — qué tan bien detecta la clase
    Especificidad = VN / (VN + FP) — qué tan bien evita falsos positivos
    """
    n_classes = len(class_names)
    metrics = {}
    for i, name in enumerate(class_names):
        # Binario: 1 si es esta clase, 0 si no
        y_true_bin = (y_true == i).astype(int)
        y_pred_bin = (y_pred == i).astype(int)

        tp = np.sum((y_true_bin == 1) & (y_pred_bin == 1))
        tn = np.sum((y_true_bin == 0) & (y_pred_bin == 0))
        fp = np.sum((y_true_bin == 0) & (y_pred_bin == 1))
        fn = np.sum((y_true_bin == 1) & (y_pred_bin == 0))

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0.0

        metrics[name] = {
            'sensibilidad': sensitivity,
            'especificidad': specificity,
            'precision': precision,
            'f1_score': f1,
            'vp': int(tp), 'vn': int(tn), 'fp': int(fp), 'fn': int(fn)
        }
    return metrics


def print_per_class_metrics(metrics, class_names=CLASS_NAMES):
    """Imprime tabla formateada de métricas por clase."""
    header = f"{'Clase':<8} {'Sensibilidad':>12} {'Especificidad':>13} {'Precisión':>10} {'F1-Score':>9} {'VP':>4} {'VN':>4} {'FP':>4} {'FN':>4}"
    sep = "-" * len(header)
    print("\n" + sep)
    print("MÉTRICAS DETALLADAS POR CLASE")
    print(sep)
    print(header)
    print(sep)
    for name in class_names:
        m = metrics[name]
        print(f"{name:<8} {m['sensibilidad']:>12.4f} {m['especificidad']:>13.4f} {m['precision']:>10.4f} {m['f1_score']:>9.4f} {m['vp']:>4d} {m['vn']:>4d} {m['fp']:>4d} {m['fn']:>4d}")
    print(sep)
    print(f"{'Promedio':<8} {np.mean([m['sensibilidad'] for m in metrics.values()]):>12.4f} {np.mean([m['especificidad'] for m in metrics.values()]):>13.4f} {np.mean([m['precision'] for m in metrics.values()]):>10.4f} {np.mean([m['f1_score'] for m in metrics.values()]):>9.4f}")
    print(sep + "\n")


# =============================================================================
# 2. MATRIZ DE CONFUSIÓN (visual + numérica)
# =============================================================================
def plot_confusion_matrix(y_true, y_pred, class_names=CLASS_NAMES, save_path='confusion_matrix.png'):
    """Matriz de confusión con formato numérico y porcentual."""
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Absolutos
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                xticklabels=class_names, yticklabels=class_names,
                annot_kws={'size': 11})
    ax1.set_xlabel('Predicción')
    ax1.set_ylabel('Real')
    ax1.set_title('Matriz de Confusión (valores absolutos)')

    # Normalizados (porcentajes por fila)
    sns.heatmap(cm_norm, annot=True, fmt='.1%', cmap='Blues', ax=ax2,
                xticklabels=class_names, yticklabels=class_names,
                annot_kws={'size': 11})
    ax2.set_xlabel('Predicción')
    ax2.set_ylabel('Real')
    ax2.set_title('Matriz de Confusión (porcentaje por clase real)')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[MÉTRICAS] Matriz de confusión guardada: {save_path}")
    plt.show()
    return cm, cm_norm


# =============================================================================
# 3. CURVAS ROC Y AUC
# =============================================================================
def plot_roc_curves(y_true_onehot, y_prob, class_names=CLASS_NAMES, save_path='roc_curves.png'):
    """
    Curvas ROC para cada clase + macro + micro promedio.
    y_true_onehot: one-hot encoding (N, n_classes)
    y_prob: probabilidades predichas (N, n_classes)
    """
    n_classes = len(class_names)
    fpr = {}
    tpr = {}
    roc_auc = {}

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_onehot[:, i], y_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Micro promedio
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_onehot.ravel(), y_prob.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Macro promedio
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Gráfica
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Clasificador aleatorio')

    for i in range(n_classes):
        ax.plot(fpr[i], tpr[i], color=COLORS[i], lw=1.5,
                label=f'{class_names[i]} (AUC = {roc_auc[i]:.3f})')

    ax.plot(fpr["micro"], tpr["micro"], 'k-', lw=2.5,
            label=f'Micro-promedio (AUC = {roc_auc["micro"]:.3f})')
    ax.plot(fpr["macro"], tpr["macro"], 'k:', lw=2.5,
            label=f'Macro-promedio (AUC = {roc_auc["macro"]:.3f})')

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel('Tasa de Falsos Positivos (1 - Especificidad)')
    ax.set_ylabel('Tasa de Verdaderos Positivos (Sensibilidad)')
    ax.set_title('Curvas ROC por Clase')
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[MÉTRICAS] Curvas ROC guardadas: {save_path}")
    plt.show()

    # Imprimir AUCs
    print("\n" + "-" * 45)
    print("  AUC (Área Bajo la Curva ROC)")
    print("-" * 45)
    for i in range(n_classes):
        print(f"  {class_names[i]:<8}  AUC = {roc_auc[i]:.4f}")
    print("-" * 45)
    print(f"  Micro     AUC = {roc_auc['micro']:.4f}")
    print(f"  Macro     AUC = {roc_auc['macro']:.4f}")
    print("-" * 45 + "\n")

    return roc_auc


# =============================================================================
# 4. GRÁFICA DE ENTRENAMIENTO (loss + accuracy)
# =============================================================================
def plot_training_history(history, save_path='training_history.png'):
    """Gráfica de pérdida y precisión durante entrenamiento."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history.history['loss']) + 1)

    ax1.plot(epochs, history.history['loss'], 'b-', label='Entrenamiento')
    ax1.plot(epochs, history.history['val_loss'], 'r-', label='Validación')
    ax1.set_xlabel('Épocas')
    ax1.set_ylabel('Pérdida')
    ax1.set_title('Pérdida durante el entrenamiento')
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(epochs, history.history['accuracy'], 'b-', label='Entrenamiento')
    ax2.plot(epochs, history.history['val_accuracy'], 'r-', label='Validación')
    ax2.set_xlabel('Épocas')
    ax2.set_ylabel('Precisión')
    ax2.set_title('Precisión durante el entrenamiento')
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[MÉTRICAS] Historial de entrenamiento guardado: {save_path}")
    plt.show()


# =============================================================================
# 5. GRÁFICA DE COMPARACIÓN DE MÉTRICAS POR CLASE
# =============================================================================
def plot_metrics_comparison(metrics, class_names=CLASS_NAMES, save_path='metrics_comparison.png'):
    """Gráfica de barras comparando sensibilidad, especificidad, precisión, F1."""
    x = np.arange(len(class_names))
    width = 0.2

    sens = [metrics[c]['sensibilidad'] for c in class_names]
    spec = [metrics[c]['especificidad'] for c in class_names]
    prec = [metrics[c]['precision'] for c in class_names]
    f1 = [metrics[c]['f1_score'] for c in class_names]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - 1.5*width, sens, width, label='Sensibilidad', color='#3498db')
    ax.bar(x - 0.5*width, spec, width, label='Especificidad', color='#2ecc71')
    ax.bar(x + 0.5*width, prec, width, label='Precisión', color='#f39c12')
    ax.bar(x + 1.5*width, f1, width, label='F1-Score', color='#e74c3c')

    ax.set_xlabel('Clase')
    ax.set_ylabel('Valor')
    ax.set_title('Métricas por Clase')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names)
    ax.legend()
    ax.set_ylim([0, 1.05])
    ax.grid(alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[MÉTRICAS] Comparación guardada: {save_path}")
    plt.show()


# =============================================================================
# 6. EVALUACIÓN COMPLETA (ejecuta todo)
# =============================================================================
def evaluate_full(y_true, y_pred, y_prob, class_names=CLASS_NAMES,
                  history=None, model_name='modelo',
                  y_true_onehot=None, save_prefix=''):
    """
    Evalúa un modelo de clasificación con todas las métricas.

    Parámetros:
      y_true: etiquetas reales (enteros)
      y_pred: etiquetas predichas (enteros)
      y_prob: probabilidades predichas (N, n_classes)
      history: historial de entrenamiento (opcional)
      model_name: nombre para mostrar
      y_true_onehot: one-hot de y_true (si es None, se calcula)
      save_prefix: prefijo para archivos de salida
    """
    print(f"\n{'=' * 60}")
    print(f"  EVALUACIÓN COMPLETA: {model_name}")
    print(f"{'=' * 60}")

    # --- Reporte de clasificación ---
    print(f"\n>>> Reporte de Clasificación:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    # --- Accuracy ---
    acc = accuracy_score(y_true, y_pred)
    print(f"  Accuracy (global): {acc:.4f} ({acc*100:.2f}%)")

    # --- Métricas por clase ---
    metrics = per_class_metrics(y_true, y_pred, class_names)
    print_per_class_metrics(metrics, class_names)

    # --- Matriz de confusión ---
    cm_path = f"{save_prefix}{model_name.replace(' ', '_')}_confusion.png" if save_prefix else f"{model_name.replace(' ', '_')}_confusion.png"
    plot_confusion_matrix(y_true, y_pred, class_names, save_path=cm_path)

    # --- Curvas ROC ---
    if y_true_onehot is None:
        from tensorflow.keras.utils import to_categorical
        y_true_onehot = to_categorical(y_true, num_classes=len(class_names))
    roc_path = f"{save_prefix}{model_name.replace(' ', '_')}_roc.png" if save_prefix else f"{model_name.replace(' ', '_')}_roc.png"
    roc_auc = plot_roc_curves(y_true_onehot, y_prob, class_names, save_path=roc_path)

    # --- Comparación de métricas ---
    comp_path = f"{save_prefix}{model_name.replace(' ', '_')}_comparison.png" if save_prefix else f"{model_name.replace(' ', '_')}_comparison.png"
    plot_metrics_comparison(metrics, class_names, save_path=comp_path)

    # --- Historial de entrenamiento ---
    if history is not None:
        hist_path = f"{save_prefix}{model_name.replace(' ', '_')}_history.png" if save_prefix else f"{model_name.replace(' ', '_')}_history.png"
        plot_training_history(history, save_path=hist_path)

    # --- Resumen numérico final ---
    print(f"\n{'=' * 60}")
    print(f"  RESUMEN - {model_name}")
    print(f"{'=' * 60}")
    print(f"  Accuracy:            {acc:.4f}")
    print(f"  Sensibilidad (prom): {np.mean([m['sensibilidad'] for m in metrics.values()]):.4f}")
    print(f"  Especificidad (prom):{np.mean([m['especificidad'] for m in metrics.values()]):.4f}")
    print(f"  Precisión (prom):    {np.mean([m['precision'] for m in metrics.values()]):.4f}")
    print(f"  F1-Score (prom):     {np.mean([m['f1_score'] for m in metrics.values()]):.4f}")
    print(f"  AUC Macro:           {roc_auc['macro']:.4f}")
    print(f"  AUC Micro:           {roc_auc['micro']:.4f}")
    print(f"{'=' * 60}\n")

    return {
        'accuracy': acc,
        'per_class': metrics,
        'roc_auc': roc_auc,
        'confusion_matrix': confusion_matrix(y_true, y_pred)
    }
