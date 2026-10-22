import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Training scripts save figures to disk; no display needed
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from .config import RESULTS_DIR, IMAGES_DIR

def evaluate(name, display_name, y_true, y_pred, class_names, sources, extra=None):
    """Prints the test report and saves metrics (results/<name>.json) and a confusion matrix image."""
    y_true, y_pred, sources = np.asarray(y_true), np.asarray(y_pred), np.asarray(sources)
    labels = list(range(len(class_names)))

    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro')

    # Accuracy per source dataset shows whether one corpus dominates the overall number
    per_source = {
        source: float(accuracy_score(y_true[sources == source], y_pred[sources == source]))
        for source in sorted(set(sources))
    }

    print(f"\n--- {display_name}: Test Evaluation ---")
    print(f"Accuracy: {accuracy * 100:.2f}% | Macro F1: {macro_f1 * 100:.2f}%")
    print("Per-dataset accuracy: " + ", ".join(f"{s}={a * 100:.1f}%" for s, a in per_source.items()))
    print(classification_report(y_true, y_pred, labels=labels, target_names=class_names, zero_division=0))

    metrics = {
        'model': display_name,
        'test_samples': int(len(y_true)),
        'accuracy': float(accuracy),
        'macro_f1': float(macro_f1),
        'per_source_accuracy': per_source,
        'classification_report': classification_report(
            y_true, y_pred, labels=labels, target_names=class_names, output_dict=True, zero_division=0
        ),
        'confusion_matrix': confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        'class_names': list(class_names)
    }
    if extra:
        metrics.update(extra)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(RESULTS_DIR / f'{name}.json', 'w') as f:
        json.dump(metrics, f, indent=4)

    plot_confusion_matrix(name, display_name, metrics['confusion_matrix'], class_names)
    return metrics

def plot_confusion_matrix(name, display_name, matrix, class_names):
    os.makedirs(IMAGES_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', linecolor='white', linewidths=0.5,
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_title(f'{display_name} Confusion Matrix (Test Set)')
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('Actual Label')
    plt.tight_layout()
    fig.savefig(IMAGES_DIR / f'{name}_confusion_matrix.png', dpi=120)
    plt.close(fig)

def plot_history(name, display_name, history):
    """Saves train/val loss and accuracy curves to images/<name>_loss_accuracy_graph.png."""
    os.makedirs(IMAGES_DIR, exist_ok=True)
    epochs_range = range(1, len(history['loss']) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(16, 5))

    ax[0].plot(epochs_range, history['loss'], label='Training Loss')
    ax[0].plot(epochs_range, history['val_loss'], label='Validation Loss')
    ax[0].set_title(f'{display_name}: Loss')
    ax[0].set_xlabel('Epochs')
    ax[0].legend()

    ax[1].plot(epochs_range, history['accuracy'], label='Training Accuracy')
    ax[1].plot(epochs_range, history['val_accuracy'], label='Validation Accuracy')
    ax[1].set_title(f'{display_name}: Accuracy')
    ax[1].set_xlabel('Epochs')
    ax[1].legend()

    plt.tight_layout()
    fig.savefig(IMAGES_DIR / f'{name}_loss_accuracy_graph.png', dpi=120)
    plt.close(fig)
