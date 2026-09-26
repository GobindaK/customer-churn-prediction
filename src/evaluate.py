from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score


def compute_classification_metrics(y_true: Iterable[int], y_pred: Iterable[int], y_prob: Iterable[float]) -> Dict[str, float]:
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_true, y_prob),
    }
    return metrics


def save_confusion_matrix(y_true: Iterable[int], y_pred: Iterable[int], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm, cmap='Blues')
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', color='black')
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_model_comparison(metrics: List[Dict[str, float]], model_names: List[str], output_path: str | Path) -> pd.DataFrame:
    comparison = pd.DataFrame(metrics, index=model_names)
    comparison = comparison[['accuracy', 'precision', 'recall', 'f1', 'roc_auc']]
    comparison.to_csv(output_path, index_label='Model')
    return comparison
