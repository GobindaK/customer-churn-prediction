from __future__ import annotations

from typing import Iterable, List


def classify_risk(probabilities: Iterable[float]) -> List[str]:
    """Assign LOW, MEDIUM, or HIGH risk based on churn probability."""
    labels: List[str] = []
    for value in probabilities:
        if value < 0.30:
            labels.append('LOW')
        elif value < 0.60:
            labels.append('MEDIUM')
        else:
            labels.append('HIGH')
    return labels
