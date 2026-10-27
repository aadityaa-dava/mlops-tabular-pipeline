"""Evaluation metrics for imbalanced fraud scoring."""

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def top_budget_stats(y_true, scores, budget: float = 0.01) -> tuple[float, float]:
    """Recall and precision when only the top `budget` fraction of scores is flagged."""
    y = np.asarray(y_true)
    s = np.asarray(scores)
    k = max(1, int(len(s) * budget))
    flagged = y[np.argsort(-s)[:k]]
    return float(flagged.sum() / y.sum()), float(flagged.mean())


def evaluate(y_true, scores, budget: float = 0.01) -> dict[str, float]:
    recall, precision = top_budget_stats(y_true, scores, budget)
    return {
        "pr_auc": float(average_precision_score(y_true, scores)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        f"recall_at_{budget:.0%}": recall,
        f"precision_at_{budget:.0%}": precision,
    }
