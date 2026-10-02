"""Metrics for answer correctness and hallucination detection."""
from __future__ import annotations
import numpy as np


def binary_metrics(labels, probs, bins=10):
    y = np.asarray(labels, dtype=int); p = np.clip(np.asarray(probs, dtype=float), 1e-8, 1-1e-8)
    pred = p >= .5
    tp = int(np.sum(pred & (y == 1))); fp = int(np.sum(pred & (y == 0)))
    fn = int(np.sum(~pred & (y == 1))); tn = int(np.sum(~pred & (y == 0)))
    ece = 0.0
    for lo in np.linspace(0, 1, bins+1)[:-1]:
        mask = (p >= lo) & (p < lo+1/bins if lo < 1-1/bins else p <= 1)
        if mask.any(): ece += mask.mean()*abs(p[mask].mean()-y[mask].mean())
    precision = tp/max(1,tp+fp); recall=tp/max(1,tp+fn)
    return {"accuracy": float((tp+tn)/max(1,len(y))), "precision": precision,
            "recall": recall, "f1": 2*precision*recall/max(1e-12,precision+recall),
            "hallucination_detection_rate": recall,
            "false_positive_rate": fp/max(1,fp+tn), "false_negative_rate": fn/max(1,fn+tp),
            "ece": float(ece), "brier_score": float(np.mean((p-y)**2)),
            "average_confidence": float(np.mean(p)), "tp": tp, "fp": fp, "fn": fn, "tn": tn}
