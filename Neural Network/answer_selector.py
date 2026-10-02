"""Inspectable multi-signal answer scoring and abstention policy."""
from __future__ import annotations

DEFAULT_WEIGHTS = {"semantic_consistency": .25, "factuality_score": .20,
                   "reliability_probability": .15, "token_confidence": .15,
                   "sequence_probability": .10, "claim_support": .10,
                   "relevance_score": .05, "evidence_support_score": .05}


def score_candidates(candidates, weights=None, penalty_weights=None):
    weights = dict(DEFAULT_WEIGHTS if weights is None else weights)
    penalties = {"contradiction": .18, "hallucination": .18, "entropy": .08,
                 "unsupported_claim": .12, "unsupported_evidence": .10}
    penalties.update(penalty_weights or {})
    if not weights or any(float(v) < 0 for v in weights.values()) or sum(weights.values()) <= 0:
        raise ValueError("Scoring weights must be non-negative with positive total")
    total = sum(weights.values())
    for item in candidates:
        metrics = item["metrics"]
        score = sum(float(w) * _unit(metrics.get(k, 0.0)) for k, w in weights.items()) / total
        penalty = (float(penalties["contradiction"]) * _unit(metrics.get("contradiction_score", 0.0))
                   + float(penalties["hallucination"]) * _unit(metrics.get("hallucination_score", 0.0))
                   + float(penalties["entropy"]) * _unit(metrics.get("entropy_penalty", 0.0))
                   + float(penalties["unsupported_claim"]) * (1-_unit(metrics.get("claim_support", 0.0)))
                   + float(penalties["unsupported_evidence"]) * (1-_unit(metrics.get("evidence_support_score", 0.5))))
        item["final_score"] = min(1.0, max(0.0, score-penalty))
    return candidates


def decide(candidates, reliability_threshold=.55, consistency_threshold=.4,
           hallucination_threshold=.55):
    if not candidates: return {"decision": "UNCERTAIN", "selected": None, "reason": "no candidates"}
    best = max(candidates, key=lambda item: item["final_score"])
    m = best["metrics"]
    reasons = []
    if m.get("reliability_probability", 0.0) < reliability_threshold: reasons.append("reliability below threshold")
    if m.get("semantic_consistency", 0.0) < consistency_threshold: reasons.append("semantic consensus below threshold")
    if m.get("hallucination_score", 1.0) > hallucination_threshold: reasons.append("hallucination score above threshold")
    # A judge formatting failure is not evidence that the answer is false. The
    # verifier supplies a conservative fallback and reports FALLBACK separately.
    return {"decision": "REGENERATE" if reasons else "ACCEPT", "selected": best,
            "reason": "; ".join(reasons) if reasons else "all configured thresholds passed"}


def _unit(value): return min(1.0, max(0.0, float(value)))
