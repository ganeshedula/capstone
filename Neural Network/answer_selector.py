"""Inspectable multi-signal answer scoring and abstention policy.

The selector is deliberately conservative: language-model fluency is useful as
one signal, but it must not outweigh factuality, agreement, and explicit risk
signals.  The functions mutate candidate dictionaries in place so the web and
CLI reporting paths keep receiving the per-candidate diagnostics they already
display.
"""
from __future__ import annotations

import math


# Token confidence is intentionally weaker than answer-level signals.  A model
# can be very confident while completing a false premise or repeating a fluent
# hallucination.
DEFAULT_WEIGHTS = {"semantic_consistency": .22, "factuality_score": .24,
                   "reliability_probability": .16, "token_confidence": .08,
                   "sequence_probability": .04, "claim_support": .10,
                   "relevance_score": .08, "evidence_support_score": .08}

DEFAULT_PENALTIES = {"contradiction": .20, "hallucination": .22, "entropy": .06,
                     "unsupported_claim": .14, "unsupported_evidence": .10}


def score_candidates(candidates, weights=None, penalty_weights=None):
    """Score candidates while keeping unavailable signals neutral.

    Missing metrics used to become zero-valued positives and were also charged
    as unsupported claims/evidence.  That made partially evaluated candidates
    look worse for reasons unrelated to their answer.  A missing positive
    signal is now neutral (0.5), and a penalty is applied only when the signal
    was actually computed.  Evidence support is only scored when
    ``evidence_available`` is true.
    """
    weights = dict(DEFAULT_WEIGHTS if weights is None else weights)
    penalties = dict(DEFAULT_PENALTIES)
    penalties.update(penalty_weights or {})
    _validate_weights(weights, "Scoring weights")
    _validate_weights(penalties, "Penalty weights")

    total = sum(float(value) for value in weights.values())
    for item in candidates:
        metrics = item.get("metrics") or {}
        components = {key: _metric(metrics, key, default=.5) for key in weights}
        positive_score = sum(float(weight) * components[key]
                             for key, weight in weights.items()) / total

        # Risk signals default to no penalty only when they were not produced.
        # This is different from an observed score of zero.
        penalty_parts = {
            "contradiction": _penalty_value(metrics, "contradiction_score", penalties["contradiction"]),
            "hallucination": _penalty_value(metrics, "hallucination_score", penalties["hallucination"]),
            "entropy": _penalty_value(metrics, "entropy_penalty", penalties["entropy"]),
        }
        if _has_metric(metrics, "claim_support"):
            penalty_parts["unsupported_claim"] = float(penalties["unsupported_claim"]) * (1 - components["claim_support"])
        else:
            penalty_parts["unsupported_claim"] = 0.0

        has_evidence = bool(metrics.get("evidence_available", False))
        if has_evidence and _has_metric(metrics, "evidence_support_score"):
            penalty_parts["unsupported_evidence"] = float(penalties["unsupported_evidence"]) * (1 - components["evidence_support_score"])
        else:
            penalty_parts["unsupported_evidence"] = 0.0

        penalty = sum(penalty_parts.values())
        item["final_score"] = _unit(positive_score - penalty, default=0.0)
        item["score_breakdown"] = {
            "positive_score": positive_score,
            "penalty": penalty,
            "penalties": penalty_parts,
            "components": components,
        }
    return candidates


def decide(candidates, reliability_threshold=.55, consistency_threshold=.4,
           hallucination_threshold=.55, factuality_threshold=.45):
    """Select the strongest trustworthy candidate, or request regeneration.

    The previous implementation ranked every candidate first and only then
    checked thresholds.  Consequently an unsafe high-scoring answer could be
    returned even when a safer candidate passed all gates.  We now rank the
    eligible pool first and fall back to the overall best candidate only when
    no candidate is eligible, so callers still have an answer to display while
    the decision is ``REGENERATE``.
    """
    if not candidates:
        return {"decision": "UNCERTAIN", "selected": None, "reason": "no candidates"}

    eligible = [item for item in candidates if not _gate_reasons(
        item, reliability_threshold, consistency_threshold,
        hallucination_threshold, factuality_threshold)]
    pool = eligible or candidates
    best = max(pool, key=_ranking_key)
    reasons = _gate_reasons(best, reliability_threshold, consistency_threshold,
                             hallucination_threshold, factuality_threshold)
    return {
        "decision": "ACCEPT" if not reasons else "REGENERATE",
        "selected": best,
        "reason": "; ".join(reasons) if reasons else "all configured thresholds passed",
    }


def _gate_reasons(item, reliability_threshold, consistency_threshold,
                  hallucination_threshold, factuality_threshold):
    metrics = item.get("metrics") or {}
    reasons = []
    _check_threshold(metrics, "reliability_probability", reliability_threshold,
                     "reliability", reasons, missing_default=0.0, below=True)
    _check_threshold(metrics, "semantic_consistency", consistency_threshold,
                     "semantic consensus", reasons, missing_default=0.0, below=True)
    # Factuality is a useful hard gate when the verifier produced it.  Keep
    # legacy callers that do not expose this newer metric compatible; the
    # aggregate score still treats the missing signal as neutral.
    if factuality_threshold is not None and _has_metric(metrics, "factuality_score"):
        _check_threshold(metrics, "factuality_score", factuality_threshold,
                         "factuality", reasons, missing_default=0.0, below=True)
    _check_threshold(metrics, "hallucination_score", hallucination_threshold,
                     "hallucination score", reasons, missing_default=1.0, below=False)
    return reasons


def _check_threshold(metrics, key, threshold, label, reasons, missing_default, below):
    if not _has_metric(metrics, key):
        reasons.append(f"{label} unavailable")
        return
    value = _metric(metrics, key, default=missing_default)
    if (value < threshold) if below else (value > threshold):
        reasons.append(f"{label} below threshold" if below else f"{label} above threshold")


def _ranking_key(item):
    """Use stable, safety-oriented tie breakers after the aggregate score."""
    metrics = item.get("metrics") or {}
    return (
        _metric(item, "final_score", default=0.0),
        _metric(metrics, "factuality_score"),
        -_metric(metrics, "hallucination_score"),
        _metric(metrics, "reliability_probability"),
        _metric(metrics, "semantic_consistency"),
        _metric(metrics, "claim_support"),
        _metric(metrics, "relevance_score"),
    )


def _penalty_value(metrics, key, weight):
    return float(weight) * _metric(metrics, key, default=0.0) if _has_metric(metrics, key) else 0.0


def _has_metric(metrics, key):
    value = metrics.get(key)
    try:
        return value is not None and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _metric(values, key, default=.5):
    value = values.get(key) if hasattr(values, "get") else None
    return _unit(value, default=default)


def _unit(value, default=.5):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return min(1.0, max(0.0, value)) if math.isfinite(value) else default


def _validate_weights(values, label):
    if not values:
        raise ValueError(f"{label} must be non-empty")
    try:
        numeric = [float(value) for value in values.values()]
    except (TypeError, ValueError):
        raise ValueError(f"{label} must be numeric") from None
    if any(not math.isfinite(value) or value < 0 for value in numeric) or sum(numeric) <= 0:
        raise ValueError(f"{label} must be non-negative with positive total")
