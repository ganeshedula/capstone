"""Simple claim segmentation and cross-candidate support diagnostics."""
from __future__ import annotations
import re


def split_claims(answer):
    # Sentence-level claims are a useful, inspectable approximation, not a full parser.
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+|\n+", str(answer)) if part.strip()]


def score_claims(answer, peer_answers, similarity_fn):
    claims = split_claims(answer)
    peers = [claim for peer in peer_answers for claim in split_claims(peer)]
    results = []
    for claim in claims:
        sims = [float(similarity_fn(claim, other)) for other in peers]
        support = max(sims, default=0.0)
        # A cross-candidate disagreement is a signal to inspect, not proof of falsity.
        contradiction = _negation_mismatch(claim, peers)
        results.append({"claim": claim, "claim_confidence": support,
                        "claim_support": support, "claim_contradiction": contradiction,
                        "claim_uncertainty": 1.0 - support})
    mean_support = sum(c["claim_support"] for c in results) / max(1, len(results))
    return {"claims": results, "claim_support": mean_support,
            "claim_contradiction": max((c["claim_contradiction"] for c in results), default=0.0),
            "claim_uncertainty": max((c["claim_uncertainty"] for c in results), default=1.0)}


def _negation_mismatch(claim, peers):
    terms = set(re.findall(r"\b\w+\b", claim.lower()))
    negated = bool(terms & {"not", "never", "no", "cannot", "doesn't", "isn't"})
    for peer in peers:
        other = set(re.findall(r"\b\w+\b", peer.lower()))
        if len(terms & other) >= 3:
            other_negated = bool(other & {"not", "never", "no", "cannot", "doesn't", "isn't"})
            if negated != other_negated: return 1.0
    return 0.0
