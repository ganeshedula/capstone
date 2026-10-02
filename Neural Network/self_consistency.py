"""Reusable candidate normalization and self-consistency voting utilities."""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Sequence


_ANSWER_MARKERS = re.compile(
    r"(?:^|\n)\s*(?:final answer|answer)\s*[:\-]\s*", re.IGNORECASE
)
_LEAD_INS = re.compile(
    r"^(?:the answer is|therefore[, ]+|thus[, ]+|so[, ]+|final answer[: ]+)\s*",
    re.IGNORECASE,
)


def extract_final_answer(text: str) -> str:
    """Extract an explicit final/answer line where present; otherwise retain text."""
    raw = str(text or "").strip()
    matches = []
    for line in raw.splitlines():
        match = _ANSWER_MARKERS.search(line)
        if match:
            matches.append(line[match.end():].strip())
    if matches:
        text = matches[-1]
    else:
        text = " ".join(raw.split())
        text = _LEAD_INS.sub("", text).strip()
    return " ".join(text.strip(" \t\r\n.,;:").split())


def normalize_candidate(text: str) -> str:
    """Normalize superficial formatting while preserving meaningful wording."""
    answer = extract_final_answer(text)
    answer = answer.casefold()
    answer = re.sub(r"\s+", " ", answer)
    answer = re.sub(r"\s*([,.;:!?])\s*", r"\1 ", answer).strip()
    answer = re.sub(r"\s+", " ", answer)
    return answer.strip(" .,!?:;")


def self_consistency_vote(candidates: Sequence[str]) -> dict:
    """Plurality vote over normalized candidates; tied maxima abstain."""
    if not candidates:
        raise ValueError("self_consistency_vote requires at least one candidate")
    keys = [normalize_candidate(c) for c in candidates]
    if any(not k for k in keys):
        raise ValueError("Candidates must contain a non-empty answer")
    counts = Counter(keys)
    top_count = max(counts.values())
    winners = [key for key, count in counts.items() if count == top_count]
    winner = winners[0] if len(winners) == 1 else None
    display = {}
    for candidate, key in zip(candidates, keys):
        display.setdefault(key, extract_final_answer(candidate))
    return {"final_answer": display[winner] if winner is not None else None,
            "vote_counts": dict(counts),
            "agreement_ratio": (top_count / len(candidates)),
            "normalized_winner": winner, "tie": winner is None}


def confidence_weighted_vote(candidates: Sequence[str], confidences: Sequence[float]) -> dict:
    """Vote with non-negative confidence weights; confidence is not a calibrated P(correct)."""
    if len(candidates) != len(confidences) or not candidates:
        raise ValueError("candidates and confidences must have the same non-zero length")
    keys = [normalize_candidate(c) for c in candidates]
    if any(not k for k in keys):
        raise ValueError("Candidates must contain a non-empty answer")
    weights = [float(c) for c in confidences]
    if any(not (0 <= c <= 1) for c in weights):
        raise ValueError("Each confidence weight must be in [0, 1]")
    scores = defaultdict(float)
    counts = Counter()
    display = {}
    for candidate, key, weight in zip(candidates, keys, weights):
        scores[key] += weight
        counts[key] += 1
        display.setdefault(key, extract_final_answer(candidate))
    top_score = max(scores.values())
    winners = [key for key, score in scores.items() if abs(score-top_score) <= 1e-12]
    winner = winners[0] if len(winners) == 1 else None
    total = sum(scores.values())
    return {"final_answer": display[winner] if winner is not None else None,
            "weighted_scores": dict(scores),
            "normalized_confidence": (top_score / total if total > 0 else 0.0),
            "agreement_ratio": (counts[winner] / len(candidates) if winner is not None else 0.0),
            "normalized_winner": winner, "tie": winner is None}
