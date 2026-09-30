"""
TruthfulQA MC1 / MC2 evaluation — Corrected.

Modes:

    baseline   — Standard TinyLlama log-likelihood.
    dola       — DoLa contrastive log-probability scoring (corrected formula).
    dola+enn   — DoLa logits + ENN logits combined, then log_softmax.

All modes use the same answer-scoring and metric-computation code so that
comparisons are fair.
"""

import os
import sys
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm


_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


# ---------------------------------------------------------------------------
# Answer scoring
# ---------------------------------------------------------------------------

def _log_likelihood(
    model,
    tokenizer,
    question: str,
    answer: str,
    device: str,
    logit_fn: Optional[Callable] = None,
    already_log_scores: bool = False,
) -> float:
    """
    Calculate mean per-token answer log-probability.

    Prompt:   Q: {question}\\nA: {answer}
    Only answer tokens (after "A: ") contribute to the score.

    If already_log_scores=False  →  logit_fn returns raw logits  →  log_softmax applied.
    If already_log_scores=True   →  logit_fn already returns log-probs  →  used directly.
    """
    prompt   = f"Q: {question}\nA: {answer}"
    q_prefix = f"Q: {question}\nA: "

    enc = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    ).to(device)

    q_enc = tokenizer(
        q_prefix,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )

    input_ids = enc["input_ids"]       # (1, T)
    q_len     = q_enc["input_ids"].shape[1]

    # --- Get scores ---
    if logit_fn is not None:
        scores = logit_fn(input_ids).float()
        if already_log_scores:
            log_scores = scores
        else:
            log_scores = F.log_softmax(scores, dim=-1)
    else:
        with torch.inference_mode():
            out = model(input_ids=input_ids, use_cache=False)
            log_scores = F.log_softmax(out.logits.float(), dim=-1)

    log_scores = log_scores[0]  # (T, V)

    # --- Sum log-probs over answer tokens ---
    ans_start = q_len - 1        # position whose *next* token is the first answer token
    ans_end   = input_ids.shape[1] - 1

    if ans_end <= ans_start:
        return 0.0

    token_scores = []
    for pos in range(ans_start, ans_end):
        next_tok     = input_ids[0, pos + 1].item()
        token_score  = log_scores[pos, next_tok]
        token_scores.append(float(token_score.detach()))

    if not token_scores:
        return 0.0

    return float(np.mean(token_scores))


# ---------------------------------------------------------------------------
# Scoring functions for each mode
# ---------------------------------------------------------------------------

def _baseline_logit_fn(model, input_ids: torch.Tensor) -> torch.Tensor:
    """Standard LLM forward pass — returns raw logits."""
    with torch.inference_mode():
        return model(input_ids=input_ids, use_cache=False).logits


def _dola_logit_fn(
    model,
    input_ids: torch.Tensor,
    alpha: float = 1.0,
    fixed_premature_layer: Optional[int] = None,
    stats=None,
) -> torch.Tensor:
    """DoLa contrastive scoring — returns log-probabilities."""
    from dola import dola_logits

    contrastive_log_probs, _, _, _, _ = dola_logits(
        model, input_ids,
        alpha=alpha,
        fixed_premature_layer=fixed_premature_layer,
        stats=stats,
    )
    return contrastive_log_probs


def _dola_enn_logit_fn(
    model,
    input_ids: torch.Tensor,
    prior_params,
    learnable_params,
    vocab_head_weight_np: np.ndarray,
    alpha: float = 1.0,
    enn_weight: float = 0.1,
    stats=None,
) -> torch.Tensor:
    """
    DoLa + ENN combined scoring.

    combined_logits = DoLa_raw_logits + enn_weight × ENN_logits
    returns log_softmax(combined_logits)
    """
    from dola import dola_logits
    from enn_torch import enn_predict

    # DoLa forward pass
    _, contrastive_raw, _, mature_hidden, premature_hidden = dola_logits(
        model, input_ids, alpha=alpha, stats=stats,
    )

    # Post-RMSNorm features for ENN
    with torch.inference_mode():
        # ``mature_hidden`` is already the model's final RMS-normalized state.
        normed_mature = mature_hidden
        normed_premature = model.model.norm(premature_hidden)

    # Convert to numpy  (T, H)
    mat_np  = normed_mature[0].float().cpu().numpy()
    prem_np = normed_premature[0].float().cpu().numpy()

    # ENN inference
    enn_logits_np, epist_unc, pred_unc, alea_unc = enn_predict(
        prior_params, learnable_params,
        mat_np, prem_np,
        vocab_head_weight_np,
    )

    # Convert ENN logits to torch
    enn_logits = torch.from_numpy(enn_logits_np).to(
        device=contrastive_raw.device,
        dtype=contrastive_raw.dtype,
    ).unsqueeze(0)   # (1, T, V)

    # Combine:  DoLa raw logits + weighted ENN logits
    combined = contrastive_raw + enn_weight * enn_logits

    return F.log_softmax(combined, dim=-1)


# ---------------------------------------------------------------------------
# Main evaluation loop (shared by all modes)
# ---------------------------------------------------------------------------

def _run_eval(
    model,
    tokenizer,
    device: str,
    mode: str,
    n_questions: int,
    score_fn: Callable,
    start_idx: int = 0,
) -> Dict[str, float]:
    """
    Run TruthfulQA MC1 and MC2 evaluation.

    Args:
        score_fn  : callable(question, answer) → float
        start_idx : first question index (for train/test split)
    """
    dataset = load_dataset(
        "truthful_qa", "multiple_choice",
        trust_remote_code=True,
    )["validation"]

    end_idx = min(start_idx + n_questions, len(dataset))
    dataset = dataset.select(range(start_idx, end_idx))

    mc1_correct = 0
    mc2_ratio_sum = 0.0
    total = 0
    all_scores = []

    for item in tqdm(dataset, desc=f"[eval/{mode}]"):
        question = item["question"]

        # ---- MC1 ----
        mc1_targets = item["mc1_targets"]
        choices1    = mc1_targets["choices"]
        labels1     = mc1_targets["labels"]

        scores1  = [score_fn(question, c) for c in choices1]
        best_idx = int(np.argmax(scores1))
        if labels1[best_idx] == 1:
            mc1_correct += 1

        # ---- MC2 ----
        mc2_targets = item["mc2_targets"]
        choices2    = mc2_targets["choices"]
        labels2     = mc2_targets["labels"]

        scores2 = [score_fn(question, c) for c in choices2]

        correct_scores   = [s for s, l in zip(scores2, labels2) if l == 1]
        incorrect_scores = [s for s, l in zip(scores2, labels2) if l == 0]

        if correct_scores and incorrect_scores:
            n_correct = sum(
                1 for c in correct_scores
                for i in incorrect_scores
                if c > i
            )
            n_total = len(correct_scores) * len(incorrect_scores)
            mc2_ratio_sum += n_correct / n_total

        all_scores.extend(scores1)
        total += 1

    mc1 = mc1_correct / total if total > 0 else 0.0
    mc2 = mc2_ratio_sum / total if total > 0 else 0.0
    avg = float(np.mean(all_scores)) if all_scores else 0.0

    return {"mc1": mc1, "mc2": mc2, "n": total, "avg_score": avg}


# ---------------------------------------------------------------------------
# Public evaluation API
# ---------------------------------------------------------------------------

def evaluate_baseline(
    model, tokenizer, device: str,
    n_questions: int = 100,
    start_idx: int = 0,
) -> Dict[str, float]:
    """Baseline TinyLlama evaluation."""

    def score_fn(q, a):
        return _log_likelihood(
            model, tokenizer, q, a, device,
            logit_fn=lambda ids: _baseline_logit_fn(model, ids),
            already_log_scores=False,
        )

    return _run_eval(
        model, tokenizer, device,
        "baseline", n_questions, score_fn,
        start_idx=start_idx,
    )


def evaluate_dola(
    model, tokenizer, device: str,
    n_questions: int = 100,
    alpha: float = 1.0,
    fixed_premature_layer: Optional[int] = None,
    start_idx: int = 0,
    stats=None,
) -> Dict[str, float]:
    """DoLa contrastive evaluation."""
    from dola import DoLaStats
    if stats is None:
        stats = DoLaStats()

    def score_fn(q, a):
        return _log_likelihood(
            model, tokenizer, q, a, device,
            logit_fn=lambda ids: _dola_logit_fn(
                model, ids,
                alpha=alpha,
                fixed_premature_layer=fixed_premature_layer,
                stats=stats,
            ),
            already_log_scores=True,   # DoLa returns log_softmax
        )

    result = _run_eval(
        model, tokenizer, device,
        "dola", n_questions, score_fn,
        start_idx=start_idx,
    )
    result["stats"] = stats
    return result


def evaluate_dola_enn(
    model, tokenizer, device: str,
    prior_params, learnable_params,
    vocab_head_weight_np: np.ndarray,
    n_questions: int = 100,
    alpha: float = 1.0,
    enn_weight: float = 0.1,
    start_idx: int = 0,
    stats=None,
) -> Dict[str, float]:
    """DoLa + ENN combined evaluation."""
    from dola import DoLaStats
    if stats is None:
        stats = DoLaStats()

    def score_fn(q, a):
        return _log_likelihood(
            model, tokenizer, q, a, device,
            logit_fn=lambda ids: _dola_enn_logit_fn(
                model, ids,
                prior_params=prior_params,
                learnable_params=learnable_params,
                vocab_head_weight_np=vocab_head_weight_np,
                alpha=alpha,
                enn_weight=enn_weight,
                stats=stats,
            ),
            already_log_scores=True,   # returns log_softmax
        )

    result = _run_eval(
        model, tokenizer, device,
        "dola+enn", n_questions, score_fn,
        start_idx=start_idx,
    )
    result["stats"] = stats
    return result


# ---------------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------------

def print_results(label: str, results: Dict) -> None:
    """Print a single experiment result block."""
    sep = "=" * 56
    print(f"\n{sep}")
    print(f"  {label}")
    print(f"{sep}")
    print(f"  MC1 accuracy  : {results['mc1']:.4f}  ({results['mc1'] * 100:.1f}%)")
    print(f"  MC2 accuracy  : {results['mc2']:.4f}  ({results['mc2'] * 100:.1f}%)")
    print(f"  Questions     : {results['n']}")
    if "avg_score" in results:
        print(f"  Avg log-score : {results['avg_score']:.4f}")
    if "stats" in results:
        stats = results["stats"]
        if hasattr(stats, "summary"):
            summary = stats.summary()
            if summary:
                print(f"\n  Premature Layer Statistics:")
                print(summary)
    print(f"{sep}\n")


def print_experiment_summary(
    label: str,
    model_name: str = "TinyLlama-1.1B",
    mature_layer: int = 21,
    premature_layer: str = "dynamic",
    alpha: float = 1.0,
    enn_weight: float = 0.0,
    enn_epochs: int = 0,
    n_questions: int = 100,
    results: Optional[Dict] = None,
) -> None:
    """Print a comprehensive experiment summary."""
    sep = "=" * 56
    print(f"\n{sep}")
    print(f"  EXPERIMENT: {label}")
    print(f"{sep}")
    print(f"  Model           : {model_name}")
    print(f"  Mature layer    : {mature_layer}")
    print(f"  Premature layer : {premature_layer}")
    print(f"  Alpha           : {alpha}")
    print(f"  ENN weight      : {enn_weight}")
    print(f"  ENN epochs      : {enn_epochs}")
    print(f"  Questions       : {n_questions}")
    if results:
        print(f"  MC1             : {results['mc1']:.4f}  ({results['mc1'] * 100:.1f}%)")
        print(f"  MC2             : {results['mc2']:.4f}  ({results['mc2'] * 100:.1f}%)")
        if "avg_score" in results:
            print(f"  Avg score       : {results['avg_score']:.4f}")
    print(f"{sep}\n")
