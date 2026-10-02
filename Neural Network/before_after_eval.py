"""
Before vs After Training evaluation module.

Evaluates user-provided questions against the model:
  1. Before ENN training  (DoLa-only scoring)
  2. After ENN training   (DoLa + ENN scoring)

Reuses the existing log-likelihood scoring from evaluate.py and
uncertainty decomposition from enn_torch.py.  Does NOT modify any
existing methodology.
"""

import json
import os
import sys
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from evaluate import _log_likelihood, _dola_logit_fn, _dola_enn_logit_fn, _encode_question_answer
from dola import DoLaStats


# ---------------------------------------------------------------------------
# Question loading & validation
# ---------------------------------------------------------------------------

def load_eval_questions(path: str) -> List[Dict[str, Any]]:
    """
    Load evaluation questions from a JSON file.

    Expected format:
    [
      {
        "question": "What is ...?",
        "answer": "...",
        "distractors": ["...", "..."]   // optional
      },
      ...
    ]

    Returns a validated list of question dicts.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Evaluation questions file not found: {path}\n"
            f"Create a JSON file with format:\n"
            f'[{{"question": "...", "answer": "...", "distractors": ["...", "..."]}}]'
        )

    with open(path, "r", encoding="utf-8") as f:
        questions = json.load(f)

    if not isinstance(questions, list) or len(questions) == 0:
        raise ValueError(
            "Evaluation questions file must be a non-empty JSON array."
        )

    validated = []
    for i, q in enumerate(questions):
        if "question" not in q or "answer" not in q:
            raise ValueError(
                f"Question {i}: must have 'question' and 'answer' fields."
            )
        entry = {
            "id": i + 1,
            "question": str(q["question"]).strip(),
            "answer": str(q["answer"]).strip(),
            "distractors": [str(d).strip() for d in q.get("distractors", [])],
        }
        validated.append(entry)

    return validated


# ---------------------------------------------------------------------------
# Scoring functions
# ---------------------------------------------------------------------------

def score_answer_log_likelihood(
    model,
    tokenizer,
    device: str,
    question: str,
    answer: str,
    logit_fn: Optional[Callable] = None,
    already_log_scores: bool = False,
) -> float:
    """
    Compute mean per-token log-probability for the given answer.

    Wraps the existing _log_likelihood function from evaluate.py.
    """
    return _log_likelihood(
        model, tokenizer, question, answer, device,
        logit_fn=logit_fn,
        already_log_scores=already_log_scores,
    )


def score_mc_accuracy(
    model,
    tokenizer,
    device: str,
    question: str,
    correct_answer: str,
    distractors: List[str],
    logit_fn: Optional[Callable] = None,
    already_log_scores: bool = False,
) -> Tuple[bool, List[float]]:
    """
    Multiple-choice style accuracy: is the correct answer ranked #1?

    Returns:
        (is_correct, all_scores)  where all_scores[0] is the correct answer score.
    """
    choices = [correct_answer] + distractors

    scores = []
    for choice in choices:
        s = _log_likelihood(
            model, tokenizer, question, choice, device,
            logit_fn=logit_fn,
            already_log_scores=already_log_scores,
        )
        scores.append(s)

    best_idx = int(np.argmax(scores))
    is_correct = (best_idx == 0)  # correct answer is at index 0

    return is_correct, scores


def compute_answer_entropy(
    model,
    tokenizer,
    device: str,
    question: str,
    answer: str,
    logit_fn: Optional[Callable] = None,
    already_log_scores: bool = False,
) -> float:
    """
    Compute the mean per-token entropy of the model's output distribution
    over the answer tokens using the scoring function for the current phase.
    """
    input_ids, answer_start, answer_end = _encode_question_answer(
        tokenizer, question, answer, device,
    )

    with torch.inference_mode():
        scores = (logit_fn(input_ids) if logit_fn is not None
                  else model(input_ids=input_ids, use_cache=False).logits)
        log_probs = scores.float() if already_log_scores else F.log_softmax(scores.float(), dim=-1)
        probs = log_probs.exp()

    probs = probs[0]  # (T, V)

    if answer_end <= answer_start:
        return 0.0

    entropies = []
    for token_index in range(answer_start, answer_end):
        pos = token_index - 1
        p = probs[pos]
        entropy = -torch.sum(p * torch.log(p.clamp_min(1e-10))).item()
        entropies.append(entropy)

    return float(np.mean(entropies)) if entropies else 0.0


def generate_free_form_answer(
    model,
    tokenizer,
    device: str,
    question: str,
    max_new_tokens: int = 64,
    logit_fn: Optional[Callable] = None,
) -> str:
    """
    Generate a free-form answer for display purposes.

    Uses greedy decoding for reproducibility.
    """
    from load_llama2 import build_prompt

    prompt = build_prompt(tokenizer, question)
    inputs = tokenizer(
        prompt, return_tensors="pt",
        truncation=True, max_length=384,
    ).to(device)

    if logit_fn is None:
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    else:
        # Route every decoding step through the same method being evaluated.
        generated = inputs["input_ids"]
        with torch.inference_mode():
            for _ in range(max_new_tokens):
                scores = logit_fn(generated).float()
                token = scores[:, -1, :].argmax(dim=-1, keepdim=True)
                generated = torch.cat([generated, token], dim=1)
                if tokenizer.eos_token_id is not None and token.item() == tokenizer.eos_token_id:
                    break
        generated_tokens = generated[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Per-question evaluation
# ---------------------------------------------------------------------------

def evaluate_single_question(
    model,
    tokenizer,
    device: str,
    q: Dict[str, Any],
    logit_fn: Optional[Callable],
    already_log_scores: bool,
    mode_label: str,
    prior_params=None,
    learnable_params=None,
    vh_weight_np: Optional[np.ndarray] = None,
    alpha: float = 1.0,
    enn_weight: float = 0.1,
) -> Dict[str, Any]:
    """
    Evaluate a single question and return all metrics.

    Args:
        mode_label: "before" or "after"
        prior_params, learnable_params, vh_weight_np: ENN params (only for after)
    """
    question = q["question"]
    answer = q["answer"]
    distractors = q["distractors"]

    result: Dict[str, Any] = {
        "id": q["id"],
        "question": question,
        "expected_answer": answer,
        "mode": mode_label,
    }

    # --- Log-likelihood of correct answer ---
    ll = score_answer_log_likelihood(
        model, tokenizer, device,
        question, answer,
        logit_fn=logit_fn,
        already_log_scores=already_log_scores,
    )
    result["log_likelihood"] = ll
    result["confidence"] = float(np.exp(ll))  # probability proxy

    # --- MC accuracy (if distractors available) ---
    if distractors:
        is_correct, all_scores = score_mc_accuracy(
            model, tokenizer, device,
            question, answer, distractors,
            logit_fn=logit_fn,
            already_log_scores=already_log_scores,
        )
        result["mc_correct"] = is_correct
        result["mc_scores"] = all_scores
        result["has_mc"] = True
    else:
        result["mc_correct"] = None
        result["has_mc"] = False

    # --- Entropy ---
    entropy = compute_answer_entropy(
        model, tokenizer, device, question, answer,
        logit_fn=logit_fn, already_log_scores=already_log_scores,
    )
    result["entropy"] = entropy

    # --- Free-form generated answer ---
    gen = generate_free_form_answer(model, tokenizer, device, question, logit_fn=logit_fn)
    result["generated_answer"] = gen

    # --- ENN uncertainty (only available after training) ---
    if prior_params is not None and learnable_params is not None and vh_weight_np is not None:
        try:
            enn_unc = _compute_enn_uncertainty(
                model, tokenizer, device,
                question, answer,
                prior_params, learnable_params,
                vh_weight_np, alpha,
            )
            result.update(enn_unc)
        except Exception as e:
            result["epistemic_uncertainty"] = None
            result["aleatoric_uncertainty"] = None
            result["predictive_uncertainty"] = None
            result["enn_uncertainty_error"] = str(e)
    else:
        result["epistemic_uncertainty"] = None
        result["aleatoric_uncertainty"] = None
        result["predictive_uncertainty"] = None

    return result


def _compute_enn_uncertainty(
    model, tokenizer, device,
    question: str, answer: str,
    prior_params, learnable_params,
    vh_weight_np: np.ndarray,
    alpha: float,
) -> Dict[str, float]:
    """Compute ENN uncertainty metrics for a question-answer pair."""
    from dola import dola_logits
    from enn_torch import enn_predict

    prompt = f"Q: {question}\nA: {answer}"
    enc = tokenizer(
        prompt, return_tensors="pt",
        truncation=True, max_length=512,
    ).to(device)

    input_ids = enc["input_ids"]

    # Run DoLa forward to get hidden states
    _, _, _, mature_hidden, premature_hidden = dola_logits(
        model, input_ids, alpha=alpha,
    )

    # Post-RMSNorm
    with torch.inference_mode():
        normed_mature = mature_hidden
        normed_premature = model.model.norm(premature_hidden)

    mat_np = normed_mature[0].float().cpu().numpy()
    prem_np = normed_premature[0].float().cpu().numpy()

    # ENN inference with uncertainty
    _, epistemic, predictive, aleatoric = enn_predict(
        prior_params, learnable_params,
        mat_np, prem_np,
        vh_weight_np,
    )

    return {
        "epistemic_uncertainty": float(np.mean(epistemic)),
        "aleatoric_uncertainty": float(np.mean(aleatoric)),
        "predictive_uncertainty": float(np.mean(predictive)),
    }


# ---------------------------------------------------------------------------
# Full before/after pipeline
# ---------------------------------------------------------------------------

def run_before_after_evaluation(
    model,
    tokenizer,
    device: str,
    questions: List[Dict[str, Any]],
    alpha: float = 1.0,
    enn_weight: float = 0.1,
    prior_params=None,
    learnable_params=None,
    vh_weight_np: Optional[np.ndarray] = None,
    phase: str = "before",
) -> List[Dict[str, Any]]:
    """
    Evaluate all questions for a single phase (before or after).

    Args:
        phase: "before" (DoLa-only) or "after" (DoLa+ENN)
    """
    results = []

    if phase == "before":
        # DoLa-only scoring
        stats = DoLaStats()

        def logit_fn(ids):
            return _dola_logit_fn(model, ids, alpha=alpha, stats=stats)

        already_log = True
        p_params = None
        l_params = None
        vh = None
    else:
        # DoLa + ENN scoring
        if prior_params is None or learnable_params is None:
            raise ValueError(
                "After-training evaluation requires ENN checkpoint. "
                "Train the ENN first."
            )
        stats = DoLaStats()

        def logit_fn(ids):
            return _dola_enn_logit_fn(
                model, ids,
                prior_params=prior_params,
                learnable_params=learnable_params,
                vocab_head_weight_np=vh_weight_np,
                alpha=alpha,
                enn_weight=enn_weight,
                stats=stats,
            )

        already_log = True
        p_params = prior_params
        l_params = learnable_params
        vh = vh_weight_np

    desc = f"[eval/{phase}]"
    for q in tqdm(questions, desc=desc):
        r = evaluate_single_question(
            model, tokenizer, device, q,
            logit_fn=logit_fn,
            already_log_scores=already_log,
            mode_label=phase,
            prior_params=p_params,
            learnable_params=l_params,
            vh_weight_np=vh,
            alpha=alpha,
            enn_weight=enn_weight,
        )
        results.append(r)

    return results


# ---------------------------------------------------------------------------
# Results aggregation
# ---------------------------------------------------------------------------

def compute_aggregate_metrics(
    results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Compute summary statistics for a set of question results."""
    n = len(results)
    if n == 0:
        return {}

    agg: Dict[str, Any] = {"n_questions": n}

    # Log-likelihood
    lls = [r["log_likelihood"] for r in results]
    agg["mean_log_likelihood"] = float(np.mean(lls))
    agg["std_log_likelihood"] = float(np.std(lls))

    # Confidence
    confs = [r["confidence"] for r in results]
    agg["mean_confidence"] = float(np.mean(confs))

    # Entropy
    ents = [r["entropy"] for r in results]
    agg["mean_entropy"] = float(np.mean(ents))
    agg["std_entropy"] = float(np.std(ents))

    # MC accuracy
    mc_results = [r for r in results if r.get("has_mc")]
    if mc_results:
        n_correct = sum(1 for r in mc_results if r["mc_correct"])
        n_mc = len(mc_results)
        agg["mc_accuracy"] = n_correct / n_mc
        agg["mc_correct"] = n_correct
        agg["mc_total"] = n_mc
    else:
        agg["mc_accuracy"] = None

    # ENN uncertainty (only meaningful for "after" phase)
    epist = [r["epistemic_uncertainty"] for r in results
             if r.get("epistemic_uncertainty") is not None]
    if epist:
        agg["mean_epistemic_uncertainty"] = float(np.mean(epist))
        agg["mean_aleatoric_uncertainty"] = float(np.mean(
            [r["aleatoric_uncertainty"] for r in results
             if r.get("aleatoric_uncertainty") is not None]
        ))
        agg["mean_predictive_uncertainty"] = float(np.mean(
            [r["predictive_uncertainty"] for r in results
             if r.get("predictive_uncertainty") is not None]
        ))
    else:
        agg["mean_epistemic_uncertainty"] = None
        agg["mean_aleatoric_uncertainty"] = None
        agg["mean_predictive_uncertainty"] = None

    return agg


def build_comparison_table(
    before_results: List[Dict[str, Any]],
    after_results: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Build a per-question comparison structure."""
    table = []
    for br, ar in zip(before_results, after_results):
        row = {
            "id": br["id"],
            "question": br["question"],
            "expected_answer": br["expected_answer"],
            # Before
            "before_log_likelihood": br["log_likelihood"],
            "before_confidence": br["confidence"],
            "before_entropy": br["entropy"],
            "before_mc_correct": br.get("mc_correct"),
            "before_generated": br.get("generated_answer", ""),
            # After
            "after_log_likelihood": ar["log_likelihood"],
            "after_confidence": ar["confidence"],
            "after_entropy": ar["entropy"],
            "after_mc_correct": ar.get("mc_correct"),
            "after_generated": ar.get("generated_answer", ""),
            # ENN uncertainty (after only)
            "after_epistemic": ar.get("epistemic_uncertainty"),
            "after_aleatoric": ar.get("aleatoric_uncertainty"),
            "after_predictive": ar.get("predictive_uncertainty"),
            # Changes
            "ll_change": ar["log_likelihood"] - br["log_likelihood"],
            "confidence_change": ar["confidence"] - br["confidence"],
            "entropy_change": ar["entropy"] - br["entropy"],
        }
        table.append(row)
    return table


# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------

def print_comparison_table(
    comparison: List[Dict[str, Any]],
    before_agg: Dict[str, Any],
    after_agg: Dict[str, Any],
) -> None:
    """Print a formatted comparison table to the console."""
    sep = "=" * 90
    print(f"\n{sep}")
    print("  BEFORE vs AFTER TRAINING — PER-QUESTION RESULTS")
    print(f"{sep}")

    # Header
    print(f"  {'Q#':<4} {'Question':<35} {'Before':>12} {'After':>12} {'Change':>10}")
    print(f"  {'-'*4} {'-'*35} {'-'*12} {'-'*12} {'-'*10}")

    for row in comparison:
        q_short = row["question"][:33] + ".." if len(row["question"]) > 35 else row["question"]

        if row["before_mc_correct"] is not None:
            before_str = "✓ Correct" if row["before_mc_correct"] else "✗ Wrong"
            after_str = "✓ Correct" if row["after_mc_correct"] else "✗ Wrong"
            change_str = ""
            if row["before_mc_correct"] != row["after_mc_correct"]:
                change_str = "IMPROVED" if row["after_mc_correct"] else "REGRESSED"
        else:
            before_str = f"{row['before_log_likelihood']:.3f}"
            after_str = f"{row['after_log_likelihood']:.3f}"
            change_str = f"{row['ll_change']:+.3f}"

        print(f"  Q{row['id']:<3} {q_short:<35} {before_str:>12} {after_str:>12} {change_str:>10}")

    # Summary
    print(f"\n{sep}")
    print("  AGGREGATE METRICS")
    print(f"{sep}")

    if before_agg.get("mc_accuracy") is not None:
        ba = before_agg["mc_accuracy"] * 100
        aa = after_agg["mc_accuracy"] * 100
        diff = aa - ba
        print(f"  MC Accuracy (Before) : {ba:.1f}%  ({before_agg['mc_correct']}/{before_agg['mc_total']})")
        print(f"  MC Accuracy (After)  : {aa:.1f}%  ({after_agg['mc_correct']}/{after_agg['mc_total']})")
        print(f"  Accuracy Change      : {diff:+.1f}%")

    print(f"\n  {'Metric':<28} {'Before':>12} {'After':>12} {'Change':>10}")
    print(f"  {'-'*28} {'-'*12} {'-'*12} {'-'*10}")

    metrics = [
        ("Mean Log-Likelihood", "mean_log_likelihood"),
        ("Mean Confidence", "mean_confidence"),
        ("Mean Entropy", "mean_entropy"),
    ]
    for label, key in metrics:
        bv = before_agg.get(key, 0.0) or 0.0
        av = after_agg.get(key, 0.0) or 0.0
        print(f"  {label:<28} {bv:>12.4f} {av:>12.4f} {av - bv:>+10.4f}")

    # ENN-only metrics
    enn_metrics = [
        ("Mean Epistemic Unc.", "mean_epistemic_uncertainty"),
        ("Mean Aleatoric Unc.", "mean_aleatoric_uncertainty"),
        ("Mean Predictive Unc.", "mean_predictive_uncertainty"),
    ]
    has_enn = after_agg.get("mean_epistemic_uncertainty") is not None
    if has_enn:
        print(f"\n  {'ENN Uncertainty (After only)':<28} {'N/A':>12} {'Value':>12}")
        print(f"  {'-'*28} {'-'*12} {'-'*12}")
        for label, key in enn_metrics:
            av = after_agg.get(key, 0.0) or 0.0
            print(f"  {label:<28} {'—':>12} {av:>12.4f}")

    print(f"\n{sep}")

    # Generated answers
    print(f"\n{sep}")
    print("  GENERATED ANSWERS COMPARISON")
    print(f"{sep}")
    for row in comparison:
        print(f"\n  Q{row['id']}: {row['question']}")
        print(f"  Expected : {row['expected_answer']}")
        print(f"  Before   : {row['before_generated'][:120]}")
        print(f"  After    : {row['after_generated'][:120]}")
    print(f"\n{sep}\n")


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------

def save_results(
    before_results: List[Dict[str, Any]],
    after_results: List[Dict[str, Any]],
    before_agg: Dict[str, Any],
    after_agg: Dict[str, Any],
    comparison: List[Dict[str, Any]],
    output_dir: str,
    train_losses: Optional[List[float]] = None,
    val_losses: Optional[List[float]] = None,
    args_dict: Optional[Dict] = None,
) -> str:
    """Save all results to a JSON file for reproducibility."""
    os.makedirs(output_dir, exist_ok=True)

    output = {
        "timestamp": datetime.now().isoformat(),
        "pipeline": "before-after-eval",
        "args": args_dict or {},
        "before_training": {
            "method": "DoLa-only",
            "aggregate_metrics": before_agg,
            "per_question": before_results,
        },
        "after_training": {
            "method": "DoLa + ENN",
            "aggregate_metrics": after_agg,
            "per_question": after_results,
        },
        "comparison": comparison,
        "training": {
            "train_losses": train_losses or [],
            "val_losses": val_losses or [],
        },
    }

    path = os.path.join(output_dir, "results.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"[before-after] Results saved → {path}")
    return path
