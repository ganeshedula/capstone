"""
C4-based token-level evaluation module.

Evaluates DoLa-only vs DoLa+ENN scoring on *held-out* C4 texts that were
NOT used during ENN training.  This gives a meaningful before/after
comparison because the ENN was trained to improve token-level predictions
on C4-style data.

Metrics computed per-token:
  - Log-likelihood of the true next token
  - Top-1 and Top-5 accuracy
  - Perplexity
  - ENN uncertainty decomposition (epistemic / aleatoric / predictive)
  - Calibration: does high ENN uncertainty correlate with actual errors?
"""

import json
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

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

from dola import dola_logits, DoLaStats, extract_dola_features
from enn_torch import enn_predict


# ---------------------------------------------------------------------------
# Load held-out C4 texts
# ---------------------------------------------------------------------------

def load_held_out_c4_texts(
    n_eval: int = 100,
    n_skip: int = 600,
    min_text_len: int = 50,
) -> List[str]:
    """
    Stream C4 texts from HuggingFace, skipping the first `n_skip` samples
    (which were used for ENN training) and collecting the next `n_eval`.

    Args:
        n_eval      : number of held-out samples to collect
        n_skip      : number of training samples to skip over
        min_text_len: minimum character length for a text to be included

    Returns:
        List of text strings
    """
    from datasets import load_dataset

    print(f"[c4-eval] Streaming C4 data — skipping first {n_skip}, "
          f"collecting {n_eval} held-out samples...")

    ds = load_dataset(
        "allenai/c4", "en",
        split="train",
        streaming=True,
        trust_remote_code=True,
    )

    texts: List[str] = []
    skipped = 0
    seen = 0

    for example in ds:
        text = example.get("text", "").strip()
        if len(text) < min_text_len:
            continue

        seen += 1
        if seen <= n_skip:
            skipped += 1
            continue

        texts.append(text)
        if len(texts) >= n_eval:
            break

    print(f"[c4-eval] Skipped {skipped} training samples, "
          f"collected {len(texts)} held-out evaluation samples")
    return texts


# ---------------------------------------------------------------------------
# Per-text token-level evaluation
# ---------------------------------------------------------------------------

def evaluate_text_tokens(
    model,
    tokenizer,
    device: str,
    text: str,
    alpha: float = 1.0,
    enn_weight: float = 0.1,
    prior_params=None,
    learnable_params=None,
    vh_weight_np: Optional[np.ndarray] = None,
    max_length: int = 128,
    skip_first_fraction: float = 0.20,
) -> Dict[str, Any]:
    """
    Evaluate a single C4 text at the token level.

    Returns metrics for both DoLa-only (before) and DoLa+ENN (after).
    """
    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    ).to(device)

    input_ids = enc["input_ids"]    # (1, T)
    seq_len = input_ids.shape[1]

    if seq_len < 4:
        return None

    skip_n = max(1, int(seq_len * skip_first_fraction))

    # ---- DoLa forward pass (shared by both modes) ----
    stats = DoLaStats()
    contrastive_log_probs, contrastive_raw, _, mature_hidden, premature_hidden = dola_logits(
        model, input_ids, alpha=alpha, stats=stats,
    )

    # ---- BEFORE: DoLa-only scoring ----
    dola_log_probs = contrastive_log_probs[0].float()   # (T, V)

    before_lls = []
    before_top1_hits = []
    before_top5_hits = []

    for pos in range(skip_n, seq_len - 1):
        next_tok = input_ids[0, pos + 1].item()

        # Log-likelihood of true token
        ll = dola_log_probs[pos, next_tok].item()
        before_lls.append(ll)

        # Top-1 accuracy
        pred_tok = torch.argmax(dola_log_probs[pos]).item()
        before_top1_hits.append(1 if pred_tok == next_tok else 0)

        # Top-5 accuracy
        top5 = torch.topk(dola_log_probs[pos], 5).indices.tolist()
        before_top5_hits.append(1 if next_tok in top5 else 0)

    # ---- AFTER: DoLa + ENN scoring ----
    has_enn = (prior_params is not None and
               learnable_params is not None and
               vh_weight_np is not None)

    after_lls = []
    after_top1_hits = []
    after_top5_hits = []
    enn_epistemic = []
    enn_aleatoric = []
    enn_predictive = []
    token_errors_after = []  # 1 if top-1 wrong, 0 if correct

    if has_enn:
        # Post-RMSNorm for ENN
        with torch.inference_mode():
            normed_mature = mature_hidden
            normed_premature = model.model.norm(premature_hidden)

        mat_np = normed_mature[0].float().cpu().numpy()    # (T, H)
        prem_np = normed_premature[0].float().cpu().numpy()  # (T, H)

        # ENN inference
        enn_logits_np, epist_unc, pred_unc, alea_unc = enn_predict(
            prior_params, learnable_params,
            mat_np, prem_np,
            vh_weight_np,
        )

        # Combined logits
        enn_logits = torch.from_numpy(enn_logits_np).to(
            device=contrastive_raw.device,
            dtype=contrastive_raw.dtype,
        ).unsqueeze(0)  # (1, T, V)

        combined = contrastive_raw + enn_weight * enn_logits
        combined_log_probs = F.log_softmax(combined, dim=-1)[0].float()  # (T, V)

        for pos in range(skip_n, seq_len - 1):
            next_tok = input_ids[0, pos + 1].item()

            # Log-likelihood
            ll = combined_log_probs[pos, next_tok].item()
            after_lls.append(ll)

            # Top-1 accuracy
            pred_tok = torch.argmax(combined_log_probs[pos]).item()
            after_top1_hits.append(1 if pred_tok == next_tok else 0)
            token_errors_after.append(0 if pred_tok == next_tok else 1)

            # Top-5 accuracy
            top5 = torch.topk(combined_log_probs[pos], 5).indices.tolist()
            after_top5_hits.append(1 if next_tok in top5 else 0)

            # ENN uncertainty for this position
            enn_epistemic.append(float(epist_unc[pos]))
            enn_aleatoric.append(float(alea_unc[pos]))
            enn_predictive.append(float(pred_unc[pos]))

    n_tokens = len(before_lls)
    if n_tokens == 0:
        return None

    result = {
        "n_tokens": n_tokens,
        "seq_len": seq_len,
        "text_preview": text[:100],

        # Before (DoLa-only)
        "before_mean_ll": float(np.mean(before_lls)),
        "before_perplexity": float(np.exp(-np.mean(before_lls))),
        "before_top1_accuracy": float(np.mean(before_top1_hits)),
        "before_top5_accuracy": float(np.mean(before_top5_hits)),
        "before_lls": before_lls,
        "before_top1_hits": before_top1_hits,
        "before_top5_hits": before_top5_hits,
    }

    if has_enn:
        result.update({
            # After (DoLa + ENN)
            "after_mean_ll": float(np.mean(after_lls)),
            "after_perplexity": float(np.exp(-np.mean(after_lls))),
            "after_top1_accuracy": float(np.mean(after_top1_hits)),
            "after_top5_accuracy": float(np.mean(after_top5_hits)),
            "after_lls": after_lls,
            "after_top1_hits": after_top1_hits,
            "after_top5_hits": after_top5_hits,

            # Changes
            "ll_change": float(np.mean(after_lls) - np.mean(before_lls)),
            "perplexity_change": float(np.exp(-np.mean(after_lls)) - np.exp(-np.mean(before_lls))),
            "top1_change": float(np.mean(after_top1_hits) - np.mean(before_top1_hits)),
            "top5_change": float(np.mean(after_top5_hits) - np.mean(before_top5_hits)),

            # ENN uncertainty
            "mean_epistemic": float(np.mean(enn_epistemic)),
            "mean_aleatoric": float(np.mean(enn_aleatoric)),
            "mean_predictive": float(np.mean(enn_predictive)),
            "enn_epistemic": enn_epistemic,
            "enn_aleatoric": enn_aleatoric,
            "enn_predictive": enn_predictive,
            "token_errors_after": token_errors_after,
        })

    return result


# ---------------------------------------------------------------------------
# Full C4 evaluation pipeline
# ---------------------------------------------------------------------------

def run_c4_evaluation(
    model,
    tokenizer,
    device: str,
    n_eval: int = 100,
    n_skip: int = 600,
    alpha: float = 1.0,
    enn_weight: float = 0.1,
    prior_params=None,
    learnable_params=None,
    vh_weight_np: Optional[np.ndarray] = None,
    max_length: int = 128,
) -> Tuple[List[Dict], Dict[str, Any]]:
    """
    Run the full C4 token-level evaluation.

    Returns:
        per_text_results : list of dicts, one per C4 text
        aggregate        : dict of aggregate metrics
    """
    texts = load_held_out_c4_texts(n_eval=n_eval, n_skip=n_skip)

    per_text_results = []
    desc = "[c4-eval] Evaluating"

    for text in tqdm(texts, desc=desc):
        r = evaluate_text_tokens(
            model, tokenizer, device, text,
            alpha=alpha,
            enn_weight=enn_weight,
            prior_params=prior_params,
            learnable_params=learnable_params,
            vh_weight_np=vh_weight_np,
            max_length=max_length,
        )
        if r is not None:
            per_text_results.append(r)

    aggregate = compute_c4_aggregate(per_text_results)
    return per_text_results, aggregate


def compute_c4_aggregate(
    results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Compute aggregate metrics across all C4 texts."""
    if not results:
        return {}

    # Flatten all token-level data
    all_before_lls = []
    all_before_top1 = []
    all_before_top5 = []
    all_after_lls = []
    all_after_top1 = []
    all_after_top5 = []
    all_epistemic = []
    all_aleatoric = []
    all_predictive = []
    all_errors = []

    for r in results:
        all_before_lls.extend(r["before_lls"])
        all_before_top1.extend(r["before_top1_hits"])
        all_before_top5.extend(r["before_top5_hits"])
        if "after_lls" in r:
            all_after_lls.extend(r["after_lls"])
            all_after_top1.extend(r["after_top1_hits"])
            all_after_top5.extend(r["after_top5_hits"])
            all_epistemic.extend(r["enn_epistemic"])
            all_aleatoric.extend(r["enn_aleatoric"])
            all_predictive.extend(r["enn_predictive"])
            all_errors.extend(r["token_errors_after"])

    n_texts = len(results)
    n_tokens = len(all_before_lls)

    agg = {
        "n_texts": n_texts,
        "n_tokens": n_tokens,

        # Before (DoLa-only)
        "before_mean_ll": float(np.mean(all_before_lls)),
        "before_std_ll": float(np.std(all_before_lls)),
        "before_perplexity": float(np.exp(-np.mean(all_before_lls))),
        "before_top1_accuracy": float(np.mean(all_before_top1)),
        "before_top5_accuracy": float(np.mean(all_before_top5)),

        # Per-text perplexities (for scatter plot)
        "before_per_text_ppl": [r["before_perplexity"] for r in results],
    }

    if all_after_lls:
        agg.update({
            # After (DoLa + ENN)
            "after_mean_ll": float(np.mean(all_after_lls)),
            "after_std_ll": float(np.std(all_after_lls)),
            "after_perplexity": float(np.exp(-np.mean(all_after_lls))),
            "after_top1_accuracy": float(np.mean(all_after_top1)),
            "after_top5_accuracy": float(np.mean(all_after_top5)),

            # Changes
            "ll_change": float(np.mean(all_after_lls) - np.mean(all_before_lls)),
            "perplexity_change": float(np.exp(-np.mean(all_after_lls))
                                       - np.exp(-np.mean(all_before_lls))),
            "top1_change": float(np.mean(all_after_top1) - np.mean(all_before_top1)),
            "top5_change": float(np.mean(all_after_top5) - np.mean(all_before_top5)),

            # ENN uncertainty
            "mean_epistemic": float(np.mean(all_epistemic)),
            "mean_aleatoric": float(np.mean(all_aleatoric)),
            "mean_predictive": float(np.mean(all_predictive)),

            # Per-text perplexities (for scatter plot)
            "after_per_text_ppl": [r["after_perplexity"] for r in results],

            # Calibration data (raw arrays for binning in plots)
            "_calibration_epistemic": all_epistemic,
            "_calibration_errors": all_errors,
        })

    return agg


# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------

def print_c4_results(
    aggregate: Dict[str, Any],
) -> None:
    """Print a formatted summary of C4 evaluation results."""
    sep = "=" * 80
    print(f"\n{sep}")
    print("  C4 TOKEN-LEVEL EVALUATION RESULTS")
    print(f"{sep}")

    print(f"\n  Texts evaluated : {aggregate['n_texts']}")
    print(f"  Tokens evaluated: {aggregate['n_tokens']}")

    # Perplexity
    print(f"\n  {'Metric':<32} {'Before (DoLa)':>14} {'After (DoLa+ENN)':>18} {'Change':>12}")
    print(f"  {'-'*32} {'-'*14} {'-'*18} {'-'*12}")

    metrics = [
        ("Perplexity ↓", "before_perplexity", "after_perplexity", "perplexity_change"),
        ("Mean Log-Likelihood ↑", "before_mean_ll", "after_mean_ll", "ll_change"),
        ("Top-1 Accuracy ↑", "before_top1_accuracy", "after_top1_accuracy", "top1_change"),
        ("Top-5 Accuracy ↑", "before_top5_accuracy", "after_top5_accuracy", "top5_change"),
    ]

    for label, bk, ak, ck in metrics:
        bv = aggregate.get(bk, 0.0)
        av = aggregate.get(ak, None)
        cv = aggregate.get(ck, None)

        if av is not None:
            if "Accuracy" in label:
                print(f"  {label:<32} {bv*100:>13.2f}% {av*100:>17.2f}% {cv*100:>+11.2f}%")
            else:
                print(f"  {label:<32} {bv:>14.4f} {av:>18.4f} {cv:>+12.4f}")
        else:
            if "Accuracy" in label:
                print(f"  {label:<32} {bv*100:>13.2f}% {'N/A':>18} {'N/A':>12}")
            else:
                print(f"  {label:<32} {bv:>14.4f} {'N/A':>18} {'N/A':>12}")

    # ENN uncertainty
    if aggregate.get("mean_epistemic") is not None:
        print(f"\n  {'ENN Uncertainty (After only)':<32} {'Value':>14}")
        print(f"  {'-'*32} {'-'*14}")
        print(f"  {'Mean Epistemic':<32} {aggregate['mean_epistemic']:>14.4f}")
        print(f"  {'Mean Aleatoric':<32} {aggregate['mean_aleatoric']:>14.4f}")
        print(f"  {'Mean Predictive':<32} {aggregate['mean_predictive']:>14.4f}")

    print(f"\n{sep}\n")


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------

def save_c4_results(
    per_text_results: List[Dict[str, Any]],
    aggregate: Dict[str, Any],
    output_dir: str,
    args_dict: Optional[Dict] = None,
) -> str:
    """Save C4 evaluation results to JSON."""
    os.makedirs(output_dir, exist_ok=True)

    # Strip large arrays from per_text_results for JSON (keep summaries)
    compact_results = []
    for r in per_text_results:
        compact = {k: v for k, v in r.items()
                   if k not in ("before_lls", "before_top1_hits", "before_top5_hits",
                                "after_lls", "after_top1_hits", "after_top5_hits",
                                "enn_epistemic", "enn_aleatoric", "enn_predictive",
                                "token_errors_after")}
        compact_results.append(compact)

    # Strip calibration arrays from aggregate for JSON
    agg_clean = {k: v for k, v in aggregate.items()
                 if not k.startswith("_")}

    output = {
        "timestamp": datetime.now().isoformat(),
        "pipeline": "c4-token-eval",
        "args": args_dict or {},
        "aggregate_metrics": agg_clean,
        "per_text_results": compact_results,
    }

    path = os.path.join(output_dir, "c4_eval_results.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"[c4-eval] Results saved → {path}")
    return path
