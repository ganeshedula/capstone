"""Shared autoregressive decoding and token diagnostics for Base, DoLa, ENN."""
from __future__ import annotations

from typing import Dict, Optional
import numpy as np
import torch
import torch.nn.functional as F


@torch.inference_mode()
def next_token_distribution(model, input_ids, method: str, alpha: float = 1.0,
                            epinet=None, vocab_head_weight: Optional[np.ndarray] = None,
                            n_z_samples: int = 3):
    """Return final logits, selected DoLa layer, and uncertainty diagnostics."""
    selected_layer = None
    uncertainty = None
    if method == "base":
        logits = model(input_ids=input_ids, use_cache=False).logits
    else:
        from dola import dola_logits
        _, dola_raw, selected, mature, premature = dola_logits(model, input_ids, alpha=alpha)
        selected_layer = int(selected[0, -1])
        logits = dola_raw
        if method == "dola_enn":
            if epinet is None or vocab_head_weight is None:
                raise ValueError("DoLa+ENN requires a trained ENN checkpoint.")
            from enn_torch import enn_predict
            mature_np = mature[0].float().cpu().numpy()  # final state is already RMS-normalized
            premature_np = model.model.norm(premature)[0].float().cpu().numpy()
            enn_logits, epistemic, predictive, aleatoric = enn_predict(
                epinet, epinet, mature_np, premature_np, vocab_head_weight, n_samples=n_z_samples)
            logits = logits + torch.from_numpy(enn_logits).to(logits.device, logits.dtype).unsqueeze(0)
            uncertainty = {"epistemic_variance": float(epistemic[-1]),
                           "predictive_uncertainty": float(predictive[-1]),
                           "aleatoric_uncertainty": float(aleatoric[-1])}
    return logits[:, -1, :].float(), selected_layer, uncertainty


@torch.inference_mode()
def generate_answer(model, tokenizer, input_ids, method: str, alpha: float = 1.0,
                    epinet=None, vocab_head_weight=None, max_new_tokens: int = 48,
                    n_z_samples: int = 3) -> Dict:
    """Greedy decoding using the requested distribution at every token step."""
    generated = input_ids.clone(); first = None
    for _ in range(max_new_tokens):
        logits, layer, uncertainty = next_token_distribution(
            model, generated, method, alpha, epinet, vocab_head_weight, n_z_samples)
        probs = F.softmax(logits, dim=-1)
        confidence, token = probs.max(dim=-1)
        entropy = -(probs * probs.clamp_min(1e-10).log()).sum(dim=-1)
        if first is None:
            top_probs, top_ids = probs[0].topk(10)
            first = {"confidence": float(confidence.item()), "entropy": float(entropy.item()),
                     "selected_premature_layer": layer, "uncertainty": uncertainty,
                     "top_tokens": [(tokenizer.decode([i.item()]), float(p.item())) for i, p in zip(top_ids, top_probs)]}
        generated = torch.cat([generated, token[:, None]], dim=1)
        if token.item() == tokenizer.eos_token_id:
            break
    first["answer"] = tokenizer.decode(generated[0, input_ids.shape[1]:], skip_special_tokens=True).strip()
    return first
