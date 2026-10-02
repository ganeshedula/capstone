"""Shared autoregressive decoding and token diagnostics for Base, DoLa, ENN."""
from __future__ import annotations

from typing import Dict, Optional
import numpy as np
import torch
import torch.nn.functional as F


@torch.inference_mode()
def next_token_distribution(model, input_ids, method: str, alpha: float = 1.0,
                            epinet=None, vocab_head_weight: Optional[np.ndarray] = None,
                            n_z_samples: int = 3, enn_weight: float = 0.0,
                            past_key_values=None, return_cache: bool = False):
    """Return final logits, selected DoLa layer, and uncertainty diagnostics."""
    selected_layer = None
    uncertainty = None
    if method == "base":
        out = model(input_ids=input_ids, past_key_values=past_key_values, use_cache=True)
        logits = out.logits
        new_cache = out.past_key_values
    else:
        from dola import dola_step_logits
        dola_raw, selected, mature, premature, new_cache = dola_step_logits(
            model, input_ids, past_key_values=past_key_values, alpha=alpha)
        selected_layer = (selected[:, -1].detach().cpu().tolist() if selected.shape[0] > 1
                          else int(selected[0, -1]))
        logits = dola_raw
        if method == "dola_enn":
            if epinet is None or vocab_head_weight is None:
                raise ValueError("DoLa+ENN requires a trained ENN checkpoint.")
            from enn_torch import enn_predict
            mature_np = mature[0].float().cpu().numpy()  # final state is already RMS-normalized
            premature_np = model.model.norm(premature)[0].float().cpu().numpy()
            enn_logits, epistemic, predictive, aleatoric = enn_predict(
                epinet, epinet, mature_np, premature_np, vocab_head_weight, n_samples=n_z_samples)
            logits = logits + enn_weight * torch.from_numpy(enn_logits).to(logits.device, logits.dtype).unsqueeze(0)
            uncertainty = {"epistemic_variance": float(epistemic[-1]),
                           "predictive_uncertainty": float(predictive[-1]),
                           "aleatoric_uncertainty": float(aleatoric[-1])}
    result = (logits[:, -1, :].float(), selected_layer, uncertainty)
    return (*result, new_cache) if return_cache else result


@torch.inference_mode()
def generate_answer(model, tokenizer, input_ids, method: str, alpha: float = 1.0,
                    epinet=None, vocab_head_weight=None, max_new_tokens: int = 48,
                    n_z_samples: int = 3, enn_weight: float = 0.0,
                    do_sample: bool = False, temperature: float = 1.0,
                    top_p: float = 1.0, generator=None, seed: Optional[int] = None) -> Dict:
    """Decode from the requested distribution; supports reproducible sampling."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")
    if seed is not None:
        generator = torch.Generator(device=input_ids.device).manual_seed(int(seed))
    generated = input_ids.clone(); first = None
    token_confidences, token_entropies, token_log_probs, layer_trace, topk_mass = [], [], [], [], []
    step_input = input_ids
    past_key_values = None
    for _ in range(max_new_tokens):
        logits, layer, uncertainty, past_key_values = next_token_distribution(
            model, step_input, method, alpha, epinet, vocab_head_weight, n_z_samples,
            enn_weight=enn_weight, past_key_values=past_key_values, return_cache=True)
        logits = logits / temperature if do_sample else logits
        if do_sample and top_p < 1:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            sorted_probs = F.softmax(sorted_logits, dim=-1)
            remove = sorted_probs.cumsum(dim=-1) - sorted_probs > top_p
            sorted_logits[remove] = -float("inf")
            logits = torch.full_like(logits, -float("inf")).scatter(-1, sorted_indices, sorted_logits)
        probs = F.softmax(logits, dim=-1)
        confidence, greedy_token = probs.max(dim=-1)
        token = torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1) if do_sample else greedy_token
        entropy = -(probs * probs.clamp_min(1e-10).log()).sum(dim=-1)
        selected_probability = probs.gather(-1, token[:, None]).squeeze(-1)
        token_confidences.append(float(selected_probability.item()))
        token_log_probs.append(float(selected_probability.clamp_min(1e-30).log().item()))
        token_entropies.append(float(entropy.item()))
        layer_trace.append(layer)
        topk_mass.append(float(probs.topk(min(5, probs.shape[-1]), dim=-1).values.sum(-1).item()))
        if first is None:
            top_probs, top_ids = probs[0].topk(10)
            first = {"confidence": float(confidence.item()), "entropy": float(entropy.item()),
                     "selected_premature_layer": layer, "uncertainty": uncertainty,
                     "top_tokens": [(tokenizer.decode([i.item()]), float(p.item())) for i, p in zip(top_ids, top_probs)]}
        generated = torch.cat([generated, token[:, None]], dim=1)
        step_input = token[:, None]
        if token.item() == tokenizer.eos_token_id:
            break
    first["answer"] = tokenizer.decode(generated[0, input_ids.shape[1]:], skip_special_tokens=True).strip()
    first.update(_aggregate_token_diagnostics(token_confidences, token_entropies, token_log_probs, topk_mass, layer_trace))
    return first


def _aggregate_token_diagnostics(confidences, entropies, log_probs, topk_mass, layers):
    """Summarize uncertainty over generated tokens, never only the first token."""
    if not confidences:
        return {"token_probabilities": [], "mean_token_confidence": 0.0,
                "min_token_confidence": 0.0, "mean_token_entropy": 0.0,
                "max_token_entropy": 0.0, "std_token_entropy": 0.0,
                "uncertain_token_ratio": 1.0, "sequence_log_probability": float("-inf"),
                "normalized_sequence_probability": 0.0, "topk_probability_mass": [],
                "uncertainty_curve": [], "dola_layers": []}
    import math
    return {"token_probabilities": confidences,
            "mean_token_confidence": float(np.mean(confidences)),
            "min_token_confidence": float(np.min(confidences)),
            "mean_token_entropy": float(np.mean(entropies)),
            "max_token_entropy": float(np.max(entropies)),
            "std_token_entropy": float(np.std(entropies)),
            "uncertain_token_ratio": float(np.mean(np.asarray(confidences) < 0.25)),
            "sequence_log_probability": float(np.sum(log_probs)),
            "normalized_sequence_probability": float(math.exp(np.mean(log_probs))),
            "topk_probability_mass": topk_mass,
            "uncertainty_curve": entropies,
            "dola_layers": layers}


@torch.inference_mode()
def generate_candidate_batch(model, tokenizer, input_ids, method: str, batch_size: int,
                             alpha: float = 1.0, max_new_tokens: int = 16,
                             temperature: float = 0.7, top_p: float = 0.9,
                             seed: int = 42):
    """Generate independent candidates from a repeated prompt in one batch."""
    if batch_size < 1 or temperature <= 0 or not 0 < top_p <= 1:
        raise ValueError("batch_size, temperature, and top_p are outside valid ranges")
    device = input_ids.device
    batch = input_ids.expand(batch_size, -1).contiguous()
    generated = batch.clone()
    generator = torch.Generator(device=device).manual_seed(int(seed))
    done = torch.zeros(batch_size, dtype=torch.bool, device=device)
    first_confidence = None
    token_confidences, token_entropies, token_log_probs, layer_trace, topk_mass = [[] for _ in range(batch_size)], [[] for _ in range(batch_size)], [[] for _ in range(batch_size)], [[] for _ in range(batch_size)], [[] for _ in range(batch_size)]
    step_input = batch
    cache = None
    for _ in range(max_new_tokens):
        logits, selected_layer, _, cache = next_token_distribution(
            model, step_input, method, alpha=alpha, past_key_values=cache, return_cache=True)
        logits = logits / temperature
        if top_p < 1:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            sorted_probs = F.softmax(sorted_logits, dim=-1)
            remove = sorted_probs.cumsum(-1) - sorted_probs > top_p
            sorted_logits[remove] = -float("inf")
            logits = torch.full_like(logits, -float("inf")).scatter(-1, sorted_indices, sorted_logits)
        probs = F.softmax(logits, dim=-1)
        if first_confidence is None:
            first_confidence = probs.max(-1).values
        entropies = -(probs * probs.clamp_min(1e-10).log()).sum(-1)
        token = torch.multinomial(probs, 1, generator=generator).squeeze(-1)
        selected_probability = probs.gather(-1, token[:, None]).squeeze(-1)
        topk = probs.topk(min(5, probs.shape[-1]), dim=-1).values.sum(-1)
        for row in range(batch_size):
            if not bool(done[row]):
                token_confidences[row].append(float(selected_probability[row].item()))
                token_log_probs[row].append(float(selected_probability[row].clamp_min(1e-30).log().item()))
                token_entropies[row].append(float(entropies[row].item()))
                topk_mass[row].append(float(topk[row].item()))
                layer_trace[row].append(selected_layer[row] if isinstance(selected_layer, list) else selected_layer)
        if tokenizer.eos_token_id is not None:
            token = torch.where(done, torch.full_like(token, tokenizer.eos_token_id), token)
        generated = torch.cat([generated, token[:, None]], dim=1)
        done |= token.eq(tokenizer.eos_token_id) if tokenizer.eos_token_id is not None else torch.zeros_like(done)
        if bool(done.all()):
            break
        step_input = token[:, None]
    results = []
    for i in range(batch_size):
        answer = tokenizer.decode(generated[i, input_ids.shape[1]:], skip_special_tokens=True).strip()
        item = {"answer": answer,
                "confidence": float(first_confidence[i].item()) if first_confidence is not None else 0.0}
        item.update(_aggregate_token_diagnostics(token_confidences[i], token_entropies[i], token_log_probs[i], topk_mass[i], layer_trace[i]))
        results.append(item)
    return results
