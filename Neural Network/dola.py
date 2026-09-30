"""
DoLa (Decoding by Contrasting Layers) for TinyLlama — Corrected.

Contrastive decoding (per Chuang et al. 2023):

    contrastive_logits = mature_logits  -  alpha * premature_logits
    final_distribution = softmax(contrastive_logits)

The premature layer is selected dynamically per token position using
Jensen-Shannon divergence, or can be fixed for controlled experiments.

TinyLlama-1.1B-Chat-v1.0
    num_hidden_layers = 22   (layers 0..21)
    hidden_size       = 2048
    vocab_size        = 32000
"""

import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# TinyLlama layer configuration
# ---------------------------------------------------------------------------

MATURE_LAYER = 21
# Dynamic selection does not require projecting every intermediate layer.
# These span TinyLlama's 22 blocks while keeping one forward pass practical on
# MPS/CPU.  A caller may still supply a denser candidate list for an ablation.
CANDIDATE_PREMATURE_LAYERS = [0, 4, 8, 12, 16, 20]


# ---------------------------------------------------------------------------
# Statistics tracker
# ---------------------------------------------------------------------------

@dataclass
class DoLaStats:
    """Accumulate premature-layer selection statistics."""

    layer_counts: Dict[int, int] = field(default_factory=dict)
    jsd_values: List[float] = field(default_factory=list)

    def update(
        self,
        selected_layer_indices: torch.Tensor,
        max_jsd_values: torch.Tensor,
    ) -> None:
        for idx in selected_layer_indices.flatten().tolist():
            self.layer_counts[idx] = self.layer_counts.get(idx, 0) + 1
        self.jsd_values.extend(max_jsd_values.flatten().tolist())

    def summary(self) -> str:
        lines = []
        if self.jsd_values:
            arr = np.array(self.jsd_values)
            lines.append(
                f"  JSD — min: {arr.min():.6f}  "
                f"max: {arr.max():.6f}  "
                f"mean: {arr.mean():.6f}  "
                f"std: {arr.std():.6f}"
            )
            n_nan = int(np.isnan(arr).sum())
            n_inf = int(np.isinf(arr).sum())
            if n_nan or n_inf:
                lines.append(
                    f"  ⚠️  Numerical issues: "
                    f"{n_nan} NaN, {n_inf} Inf"
                )
        if self.layer_counts:
            total = sum(self.layer_counts.values())
            lines.append(f"  Premature-layer distribution  (total tokens = {total})")
            for layer, count in sorted(self.layer_counts.items()):
                pct = count / total * 100
                bar = "█" * int(pct / 2)
                lines.append(f"    Layer {layer:2d}: {count:6d}  ({pct:5.1f}%)  {bar}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Jensen-Shannon Divergence  (per-position)
# ---------------------------------------------------------------------------

def js_divergence(
    logits_a: torch.Tensor,
    logits_b: torch.Tensor,
) -> torch.Tensor:
    """
    Compute JSD between two logit tensors for every (batch, position).

    Args:
        logits_a, logits_b : (B, T, V) or (T, V)

    Returns:
        (B, T) or (T,) — per-position JSD values
    """
    a = logits_a.float()
    b = logits_b.float()

    p = F.softmax(a, dim=-1)
    q = F.softmax(b, dim=-1)

    m = 0.5 * (p + q)

    # Use log-domain for numerical stability
    p_log = torch.log(p.clamp_min(1e-10))
    q_log = torch.log(q.clamp_min(1e-10))
    m_log = torch.log(m.clamp_min(1e-10))

    kl_pm = torch.sum(p * (p_log - m_log), dim=-1)
    kl_qm = torch.sum(q * (q_log - m_log), dim=-1)

    return 0.5 * (kl_pm + kl_qm)


# ---------------------------------------------------------------------------
# Forward pass — extract hidden states
# ---------------------------------------------------------------------------

def extract_hidden_states(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
) -> Tuple[List[torch.Tensor], torch.Tensor]:
    """
    Single forward pass through TinyLlama.

    Returns:
        hidden_states : list of (B, T, H) tensors — one per transformer layer
                        (index 0 = layer 0, index 21 = layer 21)
        base_logits   : (B, T, V) standard LM-head logits
    """
    with torch.inference_mode():
        out = model(
            input_ids=input_ids,
            output_hidden_states=True,
            use_cache=False,
        )
    # out.hidden_states[0] = embedding output (skip)
    # out.hidden_states[1] .. out.hidden_states[22] = transformer layers 0..21
    hidden_states = list(out.hidden_states[1:])
    expected_layers = int(model.config.num_hidden_layers)
    if len(hidden_states) != expected_layers:
        raise RuntimeError(
            f"Expected {expected_layers} transformer hidden states, got {len(hidden_states)}."
        )
    return hidden_states, out.logits


# ---------------------------------------------------------------------------
# Project hidden state → vocabulary logits
# ---------------------------------------------------------------------------

def _layer_logits(
    model: torch.nn.Module,
    hidden: torch.Tensor,
) -> torch.Tensor:
    """Project an intermediate (pre-final-norm) layer to vocabulary logits."""
    normed = model.model.norm(hidden)
    return model.lm_head(normed)


# ---------------------------------------------------------------------------
# Dynamic premature-layer selection  (per-position)
# ---------------------------------------------------------------------------

def select_premature_layers_per_position(
    model: torch.nn.Module,
    hidden_states: List[torch.Tensor],
    candidate_layers: List[int],
    mature_layer: int = MATURE_LAYER,
    stats: Optional[DoLaStats] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Select the premature layer with highest JSD for every token position.

    Returns:
        candidate_indices   : (B, T) — index into candidate_layers list
        selected_logits     : (B, T, V) — premature logits at selected layer
        selected_layer_ids  : (B, T) — actual transformer layer numbers
    """
    mature_hidden = hidden_states[mature_layer]
    # Hugging Face Llama's final hidden state has already passed model.norm.
    # Applying it again changes the base distribution, so project it directly.
    mature_logits = model.lm_head(mature_hidden)  # (B, T, V)

    jsd_scores = []          # per-candidate  (B, T)
    candidate_logits = []    # per-candidate  (B, T, V)

    for layer_idx in candidate_layers:
        prem_logits = _layer_logits(model, hidden_states[layer_idx])
        candidate_logits.append(prem_logits)
        jsd = js_divergence(mature_logits, prem_logits)  # (B, T)
        jsd_scores.append(jsd)

    # (num_candidates, B, T)
    jsd_stack = torch.stack(jsd_scores, dim=0)

    # Best candidate per position
    candidate_indices = torch.argmax(jsd_stack, dim=0)       # (B, T)
    max_jsd_values = torch.max(jsd_stack, dim=0).values      # (B, T)

    # Map candidate index → actual layer number
    layer_tensor = torch.tensor(
        candidate_layers,
        device=candidate_indices.device,
        dtype=torch.long,
    )
    selected_layer_ids = layer_tensor[candidate_indices]     # (B, T)

    # Gather the selected premature logits
    selected_logits = torch.zeros_like(candidate_logits[0])
    for i, logits in enumerate(candidate_logits):
        mask = (candidate_indices == i).unsqueeze(-1)
        selected_logits = torch.where(mask, logits, selected_logits)

    if stats is not None:
        stats.update(selected_layer_ids, max_jsd_values)

    return candidate_indices, selected_logits, selected_layer_ids


# Backward-compatible alias
select_premature_layer = select_premature_layers_per_position


# ---------------------------------------------------------------------------
# Extract DoLa features  (for data_prep)
# ---------------------------------------------------------------------------

def extract_dola_features(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    candidate_layers: Optional[List[int]] = None,
    mature_layer: int = MATURE_LAYER,
    stats: Optional[DoLaStats] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extract per-position mature and premature hidden states for ENN training.

    The premature layer is selected dynamically (max JSD) at each position.
    Hidden states are passed through the final RMSNorm so that they can be
    directly multiplied by the LM-head weight to reconstruct logits.

    Returns:
        normed_mature    : (B, T, H) — post-RMSNorm mature hidden state
        normed_premature : (B, T, H) — post-RMSNorm premature hidden state
        premature_ids    : (B, T)    — actual layer numbers selected
    """
    if candidate_layers is None:
        candidate_layers = CANDIDATE_PREMATURE_LAYERS

    with torch.inference_mode():
        hidden_states, _ = extract_hidden_states(model, input_ids)

        mature_hidden = hidden_states[mature_layer]  # (B, T, H)

        # Dynamic selection
        candidate_indices, _, selected_layer_ids = (
            select_premature_layers_per_position(
                model, hidden_states, candidate_layers,
                mature_layer, stats=stats,
            )
        )

        # Gather per-position premature hidden states
        premature_hidden = torch.zeros_like(mature_hidden)
        for i, layer_idx in enumerate(candidate_layers):
            mask = (candidate_indices == i).unsqueeze(-1)
            premature_hidden = torch.where(
                mask, hidden_states[layer_idx], premature_hidden,
            )

        # The final hidden state is already post-RMSNorm; premature states are not.
        normed_mature = mature_hidden
        normed_premature = model.model.norm(premature_hidden)

    return normed_mature, normed_premature, selected_layer_ids


# ---------------------------------------------------------------------------
# Main DoLa function
# ---------------------------------------------------------------------------

def dola_logits(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    candidate_premature_layers: Optional[List[int]] = None,
    mature_layer: int = MATURE_LAYER,
    alpha: float = 1.0,
    fixed_premature_layer: Optional[int] = None,
    stats: Optional[DoLaStats] = None,
) -> Tuple[
    torch.Tensor,   # contrastive_log_probs  (B, T, V)
    torch.Tensor,   # contrastive_logits_raw (B, T, V)
    torch.Tensor,   # premature_layer_ids    (B, T)
    torch.Tensor,   # mature_hidden          (B, T, H)
    torch.Tensor,   # premature_hidden       (B, T, H)
]:
    """
    Compute DoLa contrastive logits using the CORRECT formula:

        contrastive_logits = mature_logits − alpha × premature_logits

    Returns both:
      • log_softmax(contrastive_logits)  — for direct scoring
      • raw contrastive_logits           — for ENN combination

    Also returns hidden states (pre-norm) for ENN feature extraction
    at inference time.
    """
    if candidate_premature_layers is None:
        candidate_premature_layers = CANDIDATE_PREMATURE_LAYERS

    with torch.inference_mode():
        hidden_states, _ = extract_hidden_states(model, input_ids)

        mature_hidden = hidden_states[mature_layer]   # already post-final-RMSNorm
        mature_logits = model.lm_head(mature_hidden).float()

        if fixed_premature_layer is not None:
            # ---- Fixed premature layer (for controlled experiments) ----
            premature_hidden = hidden_states[fixed_premature_layer]
            premature_logits = _layer_logits(model, premature_hidden).float()
            premature_layer_ids = torch.full(
                mature_logits.shape[:2],
                fixed_premature_layer,
                device=mature_logits.device,
                dtype=torch.long,
            )
        else:
            # ---- Dynamic premature layer selection ----
            candidate_indices, premature_logits, premature_layer_ids = (
                select_premature_layers_per_position(
                    model, hidden_states,
                    candidate_premature_layers,
                    mature_layer,
                    stats=stats,
                )
            )
            premature_logits = premature_logits.float()

            # Gather per-position premature hidden states
            premature_hidden = torch.zeros_like(mature_hidden)
            for i, layer_idx in enumerate(candidate_premature_layers):
                mask = (candidate_indices == i).unsqueeze(-1)
                premature_hidden = torch.where(
                    mask, hidden_states[layer_idx], premature_hidden,
                )

        # ---- CORRECT contrastive formula: subtract RAW logits ----
        contrastive_logits_raw = mature_logits - alpha * premature_logits

        # Single log_softmax for scoring
        contrastive_log_probs = F.log_softmax(
            contrastive_logits_raw, dim=-1,
        )

    return (
        contrastive_log_probs,
        contrastive_logits_raw,
        premature_layer_ids,
        mature_hidden,
        premature_hidden,
    )
