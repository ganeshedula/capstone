"""Epistemic reliability head trained only from labeled answer-level examples.

The token-prediction Epinet checkpoint is intentionally not interpreted as a
hallucination classifier. Training labels must be supplied by the researcher.
"""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import torch
from torch import nn

FEATURES = ("mean_token_confidence", "min_token_confidence", "mean_token_entropy",
            "max_token_entropy", "uncertain_token_ratio", "mean_topk_probability_mass",
            "sequence_log_probability", "normalized_sequence_probability",
            "dola_confidence", "dola_layer_disagreement", "semantic_consistency",
            "largest_cluster_ratio", "average_pairwise_similarity", "factuality_score",
            "contradiction_score", "candidate_disagreement", "answer_length",
            "epistemic_uncertainty", "claim_support", "relevance_score",
            "evidence_support_score")


class ReliabilityENN(nn.Module):
    def __init__(self, input_dim=len(FEATURES), hidden_dim=64, z_dim=8):
        super().__init__(); self.z_dim = z_dim
        self.body = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(),
                                  nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 2*z_dim))
    def forward(self, x, z):
        return (self.body(x).view(-1, 2, self.z_dim) * z[None, None, :]).sum(-1)


def feature_vector(values):
    return np.asarray([float(values.get(key, 0.0)) for key in FEATURES], dtype=np.float32)


def train_reliability_enn(rows, checkpoint, epochs=80, lr=1e-3, device="cpu"):
    """Train on rows of {features: {feature: number}, reliable: 0|1}."""
    if len(rows) < 4 or len({int(r["reliable"]) for r in rows}) < 2:
        raise ValueError("Provide at least four labeled rows including reliable and unreliable examples.")
    x = torch.tensor(np.stack([feature_vector(r["features"]) for r in rows]), device=device)
    y = torch.tensor([int(r["reliable"]) for r in rows], dtype=torch.long, device=device)
    mean, std = x.mean(0), x.std(0).clamp_min(1e-5); x = (x-mean)/std
    torch.manual_seed(42); net = ReliabilityENN().to(device); opt = torch.optim.AdamW(net.parameters(), lr=lr)
    for _ in range(epochs):
        z = torch.randn(len(rows), net.z_dim, device=device)
        loss = nn.functional.cross_entropy(net(x, z), y)
        opt.zero_grad(); loss.backward(); opt.step()
    Path(checkpoint).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": net.state_dict(), "mean": mean.cpu(), "std": std.cpu(), "features": FEATURES}, checkpoint)
    return net


@torch.inference_mode()
def predict_reliability(checkpoint, values, n_samples=16, device="cpu"):
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    if tuple(payload["features"]) != FEATURES: raise ValueError("Reliability checkpoint feature schema mismatch")
    net = ReliabilityENN().to(device); net.load_state_dict(payload["state"]); net.eval()
    x = torch.tensor(feature_vector(values), device=device)[None]
    x = (x-payload["mean"].to(device))/payload["std"].to(device).clamp_min(1e-5)
    logits = torch.stack([net(x, torch.randn(net.z_dim, device=device)).softmax(-1)[0, 1]
                          for _ in range(n_samples)])
    p = float(logits.mean().item())
    return {"reliability_probability": p, "hallucination_probability": 1-p,
            "epistemic_uncertainty": float(logits.var(unbiased=False).item()),
            "source": "labeled ReliabilityENN"}
