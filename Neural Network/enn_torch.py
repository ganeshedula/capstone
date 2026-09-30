"""MPS-compatible Epinet for frozen TinyLlama features.

The prior is random and permanently frozen; only ``learnable`` is passed to
the optimizer.  Inputs are post-final-RMSNorm mature features concatenated
with final-RMSNorm-projected premature features.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class _IndexedMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, z_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, output_dim * z_dim),
        )
        self.output_dim, self.z_dim = output_dim, z_dim

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        values = self.net(x).view(x.shape[0], self.output_dim, self.z_dim)
        return torch.matmul(values, z)


class Epinet(nn.Module):
    def __init__(self, hidden_size: int, enn_hidden_dim: int = 512, z_dim: int = 6):
        super().__init__()
        self.hidden_size, self.z_dim = hidden_size, z_dim
        self.prior = _IndexedMLP(hidden_size * 2, enn_hidden_dim, hidden_size, z_dim)
        self.learnable = _IndexedMLP(hidden_size * 2, enn_hidden_dim, hidden_size, z_dim)
        for p in self.prior.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            prior = self.prior(x, z)
        return prior + self.learnable(x, z)


def _make_epinet(hidden_size: int, hidden_dim: int, z_dim: int, device: str, seed: int) -> Epinet:
    torch.manual_seed(seed)
    return Epinet(hidden_size, hidden_dim, z_dim).to(device)


def _checkpoint_payload(net: Epinet, hidden_dim: int) -> Dict:
    return {"format": "torch-epinet-v1", "hidden_size": net.hidden_size,
            "enn_hidden_dim": hidden_dim, "z_dim": net.z_dim, "state_dict": net.state_dict()}


def save_enn(net: Epinet, path: str, hidden_dim: int) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(_checkpoint_payload(net, hidden_dim), path)


def load_enn(path: str, device: Optional[str] = None) -> Epinet:
    device = device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("format") != "torch-epinet-v1":
        raise ValueError("Legacy JAX checkpoint detected. Train a new MPS-compatible ENN checkpoint.")
    net = _make_epinet(payload["hidden_size"], payload["enn_hidden_dim"], payload["z_dim"], device, 42)
    net.load_state_dict(payload["state_dict"])
    net.eval()
    return net


def train_enn(mature_features: np.ndarray, premature_features: np.ndarray, labels: np.ndarray,
              vocab_head_weight: np.ndarray, alpha: float = 1.0, epochs: int = 2,
              batch_size: int = 1, lr: float = 1e-4, n_z_samples: int = 3, seed: int = 42,
              checkpoint_path: Optional[str] = None, val_mature=None, val_premature=None,
              val_labels=None, enn_hidden_dim: int = 512, epistemic_index_dim: int = 6,
              device: Optional[str] = None) -> Tuple[Epinet, Epinet, List[float], List[float]]:
    """Train only the learnable ENN using CE(DoLa logits + ENN logits)."""
    if len(labels) == 0:
        raise ValueError("No ENN training tokens were extracted.")
    device = device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    h = mature_features.shape[-1]
    if premature_features.shape[-1] != h or vocab_head_weight.shape[1] != h:
        raise ValueError("ENN feature / LM-head dimensions do not match.")
    net = _make_epinet(h, enn_hidden_dim, epistemic_index_dim, device, seed)
    optimizer = torch.optim.AdamW(net.learnable.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=1)
    x = torch.from_numpy(np.concatenate([mature_features, premature_features], axis=-1)).to(device)
    mat = torch.from_numpy(mature_features).to(device)
    prem = torch.from_numpy(premature_features).to(device)
    y = torch.from_numpy(labels.astype(np.int64)).to(device)
    head = torch.from_numpy(vocab_head_weight).to(device=device, dtype=torch.float32)
    train_losses, val_losses = [], []
    best = float("inf")
    checkpoint_dir = Path(checkpoint_path).parent if checkpoint_path else None
    for epoch in range(epochs):
        net.train(); order = torch.randperm(len(y), device=device); running = 0.0; steps = 0
        for start in range(0, len(y), batch_size):
            idx = order[start:start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            dola = mat[idx].float() @ head.T - alpha * (prem[idx].float() @ head.T)
            loss = 0.0
            for _ in range(n_z_samples):
                z = torch.randn(net.z_dim, device=device)
                enn_logits = net(x[idx].float(), z) @ head.T
                loss = loss + F.cross_entropy(dola + enn_logits, y[idx]) / n_z_samples
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.learnable.parameters(), max_norm=1.0)
            optimizer.step(); running += float(loss.detach().cpu()); steps += 1
        mean_loss = running / max(steps, 1); train_losses.append(mean_loss)
        scheduler.step(mean_loss)
        print(f"[ENN] epoch {epoch + 1}/{epochs}: train_loss={mean_loss:.4f}; trainable={sum(p.numel() for p in net.learnable.parameters()):,}")
        if checkpoint_dir:
            save_enn(net, str(checkpoint_dir / f"enn_epoch_{epoch + 1}.pt"), enn_hidden_dim)
            if mean_loss < best:
                best = mean_loss; save_enn(net, checkpoint_path, enn_hidden_dim)
    net.eval()
    # The historical two-return convention is retained for callers; both paths
    # are represented by the one Epinet module and remain separately frozen/trainable.
    return net, net, train_losses, val_losses


@torch.inference_mode()
def enn_predict(prior_params, learnable_params, normed_mature: np.ndarray, normed_premature: np.ndarray,
                vocab_head_weight: np.ndarray, n_samples: int = 3, seed: int = 0):
    net = prior_params if isinstance(prior_params, Epinet) else learnable_params
    if not isinstance(net, Epinet):
        raise TypeError("Expected an Epinet module; retrain from the legacy JAX checkpoint.")
    device = next(net.parameters()).device
    x = torch.from_numpy(np.concatenate([normed_mature, normed_premature], axis=-1)).to(device, dtype=torch.float32)
    head = torch.from_numpy(vocab_head_weight).to(device, dtype=torch.float32)
    torch.manual_seed(seed)
    logits = torch.stack([net(x, torch.randn(net.z_dim, device=device)) @ head.T for _ in range(n_samples)])
    probs = logits.softmax(-1); mean_probs = probs.mean(0)
    predictive = -(mean_probs * mean_probs.clamp_min(1e-10).log()).sum(-1)
    aleatoric = (-(probs * probs.clamp_min(1e-10).log()).sum(-1)).mean(0)
    epistemic = (predictive - aleatoric).clamp_min(0)
    return tuple(t.detach().cpu().numpy().astype(np.float32) for t in (logits.mean(0), epistemic, predictive, aleatoric))
