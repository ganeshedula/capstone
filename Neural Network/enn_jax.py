"""
Epistemic Neural Network (Epinet) in JAX — Corrected.

Architecture per the capstone paper (Verma et al.):

  1. Prior network  : frozen random 2-hidden-layer MLP, output modulated by
                      epistemic index z  (stop-gradient)
  2. Learnable net  : trained 2-hidden-layer MLP, output modulated by z
  3. ENN output     : prior(x,z) + learnable(x,z)   →  hidden-size vector
  4. Vocab logits   : ENN output  @  frozen LM-head weight^T
  5. Training loss  : cross-entropy on  softmax(DoLa_logits + ENN_logits)
                      where DoLa logits are reconstructed from the stored
                      post-norm hidden features

Input  : concat [normed_mature, normed_premature]  →  2048*2 = 4096  dims
Output : hidden-size vector  (2048)  →  projected to vocab via LM head

Epistemic uncertainty is estimated by sampling multiple z vectors and
computing the mutual-information decomposition:
    epistemic  = H[mean(p_k)]  −  mean[H(p_k)]
"""

import os
import pickle
from typing import Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np


# ---------------------------------------------------------------------------
# Architecture constants  (tuned for TinyLlama-1.1B)
# ---------------------------------------------------------------------------

HIDDEN_SIZE         = 2048          # TinyLlama per-layer dim
INPUT_DIM           = HIDDEN_SIZE * 2   # concat mature + premature
BOTTLENECK_DIM      = 512           # MLP hidden width
OUTPUT_DIM          = HIDDEN_SIZE   # maps back to hidden space
Z_DIM               = 6            # epistemic index dimension
N_EPISTEMIC_SAMPLES = 16           # z samples at inference
VOCAB_SIZE          = 32_000       # TinyLlama vocab  (used only for docs)


# ---------------------------------------------------------------------------
# Network initialisation
# ---------------------------------------------------------------------------

def init_prior_network(
    rng: jax.random.PRNGKey,
    scale: float = 0.01,
) -> Dict:
    """Frozen prior network — small random weights, never updated."""
    k1, k2, k3 = jax.random.split(rng, 3)
    return {
        "w1": jax.random.normal(k1, (INPUT_DIM, BOTTLENECK_DIM)) * scale,
        "b1": jnp.zeros(BOTTLENECK_DIM),
        "w2": jax.random.normal(k2, (BOTTLENECK_DIM, BOTTLENECK_DIM)) * scale,
        "b2": jnp.zeros(BOTTLENECK_DIM),
        "w3": jax.random.normal(k3, (BOTTLENECK_DIM, OUTPUT_DIM * Z_DIM)) * scale,
        "b3": jnp.zeros(OUTPUT_DIM * Z_DIM),
    }


def init_learnable_network(rng: jax.random.PRNGKey) -> Dict:
    """Trainable learnable network — Xavier initialisation."""
    k1, k2, k3 = jax.random.split(rng, 3)
    lim1 = jnp.sqrt(6.0 / (INPUT_DIM + BOTTLENECK_DIM))
    lim2 = jnp.sqrt(6.0 / (BOTTLENECK_DIM + BOTTLENECK_DIM))
    lim3 = jnp.sqrt(6.0 / (BOTTLENECK_DIM + OUTPUT_DIM * Z_DIM))
    return {
        "w1": jax.random.uniform(k1, (INPUT_DIM, BOTTLENECK_DIM),
                                 minval=-lim1, maxval=lim1),
        "b1": jnp.zeros(BOTTLENECK_DIM),
        "w2": jax.random.uniform(k2, (BOTTLENECK_DIM, BOTTLENECK_DIM),
                                 minval=-lim2, maxval=lim2),
        "b2": jnp.zeros(BOTTLENECK_DIM),
        "w3": jax.random.uniform(k3, (BOTTLENECK_DIM, OUTPUT_DIM * Z_DIM),
                                 minval=-lim3, maxval=lim3),
        "b3": jnp.zeros(OUTPUT_DIM * Z_DIM),
    }


# ---------------------------------------------------------------------------
# Forward passes
# ---------------------------------------------------------------------------

def _mlp_forward(params: Dict, x: jnp.ndarray) -> jnp.ndarray:
    """Two-hidden-layer MLP:  Linear → ReLU → Linear → ReLU → Linear."""
    h = jax.nn.relu(x @ params["w1"] + params["b1"])
    h = jax.nn.relu(h @ params["w2"] + params["b2"])
    return h @ params["w3"] + params["b3"]


def epinet_forward(
    prior_params: Dict,
    learnable_params: Dict,
    x: jnp.ndarray,
    z: jnp.ndarray,
) -> jnp.ndarray:
    """
    Epinet forward pass for a single epistemic index vector z.

    Args:
        x : (batch, INPUT_DIM)  — concatenated post-norm features
        z : (Z_DIM,)            — epistemic index sample

    Returns:
        (batch, OUTPUT_DIM)     — ENN hidden-space output
    """
    batch = x.shape[0]

    # Prior path  (frozen — stop_gradient ensures no gradient flow)
    prior_multi = _mlp_forward(prior_params, x)                # (batch, OUTPUT_DIM * Z_DIM)
    prior_multi = prior_multi.reshape(batch, OUTPUT_DIM, Z_DIM)
    prior_out   = jax.lax.stop_gradient(prior_multi @ z)       # (batch, OUTPUT_DIM)

    # Learnable path  (trained)
    learn_multi = _mlp_forward(learnable_params, x)            # (batch, OUTPUT_DIM * Z_DIM)
    learn_multi = learn_multi.reshape(batch, OUTPUT_DIM, Z_DIM)
    learn_out   = learn_multi @ z                              # (batch, OUTPUT_DIM)

    return prior_out + learn_out


# ---------------------------------------------------------------------------
# Loss and training step
# ---------------------------------------------------------------------------

def _combined_loss(
    learnable_params: Dict,
    prior_params: Dict,
    x_concat: jnp.ndarray,       # (batch, INPUT_DIM)
    x_mature: jnp.ndarray,       # (batch, HIDDEN_SIZE)
    x_premature: jnp.ndarray,    # (batch, HIDDEN_SIZE)
    y: jnp.ndarray,              # (batch,)  int32
    z: jnp.ndarray,              # (Z_DIM,)
    vh_weight: jnp.ndarray,      # (VOCAB, HIDDEN_SIZE)
    alpha: float,
) -> jnp.ndarray:
    """
    Cross-entropy on  softmax(DoLa_logits + ENN_logits)  for one z sample.

    DoLa logits are reconstructed from the stored post-norm hidden features:
        dola = normed_mature @ W_lm^T  −  alpha * normed_premature @ W_lm^T
    """
    # Reconstruct DoLa logits  (no gradient needed — these are from frozen LLM)
    dola_logits = (
        x_mature @ vh_weight.T
        - alpha * x_premature @ vh_weight.T
    )
    dola_logits = jax.lax.stop_gradient(dola_logits)

    # ENN hidden output → vocab logits via frozen LM head
    enn_hidden = epinet_forward(prior_params, learnable_params, x_concat, z)
    enn_logits = enn_hidden @ vh_weight.T

    # Combined
    combined = dola_logits + enn_logits
    log_probs = jax.nn.log_softmax(combined, axis=-1)
    return -jnp.mean(log_probs[jnp.arange(len(y)), y])


def _multi_z_loss(
    learnable_params: Dict,
    prior_params: Dict,
    x_concat: jnp.ndarray,
    x_mature: jnp.ndarray,
    x_premature: jnp.ndarray,
    y: jnp.ndarray,
    z_batch: jnp.ndarray,        # (n_z, Z_DIM)
    vh_weight: jnp.ndarray,
    alpha: float,
) -> jnp.ndarray:
    """Mean loss across multiple z samples."""
    def single_z(z):
        return _combined_loss(
            learnable_params, prior_params,
            x_concat, x_mature, x_premature,
            y, z, vh_weight, alpha,
        )
    losses = jax.vmap(single_z)(z_batch)
    return jnp.mean(losses)


@jax.jit
def _train_step(
    learnable_params: Dict,
    prior_params: Dict,
    x_concat: jnp.ndarray,
    x_mature: jnp.ndarray,
    x_premature: jnp.ndarray,
    y: jnp.ndarray,
    z_batch: jnp.ndarray,
    vh_weight: jnp.ndarray,
    alpha: float,
    lr: float,
) -> Tuple[Dict, jnp.ndarray]:
    """Single SGD step — differentiate only w.r.t. learnable_params."""
    loss, grads = jax.value_and_grad(_multi_z_loss)(
        learnable_params, prior_params,
        x_concat, x_mature, x_premature,
        y, z_batch, vh_weight, alpha,
    )
    learnable_params = jax.tree_util.tree_map(
        lambda p, g: p - lr * g,
        learnable_params, grads,
    )
    return learnable_params, loss


# ---------------------------------------------------------------------------
# Public training API
# ---------------------------------------------------------------------------

def train_enn(
    mature_features: np.ndarray,       # (N, H)  float32 — post-RMSNorm
    premature_features: np.ndarray,    # (N, H)  float32 — post-RMSNorm
    labels: np.ndarray,                # (N,)    int32
    vocab_head_weight: np.ndarray,     # (V, H)  float32 — frozen LM head
    alpha: float = 1.0,
    epochs: int = 5,
    batch_size: int = 64,
    lr: float = 1e-4,
    n_z_samples: int = 4,
    seed: int = 42,
    checkpoint_path: Optional[str] = None,
    val_mature: Optional[np.ndarray] = None,
    val_premature: Optional[np.ndarray] = None,
    val_labels: Optional[np.ndarray] = None,
) -> Tuple[Dict, Dict, List[float], List[float]]:
    """
    Train the epinet on next-token prediction with combined DoLa+ENN loss.

    Only the learnable network is updated.  The prior network is frozen.

    Returns:
        prior_params     : frozen weights
        learnable_params : trained weights
        train_losses     : per-epoch training loss
        val_losses       : per-epoch validation loss (empty if no val data)
    """
    N = len(mature_features)
    if N == 0:
        raise ValueError(
            "[ENN] No training samples provided. "
            "Feature extraction likely failed — check C4 data loading."
        )

    rng = jax.random.PRNGKey(seed)
    k1, k2, k3 = jax.random.split(rng, 3)

    prior_params     = init_prior_network(k1)
    learnable_params = init_learnable_network(k2)

    # Convert to JAX arrays
    x_mat_all  = jnp.array(mature_features,    dtype=jnp.float32)
    x_prem_all = jnp.array(premature_features, dtype=jnp.float32)
    x_cat_all  = jnp.concatenate([x_mat_all, x_prem_all], axis=-1)
    y_all      = jnp.array(labels, dtype=jnp.int32)
    vh         = jnp.array(vocab_head_weight, dtype=jnp.float32)

    # Validation data (optional)
    has_val = (
        val_mature is not None
        and val_premature is not None
        and val_labels is not None
        and len(val_labels) > 0
    )
    if has_val:
        v_mat  = jnp.array(val_mature,    dtype=jnp.float32)
        v_prem = jnp.array(val_premature, dtype=jnp.float32)
        v_cat  = jnp.concatenate([v_mat, v_prem], axis=-1)
        v_y    = jnp.array(val_labels, dtype=jnp.int32)

    n_batches = max(1, N // batch_size)

    # Parameter count
    n_learn  = sum(p.size for p in jax.tree_util.tree_leaves(learnable_params))
    n_prior  = sum(p.size for p in jax.tree_util.tree_leaves(prior_params))

    print(f"[ENN] Epinet training")
    print(f"  Samples     : {N} train" + (f", {len(val_labels)} val" if has_val else ""))
    print(f"  Epochs      : {epochs}")
    print(f"  Batch size  : {batch_size}")
    print(f"  LR          : {lr}")
    print(f"  z_dim       : {Z_DIM}")
    print(f"  n_z_samples : {n_z_samples}")
    print(f"  Bottleneck  : {BOTTLENECK_DIM}")
    print(f"  Alpha       : {alpha}")
    print(f"  Learnable   : {n_learn:,} params")
    print(f"  Prior       : {n_prior:,} params (frozen)")
    print(f"  Total ENN   : {n_learn + n_prior:,} params")

    rng_np = np.random.default_rng(seed)
    z_rng  = k3

    train_losses: List[float] = []
    val_losses:   List[float] = []

    for epoch in range(epochs):
        # Shuffle training data
        perm = rng_np.permutation(N)
        x_cat_shuf  = x_cat_all[perm]
        x_mat_shuf  = x_mat_all[perm]
        x_prem_shuf = x_prem_all[perm]
        y_shuf      = y_all[perm]

        epoch_loss = 0.0
        for i in range(n_batches):
            s = i * batch_size
            e = s + batch_size
            if e > N:
                break  # drop last incomplete batch

            # Fresh z samples per step
            z_rng, z_key = jax.random.split(z_rng)
            z_batch = jax.random.normal(z_key, (n_z_samples, Z_DIM))

            learnable_params, loss = _train_step(
                learnable_params, prior_params,
                x_cat_shuf[s:e],
                x_mat_shuf[s:e],
                x_prem_shuf[s:e],
                y_shuf[s:e],
                z_batch, vh, alpha, lr,
            )
            epoch_loss += float(loss)

        avg_train = epoch_loss / max(1, n_batches)
        train_losses.append(avg_train)

        # Validation loss
        if has_val:
            z_rng, z_key = jax.random.split(z_rng)
            z_val = jax.random.normal(z_key, (n_z_samples, Z_DIM))
            # Evaluate on full val set (or first batch_size samples)
            n_val = min(len(v_y), batch_size * 4)
            v_loss = float(_multi_z_loss(
                learnable_params, prior_params,
                v_cat[:n_val], v_mat[:n_val], v_prem[:n_val],
                v_y[:n_val], z_val, vh, alpha,
            ))
            val_losses.append(v_loss)
            print(
                f"  Epoch {epoch + 1:3d}/{epochs}"
                f"  train_loss = {avg_train:.4f}"
                f"  val_loss = {v_loss:.4f}"
            )
        else:
            print(f"  Epoch {epoch + 1:3d}/{epochs}  train_loss = {avg_train:.4f}")

    if checkpoint_path:
        save_enn(prior_params, learnable_params, checkpoint_path)
        print(f"[ENN] Checkpoint saved → {checkpoint_path}")

    return prior_params, learnable_params, train_losses, val_losses


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_enn(prior_params: Dict, learnable_params: Dict, path: str) -> None:
    """Serialise ENN params to disk."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as f:
        prior_np     = jax.tree_util.tree_map(np.array, prior_params)
        learnable_np = jax.tree_util.tree_map(np.array, learnable_params)
        pickle.dump({"prior": prior_np, "learnable": learnable_np}, f)


def load_enn(path: str) -> Tuple[Dict, Dict]:
    """Deserialise ENN params from checkpoint."""
    with open(path, "rb") as f:
        ckpt = pickle.load(f)
    prior_params     = jax.tree_util.tree_map(jnp.array, ckpt["prior"])
    learnable_params = jax.tree_util.tree_map(jnp.array, ckpt["learnable"])
    return prior_params, learnable_params


# ---------------------------------------------------------------------------
# Inference  (numpy I/O for easy PyTorch interop)
# ---------------------------------------------------------------------------

def _epinet_logits_single_z(
    prior_params: Dict,
    learnable_params: Dict,
    x: jnp.ndarray,
    z: jnp.ndarray,
    vh_weight: jnp.ndarray,
) -> jnp.ndarray:
    """Compute ENN vocab logits for a single z sample."""
    enn_hidden = epinet_forward(prior_params, learnable_params, x, z)
    return enn_hidden @ vh_weight.T

# Vectorise over z samples
_vmap_epinet_over_z = jax.vmap(
    _epinet_logits_single_z,
    in_axes=(None, None, None, 0, None),
)


def enn_predict(
    prior_params: Dict,
    learnable_params: Dict,
    normed_mature: np.ndarray,       # (T, H)  float32 — post-RMSNorm
    normed_premature: np.ndarray,    # (T, H)  float32 — post-RMSNorm
    vocab_head_weight: np.ndarray,   # (V, H)  float32
    n_samples: int = N_EPISTEMIC_SAMPLES,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Run epinet inference with multiple z samples.

    Returns:
        mean_logits     : (T, V) — mean ENN logits across z samples
        epistemic_unc   : (T,)   — epistemic uncertainty  (mutual information)
        predictive_unc  : (T,)   — total predictive uncertainty  (entropy)
        aleatoric_unc   : (T,)   — aleatoric uncertainty  (mean entropy)
    """
    x = np.concatenate([normed_mature, normed_premature], axis=-1)
    x_jax = jnp.array(x,                 dtype=jnp.float32)
    vh    = jnp.array(vocab_head_weight,  dtype=jnp.float32)

    # Sample z vectors
    rng       = jax.random.PRNGKey(seed)
    z_samples = jax.random.normal(rng, (n_samples, Z_DIM))

    # Logits for all z:  (n_samples, T, V)
    all_logits = _vmap_epinet_over_z(
        prior_params, learnable_params, x_jax, z_samples, vh,
    )

    # Probabilities per z
    all_probs = jax.nn.softmax(all_logits, axis=-1)           # (n_z, T, V)

    # ---- Uncertainty decomposition ----
    mean_probs   = jnp.mean(all_probs, axis=0)                # (T, V)
    mean_logits  = jnp.mean(all_logits, axis=0)               # (T, V)

    # Predictive uncertainty  =  H[E_z[p(y|x,z)]]
    predictive_unc = -jnp.sum(
        mean_probs * jnp.log(jnp.clip(mean_probs, 1e-10, 1.0)),
        axis=-1,
    )  # (T,)

    # Aleatoric uncertainty  =  E_z[H[p(y|x,z)]]
    per_z_entropy = -jnp.sum(
        all_probs * jnp.log(jnp.clip(all_probs, 1e-10, 1.0)),
        axis=-1,
    )  # (n_z, T)
    aleatoric_unc = jnp.mean(per_z_entropy, axis=0)           # (T,)

    # Epistemic uncertainty  =  I(y; z | x)  =  predictive − aleatoric
    epistemic_unc = jnp.maximum(predictive_unc - aleatoric_unc, 0.0)

    return (
        np.array(mean_logits,    dtype=np.float32),
        np.array(epistemic_unc,  dtype=np.float32),
        np.array(predictive_unc, dtype=np.float32),
        np.array(aleatoric_unc,  dtype=np.float32),
    )
