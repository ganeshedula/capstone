"""
C4 feature extraction for ENN training — Corrected.

For each C4 text sample:
  1. Tokenise (truncated to max_length).
  2. Run a single LLM forward pass → all hidden states.
  3. Select the premature layer dynamically (max JSD per token position).
  4. Apply RMSNorm to mature + premature hidden states so they can be
     multiplied directly by the LM-head weight to reconstruct logits.
  5. Skip the first 20 % of token positions (per the capstone paper).
  6. Record:
       normed_mature[pos]     → (2048,)  float32
       normed_premature[pos]  → (2048,)  float32
       input_ids[pos+1]       → next-token label  (int32)

Features are cached as a .npz file (v2 format) so the slow extraction
runs only once.
"""

import gzip
import json
import os
import sys
from typing import List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

# Allow imports from the same directory
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from dola import (
    CANDIDATE_PREMATURE_LAYERS,
    MATURE_LAYER,
    DoLaStats,
    extract_dola_features,
)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_PROJECT   = os.path.abspath(os.path.join(_HERE, ".."))
C4_DIR     = os.path.join(_PROJECT, "dataset", "c4", "en.noblocklist")
MODEL_DIR  = os.path.join(_PROJECT, "models", "TinyLlama-1.1B-Chat-v1.0")
CACHE_PATH = os.path.join(_HERE, "features_cache.npz")

# Cache version — bump whenever the feature format changes
_CACHE_VERSION = 3  # v3 fixes final-layer double-RMSNorm features


# ---------------------------------------------------------------------------
# C4 text loading
# ---------------------------------------------------------------------------

def _is_lfs_pointer(path: str) -> bool:
    """Return True if the file is a Git LFS pointer (not real data)."""
    try:
        with open(path, "rb") as f:
            return f.read(34) == b"version https://git-lfs.github.com"
    except OSError:
        return False


def _load_c4_texts_hf(n_samples: int) -> List[str]:
    """Stream C4 texts directly from HuggingFace."""
    from datasets import load_dataset
    print(
        "[data_prep] Local C4 files are Git LFS pointers "
        "— streaming from HuggingFace instead…"
    )
    ds = load_dataset(
        "allenai/c4", "en",
        split="train",
        streaming=True,
        trust_remote_code=True,
    )
    texts: List[str] = []
    for example in ds:
        text = example.get("text", "").strip()
        if len(text) > 50:
            texts.append(text)
        if len(texts) >= n_samples:
            break
    return texts


def load_c4_texts(n_samples: int = 600, c4_dir: str = C4_DIR) -> List[str]:
    """
    Collect C4 text samples from local .json.gz shards or HuggingFace.

    Falls back to HuggingFace streaming if local files are Git LFS pointers.
    """
    texts: List[str] = []

    if not os.path.isdir(c4_dir):
        return _load_c4_texts_hf(n_samples)

    gz_files = sorted(
        os.path.join(c4_dir, f)
        for f in os.listdir(c4_dir)
        if f.endswith(".json.gz")
    )

    if not gz_files:
        return _load_c4_texts_hf(n_samples)

    # Detect LFS pointers
    if _is_lfs_pointer(gz_files[0]):
        texts = _load_c4_texts_hf(n_samples)
        print(f"[data_prep] Loaded {len(texts)} C4 texts via HuggingFace streaming")
        return texts

    for gz_path in gz_files:
        if len(texts) >= n_samples:
            break
        try:
            with gzip.open(gz_path, "rt", encoding="utf-8") as fh:
                for line in fh:
                    if len(texts) >= n_samples:
                        break
                    try:
                        obj = json.loads(line)
                        text = obj.get("text", "").strip()
                        if len(text) > 50:
                            texts.append(text)
                    except json.JSONDecodeError:
                        continue
        except (gzip.BadGzipFile, OSError) as e:
            print(
                f"[data_prep] Skipping corrupt file "
                f"{os.path.basename(gz_path)}: {e}"
            )
            continue

    print(f"[data_prep] Loaded {len(texts)} C4 texts from {c4_dir}")
    return texts


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_features_from_texts(
    model,
    tokenizer,
    texts: List[str],
    device: str,
    max_length: int = 128,
    skip_first_fraction: float = 0.20,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run the LLM forward pass on each text and collect ENN training triples.

    Hidden states are post-RMSNorm so that they can be directly multiplied
    by the LM-head weight to reconstruct DoLa logits during ENN training.

    Returns
    -------
    normed_mature    : float32  (N, hidden_size)
    normed_premature : float32  (N, hidden_size)
    next_token_ids   : int32    (N,)
    """
    all_mature:    List[np.ndarray] = []
    all_premature: List[np.ndarray] = []
    all_labels:    List[int]        = []

    stats = DoLaStats()

    for text in tqdm(texts, desc="[data_prep] Extracting features"):
        enc = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        ).to(device)

        input_ids = enc["input_ids"]   # (1, T)
        seq_len   = input_ids.shape[1]

        if seq_len < 4:
            continue

        # --- Extract post-norm hidden states via DoLa ---
        normed_mature, normed_premature, _ = extract_dola_features(
            model, input_ids,
            CANDIDATE_PREMATURE_LAYERS, MATURE_LAYER,
            stats=stats,
        )
        # normed_mature    : (1, T, H)  post-RMSNorm
        # normed_premature : (1, T, H)  post-RMSNorm

        # --- Skip first 20 % of tokens  (paper prescription) ---
        skip_n = max(1, int(seq_len * skip_first_fraction))

        for pos in range(skip_n, seq_len - 1):   # need pos+1 as label
            m_h = normed_mature[0, pos, :].float().cpu().numpy()     # (H,)
            p_h = normed_premature[0, pos, :].float().cpu().numpy()  # (H,)
            next_tok = int(input_ids[0, pos + 1].item())

            all_mature.append(m_h)
            all_premature.append(p_h)
            all_labels.append(next_tok)

    mature_arr    = np.array(all_mature,    dtype=np.float32)   # (N, H)
    premature_arr = np.array(all_premature, dtype=np.float32)   # (N, H)
    labels_arr    = np.array(all_labels,    dtype=np.int32)     # (N,)

    print(f"[data_prep] Extracted {len(labels_arr)} training triples")
    print(f"[data_prep] Premature layer statistics:")
    print(stats.summary())

    return mature_arr, premature_arr, labels_arr


# ---------------------------------------------------------------------------
# Public API: prepare (or load cached) features with train/val split
# ---------------------------------------------------------------------------

def prepare_enn_features(
    model,
    tokenizer,
    device: str,
    n_c4_samples: int = 600,
    max_length: int = 128,
    force_recompute: bool = False,
    cache_path: str = CACHE_PATH,
    val_fraction: float = 0.1,
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray,     # train: mature, premature, labels
    np.ndarray, np.ndarray, np.ndarray,     # val:   mature, premature, labels
]:
    """
    Return train/val feature arrays for ENN training.

    If a compatible cache exists it is loaded.  Otherwise features are
    extracted from C4 and saved.

    Returns
    -------
    train_mature, train_premature, train_labels,
    val_mature,   val_premature,   val_labels
    """
    # --- Try loading from cache ---
    if not force_recompute and os.path.exists(cache_path):
        data = np.load(cache_path, allow_pickle=True)

        # Version check
        version = int(data["version"][0]) if "version" in data else 1
        if version < _CACHE_VERSION:
            print(
                f"[data_prep] Cache version mismatch (v{version} vs v{_CACHE_VERSION}). "
                f"Re-extracting features…"
            )
        else:
            mature    = data["mature"]
            premature = data["premature"]
            labels    = data["labels"]
            N         = len(labels)
            print(f"[data_prep] Loaded {N} cached samples (v{version})")

            # Split
            n_val = max(1, int(N * val_fraction))
            return (
                mature[n_val:],    premature[n_val:],    labels[n_val:],
                mature[:n_val],    premature[:n_val],    labels[:n_val],
            )

    # --- Extract from C4 ---
    print(f"[data_prep] Extracting features from {n_c4_samples} C4 samples…")
    texts = load_c4_texts(n_c4_samples)

    mature, premature, labels = extract_features_from_texts(
        model, tokenizer, texts, device, max_length=max_length,
    )

    # Save cache
    os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
    np.savez(
        cache_path,
        mature=mature,
        premature=premature,
        labels=labels,
        version=np.array([_CACHE_VERSION]),
    )
    print(f"[data_prep] Features cached → {cache_path}")

    # Split
    N     = len(labels)
    n_val = max(1, int(N * val_fraction))
    return (
        mature[n_val:],    premature[n_val:],    labels[n_val:],
        mature[:n_val],    premature[:n_val],    labels[:n_val],
    )
