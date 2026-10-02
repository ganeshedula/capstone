"""Semantic clustering for sampled answers using the loaded language model."""
from __future__ import annotations

import numpy as np
import torch


@torch.inference_mode()
def answer_embeddings(model, tokenizer, answers, device, max_length=192):
    """Mean-pool final hidden states; no external embedding download is needed."""
    vectors = []
    for answer in answers:
        ids = tokenizer(str(answer), return_tensors="pt", truncation=True,
                        max_length=max_length).input_ids.to(device)
        if ids.shape[1] == 0:
            vectors.append(np.zeros(model.config.hidden_size, dtype=np.float32)); continue
        out = model.model(input_ids=ids, use_cache=False, return_dict=True)
        hidden = out.last_hidden_state.float()[0]
        vector = hidden.mean(0)
        vectors.append(torch.nn.functional.normalize(vector, dim=0).cpu().numpy())
    return np.asarray(vectors, dtype=np.float32)


def cluster_answers(answers, embeddings, threshold=0.78):
    """Connected-component clustering of cosine-similar answers."""
    n = len(answers)
    if n == 0:
        return {"cluster_ids": [], "clusters": [], "largest_cluster_ratio": 0.0,
                "semantic_consistency_score": 0.0, "average_pairwise_similarity": 0.0,
                "cluster_count": 0, "similarity_matrix": []}
    vectors = np.asarray(embeddings, dtype=np.float64)
    vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    similarity = np.clip(vectors @ vectors.T, -1.0, 1.0)
    parent = list(range(n))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    for i in range(n):
        for j in range(i + 1, n):
            if similarity[i, j] >= threshold:
                a, b = find(i), find(j)
                if a != b: parent[b] = a
    groups = {}
    for i in range(n): groups.setdefault(find(i), []).append(i)
    clusters = sorted(groups.values(), key=lambda g: (-len(g), min(g)))
    ids = [0] * n
    for ci, group in enumerate(clusters):
        for i in group: ids[i] = ci
    pairs = similarity[np.triu_indices(n, 1)]
    largest_ratio = len(clusters[0]) / n
    return {"cluster_ids": ids, "clusters": clusters,
            "largest_cluster_ratio": float(largest_ratio),
            "semantic_consistency_score": float(largest_ratio),
            "average_pairwise_similarity": float(pairs.mean()) if len(pairs) else 1.0,
            "cluster_count": len(clusters), "similarity_matrix": similarity.tolist()}
