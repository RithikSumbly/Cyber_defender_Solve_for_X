"""Groups currently-active keys into connected components by feature-centroid
cosine similarity, so a genuine multi-key campaign (not just a pair) pools
its full evidence together. A pair is just a component of size 2, and a
5-key campaign sums the evidence of all five members at once.
"""
import numpy as np


def cosine_sim(c1: np.ndarray, c2: np.ndarray) -> float:
    denom = np.linalg.norm(c1) * np.linalg.norm(c2)
    return float(np.dot(c1, c2) / denom) if denom > 1e-9 else 0.0


def find_clusters(centroids: dict, sim_threshold: float) -> list:
    """centroids: {key: np.ndarray}. Returns a list of key-lists, each of
    size >= 2, one per connected component above sim_threshold."""
    keys = list(centroids.keys())
    parent = {k: k for k in keys}

    def find(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            if cosine_sim(centroids[keys[i]], centroids[keys[j]]) >= sim_threshold:
                union(keys[i], keys[j])

    groups = {}
    for k in keys:
        groups.setdefault(find(k), []).append(k)
    return [members for members in groups.values() if len(members) >= 2]
