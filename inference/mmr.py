from typing import List
import numpy as np

def cosine_sim(a, b):
    na = np.linalg.norm(a) + 1e-8
    nb = np.linalg.norm(b) + 1e-8
    return float(np.dot(a, b) / (na * nb))

def mmr(candidates: List[str], embeddings, k=10, lambda_diversity=0.5):
    """
    Select k items balancing relevance (embedding norm as proxy) and diversity (pairwise cos-sim).
    Provide embeddings: dict name->vec.
    """
    if not candidates:
        return []
    selected = []
    remaining = set(candidates)
    # start with the item that has the largest norm
    start = max(remaining, key=lambda x: float(np.linalg.norm(embeddings[x])))
    selected.append(start); remaining.remove(start)

    while remaining and len(selected) < k:
        best = None
        best_score = -1e9
        for c in list(remaining):
            rel = float(np.linalg.norm(embeddings[c]))
            div = min([cosine_sim(embeddings[c], embeddings[s]) for s in selected])
            score = lambda_diversity * rel - (1.0 - lambda_diversity) * div
            if score > best_score:
                best_score = score
                best = c
        selected.append(best); remaining.remove(best)
    return selected
