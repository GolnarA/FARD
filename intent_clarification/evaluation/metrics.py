# Ranking metrics for the clarification task (Section 4-3).


from __future__ import annotations

import math
from typing import Dict, Optional, Sequence


def first_relevant_rank(rels: Sequence[int]) -> Optional[int]:
    for i, r in enumerate(rels, start=1):
        if r:
            return i
    return None


def reciprocal_rank(rels: Sequence[int]) -> float:
    rank = first_relevant_rank(rels)
    return 0.0 if rank is None else 1.0 / rank


def dcg_at_k(rels: Sequence[int], k: int) -> float:
    return sum(float(r) / math.log2(i + 1) for i, r in enumerate(rels[:k], start=1))


def ndcg_at_k(rels: Sequence[int], k: int, n_relevant: int, ideal: str = "all") -> float:
    if ideal == "single":
        n_ideal = 1 if n_relevant > 0 else 0
    elif ideal == "all":
        n_ideal = min(k, n_relevant)
    else:
        raise ValueError(f"unknown ideal={ideal!r}")
    if n_ideal == 0:
        return 0.0
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, n_ideal + 1))
    return dcg_at_k(rels, k) / idcg


def episode_metrics(rels: Sequence[int], n_relevant: int, ideal: str = "all") -> Dict[str, float]:
    return {
        "mrr": reciprocal_rank(rels),
        "ndcg@3": ndcg_at_k(rels, 3, n_relevant, ideal),
        "ndcg@5": ndcg_at_k(rels, 5, n_relevant, ideal),
    }
