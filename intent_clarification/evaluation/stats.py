# Significance tests for the ranking and classification results.


from __future__ import annotations

import csv
import itertools
import json
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


# Runfiles
def load_records(paths: Iterable[str]) -> List[dict]:
    """Read JSONL files, or CSV/TSV files with a header. Needed keys: system, seed, topic_id,
    plus the metric columns (ranking) or gold and pred (classification)."""
    out: List[dict] = []
    for p in paths:
        if p.endswith(".jsonl") or p.endswith(".json"):
            with open(p, encoding="utf-8") as f:
                out.extend(json.loads(line) for line in f if line.strip())
        else:
            delim = "\t" if p.endswith(".tsv") else ","
            with open(p, encoding="utf-8", newline="") as f:
                out.extend(dict(r) for r in csv.DictReader(f, delimiter=delim))
    for r in out:
        r["system"] = str(r["system"])
        r["seed"] = str(r.get("seed", "0"))
        r["topic_id"] = str(r["topic_id"])
    return out


def systems_in(records: List[dict]) -> List[str]:
    return sorted({r["system"] for r in records})


# Aggregation
def per_seed_topic_scores(
    records: List[dict], system: str, metric: str
) -> Dict[str, Dict[str, float]]:
    """seed -> topic -> mean metric over the topic's episodes."""
    acc: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    for r in records:
        if r["system"] == system and metric in r and r[metric] not in ("", None):
            acc[(r["seed"], r["topic_id"])].append(float(r[metric]))
    out: Dict[str, Dict[str, float]] = defaultdict(dict)
    for (seed, topic), vals in acc.items():
        out[seed][topic] = float(np.mean(vals))
    return dict(out)


def topic_scores(records: List[dict], system: str, metric: str) -> Dict[str, float]:
    """topic -> mean over seeds of the per-seed topic score."""
    by_seed = per_seed_topic_scores(records, system, metric)
    acc: Dict[str, List[float]] = defaultdict(list)
    for scores in by_seed.values():
        for t, v in scores.items():
            acc[t].append(v)
    return {t: float(np.mean(v)) for t, v in acc.items()}


def seed_summary(records: List[dict], system: str, metric: str) -> dict:
    by_seed = per_seed_topic_scores(records, system, metric)
    vals = np.array([np.mean(list(s.values())) for _, s in sorted(by_seed.items())])
    return {
        "n_seeds": int(len(vals)),
        "mean": float(vals.mean()) if len(vals) else float("nan"),
        "sd": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
        "per_seed": {seed: float(np.mean(list(s.values()))) for seed, s in sorted(by_seed.items())},
        "n_topics": int(len(topic_scores(records, system, metric))),
    }


# Tests on paired differences
def paired_randomization_test(
    d: np.ndarray, n_resamples: int = 100_000, rng: Optional[np.random.Generator] = None
) -> float:
    """Two-sided sign-flip test of H0: the two systems are exchangeable within every unit
    (mean difference 0)."""
    d = np.asarray(d, dtype=float)
    n = len(d)
    if n == 0:
        return float("nan")
    obs = abs(d.mean())
    eps = 1e-12
    if n <= 20:  # small n: enumerate all 2^n sign patterns
        signs = np.array(list(itertools.product([-1.0, 1.0], repeat=n)))
        stats = np.abs((signs * d).mean(axis=1))
        return float(np.mean(stats >= obs - eps))
    rng = rng or np.random.default_rng(0)
    count, done, chunk = 0, 0, 10_000
    while done < n_resamples:
        m = min(chunk, n_resamples - done)
        signs = rng.choice([-1.0, 1.0], size=(m, n))
        count += int(np.sum(np.abs((signs * d).mean(axis=1)) >= obs - eps))
        done += m
    return float((count + 1) / (n_resamples + 1))


def cohens_dz(d: np.ndarray) -> float:
    d = np.asarray(d, dtype=float)
    if len(d) < 2:
        return float("nan")
    sd = d.std(ddof=1)
    if sd == 0:
        return 0.0 if d.mean() == 0 else float("inf") * np.sign(d.mean())
    return float(d.mean() / sd)


def topic_bootstrap(
    d: np.ndarray,
    n_resamples: int = 10_000,
    alpha: float = 0.05,
    rng: Optional[np.random.Generator] = None,
) -> dict:
    """Percentile CIs for the mean difference and for d_z, resampling topics."""
    d = np.asarray(d, dtype=float)
    rng = rng or np.random.default_rng(1)
    n = len(d)
    idx = rng.integers(0, n, size=(n_resamples, n))
    samples = d[idx]
    means = samples.mean(axis=1)
    sds = samples.std(axis=1, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        dz = np.where(sds > 0, means / sds, 0.0)
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return {
        "ci_low": float(np.percentile(means, lo)),
        "ci_high": float(np.percentile(means, hi)),
        "dz_ci_low": float(np.percentile(dz, lo)),
        "dz_ci_high": float(np.percentile(dz, hi)),
    }


def holm(pvalues: Sequence[float]) -> List[float]:
    """Holm-Bonferroni adjusted p-values, in the order of the input."""
    p = np.asarray(pvalues, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj.tolist()


def compare_ranking(
    records: List[dict],
    system_a: str,
    system_b: str,
    metric: str,
    n_perm: int = 100_000,
    n_boot: int = 10_000,
    seed: int = 0,
) -> dict:
    """Topic unit: one averaged score per topic."""
    sa, sb = topic_scores(records, system_a, metric), topic_scores(records, system_b, metric)
    topics = sorted(set(sa) & set(sb))
    if not topics:
        raise ValueError(f"no common topics for {system_a} vs {system_b} on {metric}")
    a = np.array([sa[t] for t in topics])
    b = np.array([sb[t] for t in topics])
    d = a - b
    rng = np.random.default_rng(seed)
    boot = topic_bootstrap(d, n_boot, rng=rng)
    return {
        "metric": metric,
        "system_a": system_a,
        "system_b": system_b,
        "n_topics": len(topics),
        "only_in_a": len(set(sa) - set(sb)),
        "only_in_b": len(set(sb) - set(sa)),
        "mean_a": float(a.mean()),
        "mean_b": float(b.mean()),
        "diff": float(d.mean()),
        "rel_diff_pct": float(100 * d.mean() / b.mean()) if b.mean() else float("nan"),
        **boot,
        "p_value": paired_randomization_test(d, n_perm, rng=rng),
        "cohens_dz": cohens_dz(d),
        "wins": int(np.sum(d > 1e-12)),
        "ties": int(np.sum(np.abs(d) <= 1e-12)),
        "losses": int(np.sum(d < -1e-12)),
        "seeds_a": seed_summary(records, system_a, metric),
        "seeds_b": seed_summary(records, system_b, metric),
    }


# Facet unit with topic-clustered inference
def facet_scores(records: List[dict], system: str, metric: str) -> Dict[Tuple[str, str], float]:
    """(topic, facet) -> metric averaged over seeds. One episode is one facet."""
    acc: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    for r in records:
        if r["system"] == system and metric in r and r[metric] not in ("", None):
            acc[(r["topic_id"], str(r.get("facet_id", "")))].append(float(r[metric]))
    return {k: float(np.mean(v)) for k, v in acc.items()}


def compare_ranking_facet(
    records: List[dict],
    system_a: str,
    system_b: str,
    metric: str,
    n_perm: int = 100_000,
    n_boot: int = 10_000,
    seed: int = 0,
) -> dict:
    """The metric is the mean over facet episodes (as in Tables 6-11). Facets of a topic share
    the initial query, so inference is clustered by topic:
      * randomisation test: the two systems are swapped for all facets of a topic together
      * bootstrap: topics are resampled, each with all its facets
    """
    fa, fb = facet_scores(records, system_a, metric), facet_scores(records, system_b, metric)
    keys = sorted(set(fa) & set(fb))
    if not keys:
        raise ValueError(f"no common facets for {system_a} vs {system_b} on {metric}")
    a = np.array([fa[k] for k in keys])
    b = np.array([fb[k] for k in keys])
    d = a - b
    topics = sorted({k[0] for k in keys})
    tix = {t: i for i, t in enumerate(topics)}
    cl = np.array([tix[k[0]] for k in keys])
    S = np.bincount(cl, weights=d, minlength=len(topics))  # per-topic sum of differences
    n_t = np.bincount(cl, minlength=len(topics)).astype(float)  # facets per topic
    N = n_t.sum()
    obs = d.mean()
    rng = np.random.default_rng(seed)
    count, done = 0, 0
    while done < n_perm:
        m = min(10_000, n_perm - done)
        signs = rng.choice([-1.0, 1.0], size=(m, len(topics)))
        count += int(np.sum(np.abs(signs @ S / N) >= abs(obs) - 1e-12))
        done += m
    p = (count + 1) / (n_perm + 1)
    idx = rng.integers(0, len(topics), size=(n_boot, len(topics)))
    boot = S[idx].sum(axis=1) / n_t[idx].sum(axis=1)
    # effect size on topic-level mean differences, which are independent
    dt = S / n_t
    dz_boot = []
    for row in idx[: min(n_boot, 5000)]:
        x = dt[row]
        sd = x.std(ddof=1)
        dz_boot.append(x.mean() / sd if sd > 0 else 0.0)
    return {
        "metric": metric,
        "system_a": system_a,
        "system_b": system_b,
        "unit": "facet (clustered by topic)",
        "n_facets": len(keys),
        "n_topics": len(topics),
        "mean_a": float(a.mean()),
        "mean_b": float(b.mean()),
        "diff": float(obs),
        "rel_diff_pct": float(100 * obs / b.mean()) if b.mean() else float("nan"),
        "ci_low": float(np.percentile(boot, 2.5)),
        "ci_high": float(np.percentile(boot, 97.5)),
        "p_value": float(p),
        "cohens_dz": cohens_dz(dt),
        "dz_ci_low": float(np.percentile(dz_boot, 2.5)),
        "dz_ci_high": float(np.percentile(dz_boot, 97.5)),
        "wins": int(np.sum(d > 1e-12)),
        "ties": int(np.sum(np.abs(d) <= 1e-12)),
        "losses": int(np.sum(d < -1e-12)),
        "seeds_a": seed_summary_facet(records, system_a, metric),
        "seeds_b": seed_summary_facet(records, system_b, metric),
    }


def seed_summary_facet(records: List[dict], system: str, metric: str) -> dict:
    by_seed: Dict[str, List[float]] = defaultdict(list)
    for r in records:
        if r["system"] == system and metric in r and r[metric] not in ("", None):
            by_seed[r["seed"]].append(float(r[metric]))
    vals = np.array([np.mean(v) for _, v in sorted(by_seed.items())])
    return {
        "n_seeds": int(len(vals)),
        "mean": float(vals.mean()) if len(vals) else float("nan"),
        "sd": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
        "n_topics": len({r["topic_id"] for r in records if r["system"] == system}),
    }


def run_family(
    records: List[dict],
    pairs: Sequence[Tuple[str, str]],
    metrics: Sequence[str],
    n_perm: int = 100_000,
    n_boot: int = 10_000,
    seed: int = 0,
    unit: str = "facet",
) -> List[dict]:
    """Run every comparison, with the Holm correction applied per metric over `pairs`.
    unit="facet": per-facet metric, topic-clustered tests (default); unit="topic": one score
    per topic."""
    fn = compare_ranking_facet if unit == "facet" else compare_ranking
    results = []
    for metric in metrics:
        block = [fn(records, a, b, metric, n_perm, n_boot, seed) for a, b in pairs]
        for r, p_adj in zip(block, holm([r["p_value"] for r in block])):
            r["p_holm"] = p_adj
            r["family_size"] = len(block)
        results.extend(block)
    return results


# Classification (ambiguity detection, one prediction per topic)
def per_class_prf(gold: np.ndarray, pred: np.ndarray, labels: Sequence) -> dict:
    rows = {}
    for c in labels:
        tp = int(np.sum((pred == c) & (gold == c)))
        fp = int(np.sum((pred == c) & (gold != c)))
        fn = int(np.sum((pred != c) & (gold == c)))
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        rows[str(c)] = {"precision": p, "recall": r, "f1": f, "support": int(np.sum(gold == c))}
    return rows


def macro_f1(gold: np.ndarray, pred: np.ndarray, labels: Sequence) -> float:
    """Mean of the per-class F1 scores (not the harmonic mean of macro-P and macro-R)."""
    return float(np.mean([v["f1"] for v in per_class_prf(gold, pred, labels).values()]))


def _class_matrix(records: List[dict], system: str) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """Return topics, gold[topic] and pred[seed, topic], with seeds in sorted order."""
    by_seed: Dict[str, Dict[str, str]] = defaultdict(dict)
    gold: Dict[str, str] = {}
    for r in records:
        if r["system"] == system:
            by_seed[r["seed"]][r["topic_id"]] = str(r["pred"])
            gold[r["topic_id"]] = str(r["gold"])
    topics = sorted(gold)
    preds = np.array([[s.get(t, "") for t in topics] for _, s in sorted(by_seed.items())])
    return topics, np.array([gold[t] for t in topics]), preds


def compare_classification(
    records: List[dict],
    system_a: str,
    system_b: str,
    n_perm: int = 10_000,
    n_boot: int = 10_000,
    seed: int = 0,
) -> dict:
    ta, ga, pa = _class_matrix(records, system_a)
    tb, gb, pb = _class_matrix(records, system_b)
    if ta != tb or not np.array_equal(ga, gb):
        raise ValueError(
            "the two systems must be evaluated on the same topics with the same gold labels"
        )
    if pa.shape[0] != pb.shape[0]:
        raise ValueError("both systems need the same number of seeds")
    gold = ga
    labels = sorted(set(gold.tolist()))

    def stat(PA, PB, idx=None):
        g = gold if idx is None else gold[idx]
        A = PA if idx is None else PA[:, idx]
        B = PB if idx is None else PB[:, idx]
        fa = np.mean([macro_f1(g, A[s], labels) for s in range(A.shape[0])])
        fb = np.mean([macro_f1(g, B[s], labels) for s in range(B.shape[0])])
        return fa, fb

    fa, fb = stat(pa, pb)
    obs = abs(fa - fb)
    rng = np.random.default_rng(seed)
    n = len(gold)
    count = 0
    for _ in range(n_perm):
        swap = rng.random(n) < 0.5
        A = np.where(swap[None, :], pb, pa)
        B = np.where(swap[None, :], pa, pb)
        x, y = stat(A, B)
        count += abs(x - y) >= obs - 1e-12
    p_perm = (count + 1) / (n_perm + 1)

    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        x, y = stat(pa, pb, idx)
        diffs.append(x - y)
    diffs = np.array(diffs)

    # exact McNemar test on per-topic correctness, using the first seed of each system
    ca, cb = pa[0] == gold, pb[0] == gold
    n01, n10 = int(np.sum(~ca & cb)), int(np.sum(ca & ~cb))
    p_mcnemar = _binom_two_sided(min(n01, n10), n01 + n10)

    return {
        "system_a": system_a,
        "system_b": system_b,
        "n_topics": n,
        "n_seeds": int(pa.shape[0]),
        "macro_f1_a": float(fa),
        "macro_f1_b": float(fb),
        "diff": float(fa - fb),
        "ci_low": float(np.percentile(diffs, 2.5)),
        "ci_high": float(np.percentile(diffs, 97.5)),
        "p_value": float(p_perm),
        "mcnemar_b": n10,
        "mcnemar_c": n01,
        "p_mcnemar": p_mcnemar,
        "per_class_a": per_class_prf(gold, pa[0], labels),
        "per_class_b": per_class_prf(gold, pb[0], labels),
        "class_support": {str(c): int(np.sum(gold == c)) for c in labels},
    }


def _binom_two_sided(k: int, n: int) -> float:
    if n == 0:
        return 1.0
    from math import comb

    tail = sum(comb(n, i) for i in range(0, k + 1)) / 2**n
    return float(min(1.0, 2 * tail))
