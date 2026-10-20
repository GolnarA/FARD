
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Sequence

import numpy as np

FOUR_CLASS = ["1", "2", "3", "4"]
BINARY = ["1", "2-4"]  # positive class of the binary task = "2-4" (ambiguous)


def to_binary(label) -> str:
    """Level 1 (clear) stays "1"; levels 2-4 (ambiguous) become "2-4". Idempotent."""
    return "1" if str(label) == "1" else "2-4"


def harmonic_f1(macro_precision: float, macro_recall: float) -> float:
    """Harmonic mean of macro-P and macro-R. NOT the macro-F1; kept only to reproduce Table R2."""
    s = macro_precision + macro_recall
    return 2 * macro_precision * macro_recall / s if s else 0.0


def confusion_matrix(gold, pred, labels: Sequence[str]) -> np.ndarray:
    """Rows = gold label, columns = predicted label, in the order of `labels`."""
    index = {str(c): i for i, c in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for g, p in zip(gold, pred):
        cm[index[str(g)], index[str(p)]] += 1
    return cm


def run_report(gold, pred, labels: Sequence[str] = FOUR_CLASS) -> dict:
    """Metrics of ONE run: per-class precision / recall / F1 / support, macro averages, accuracy."""
    gold = np.asarray([str(g) for g in gold])
    pred = np.asarray([str(p) for p in pred])
    if len(gold) != len(pred):
        raise ValueError("gold and pred must have the same length")
    labels = [str(c) for c in labels]
    unknown = (set(gold) | set(pred)) - set(labels)
    if unknown:
        raise ValueError(f"labels outside {labels}: {sorted(unknown)}")
    per_class: Dict[str, dict] = {}
    for c in labels:
        tp = int(np.sum((pred == c) & (gold == c)))
        fp = int(np.sum((pred == c) & (gold != c)))
        fn = int(np.sum((pred != c) & (gold == c)))
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        per_class[c] = {"precision": p, "recall": r, "f1": f, "support": int(np.sum(gold == c))}
    mp = float(np.mean([v["precision"] for v in per_class.values()]))
    mr = float(np.mean([v["recall"] for v in per_class.values()]))
    return {
        "labels": labels,
        "n": int(len(gold)),
        "per_class": per_class,
        "macro_precision": mp,
        "macro_recall": mr,
        "macro_f1": float(np.mean([v["f1"] for v in per_class.values()])),
        "harmonic_f1": harmonic_f1(mp, mr),
        "accuracy": float(np.mean(gold == pred)) if len(gold) else 0.0,
        "confusion": confusion_matrix(gold, pred, labels).tolist(),
    }


def average_reports(reports: List[dict]) -> dict:
    """Mean over runs (the repeats of the cross-validation); `macro_f1_sd` is the sample sd of the
    per-run macro-F1. `harmonic_f1` is computed from the averaged macro-P and macro-R."""
    if not reports:
        raise ValueError("no reports to average")
    labels = reports[0]["labels"]
    if any(r["labels"] != labels for r in reports):
        raise ValueError("reports with different label sets cannot be averaged")

    def mean(key):
        return float(np.mean([r[key] for r in reports]))

    per_class = {
        c: {
            **{
                k: float(np.mean([r["per_class"][c][k] for r in reports]))
                for k in ("precision", "recall", "f1")
            },
            "support": reports[0]["per_class"][c]["support"],
        }
        for c in labels
    }
    f1s = [r["macro_f1"] for r in reports]
    cms = np.array([r["confusion"] for r in reports], dtype=float)
    mp, mr = mean("macro_precision"), mean("macro_recall")
    return {
        "labels": labels,
        "n": reports[0]["n"],
        "n_runs": len(reports),
        "per_class": per_class,
        "macro_precision": mp,
        "macro_recall": mr,
        "macro_f1": mean("macro_f1"),
        "macro_f1_sd": float(np.std(f1s, ddof=1)) if len(f1s) > 1 else 0.0,
        "harmonic_f1": harmonic_f1(mp, mr),
        "accuracy": mean("accuracy"),
        "confusion_mean": cms.mean(axis=0).tolist(),
    }


def evaluate_records(records: List[dict], system: str, binary: bool = False) -> dict:
    """Metrics of one system from prediction records {system, seed, topic_id, gold, pred}.

    One record = one topic in one run (`seed`). Every run must contain every topic exactly once
    and the gold label of a topic must be the same in all runs; otherwise ValueError (repeated or
    missing rows would silently re-weight topics, the pseudo-replication the paper avoids).
    With binary=True gold and pred are collapsed to "1" versus "2-4".
    """
    rows = [r for r in records if str(r["system"]) == system]
    if not rows:
        raise ValueError(f"no records for system {system!r}")
    by_seed: Dict[str, Dict[str, str]] = defaultdict(dict)
    gold: Dict[str, str] = {}
    for r in rows:
        seed, tid = str(r.get("seed", "0")), str(r["topic_id"])
        g, p = str(r["gold"]), str(r["pred"])
        if binary:
            g, p = to_binary(g), to_binary(p)
        if tid in by_seed[seed]:
            raise ValueError(f"{system}: topic {tid} appears twice in run {seed}")
        if gold.setdefault(tid, g) != g:
            raise ValueError(f"{system}: topic {tid} has different gold labels in different rows")
        by_seed[seed][tid] = p
    topics = sorted(gold)
    for seed, preds in by_seed.items():
        if set(preds) != set(topics):
            raise ValueError(
                f"{system}: run {seed} has {len(preds)} topics, expected the {len(topics)} topics "
                "found in the other runs"
            )
    labels = BINARY if binary else FOUR_CLASS
    g_arr = [gold[t] for t in topics]
    reports = [
        run_report(g_arr, [by_seed[s][t] for t in topics], labels) for s in sorted(by_seed)
    ]
    out = average_reports(reports)
    out.update({"system": system, "seeds": sorted(by_seed), "binary": binary})
    return out


def format_confusion(report: dict) -> str:
    """Markdown confusion matrix: row-normalised percentages, mean counts in parentheses."""
    cm = np.asarray(report.get("confusion_mean", report.get("confusion")), dtype=float)
    labels = report["labels"]
    lines = [
        "| gold \\ predicted | " + " | ".join(labels) + " |",
        "|---|" + "---|" * len(labels),
    ]
    for i, c in enumerate(labels):
        total = cm[i].sum()
        cells = [
            f"{100 * cm[i, j] / total:.1f}% ({cm[i, j]:.1f})" if total else "-"
            for j in range(len(labels))
        ]
        lines.append(f"| {c} | " + " | ".join(cells) + " |")
    return "\n".join(lines)
