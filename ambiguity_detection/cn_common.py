"""
Shared utilities for the clarification-need (ambiguity) experiments.


Splits: the official ClariQ train (187 topics), dev (50) and test (61)
topic sets. `load_splits` checks that no topic_id appears in two splits.

Tasks
-----
binary : label 0 = "clear" (level 1), 1 = "needs_clarification" (levels 2-4).
         POSITIVE CLASS = needs_clarification.
four   : the original four ClariQ levels 1, 2, 3, 4.

Metrics
----------------------------
Per-class precision / recall / F1 / support
"""
from __future__ import annotations

import json
import os
import random
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

SPLIT_FILES = {"train": "train.tsv", "dev": "dev.tsv", "test": "test_with_labels.tsv"}

TASKS = {
    "binary": {"labels": [0, 1], "names": ["clear", "needs_clarification"], "positive": 1},
    "four": {"labels": [1, 2, 3, 4], "names": ["level_1", "level_2", "level_3", "level_4"], "positive": None},
}


# --------------------------------------------------------------------------- data
def load_topics(path: str) -> pd.DataFrame:
    """One row per unique topic_id: topic_id, initial_request, level."""
    df = pd.read_csv(path, sep="\t")
    n_labels = df.groupby("topic_id")["clarification_need"].nunique()
    bad = n_labels[n_labels > 1]
    if len(bad):
        raise ValueError(f"{len(bad)} topics have more than one clarification_need value: {list(bad.index[:5])}")
    topics = (
        df.drop_duplicates("topic_id")[["topic_id", "initial_request", "clarification_need"]]
        .rename(columns={"clarification_need": "level"})
        .reset_index(drop=True)
    )
    topics["level"] = topics["level"].astype(int)
    topics["n_rows_in_file"] = topics["topic_id"].map(df.groupby("topic_id").size())
    return topics


def load_splits(data_dir: str) -> Dict[str, pd.DataFrame]:
    splits = {}
    for name, fname in SPLIT_FILES.items():
        path = os.path.join(data_dir, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing {path}. Run: python download_data.py --out {data_dir}")
        splits[name] = load_topics(path)
    ids = {k: set(v["topic_id"]) for k, v in splits.items()}
    for a in ids:
        for b in ids:
            if a < b and ids[a] & ids[b]:
                raise ValueError(f"topic overlap between {a} and {b}: {sorted(ids[a] & ids[b])[:5]}")
    return splits


def task_labels(levels, task: str) -> np.ndarray:
    levels = np.asarray(levels, dtype=int)
    if task == "binary":
        return (levels > 1).astype(int)
    if task == "four":
        return levels
    raise ValueError(task)


def data_summary(splits: Dict[str, pd.DataFrame]) -> dict:
    out = {"instance_definition": "one unique ClariQ topic_id (initial_request) with its single clarification_need label",
           "topic_overlap_between_splits": 0, "splits": {}}
    for name, df in splits.items():
        lv = df["level"].value_counts().sort_index()
        out["splits"][name] = {
            "n_topics": int(len(df)),
            "n_rows_in_original_file": int(df["n_rows_in_file"].sum()),
            "level_counts": {int(k): int(v) for k, v in lv.items()},
            "binary_counts": {"clear (level 1)": int((df["level"] == 1).sum()),
                              "needs_clarification (levels 2-4)": int((df["level"] > 1).sum())},
        }
    return out


# --------------------------------------------------------------------------- metrics
def compute_metrics(y_true, y_pred, task: str) -> dict:
    spec = TASKS[task]
    labels, names = spec["labels"], spec["names"]
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, average=None, zero_division=0)
    res = {
        "n": int(len(y_true)),
        "labels": labels,
        "label_names": names,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "per_class": [
            {"label": int(l), "name": n, "precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]),
             "support": int(s[i])}
            for i, (l, n) in enumerate(zip(labels, names))
        ],
        "accuracy": float((y_true == y_pred).mean()),
    }
    for avg in ("macro", "weighted"):
        ap, ar, af, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average=avg, zero_division=0)
        res[f"{avg}_precision"], res[f"{avg}_recall"], res[f"{avg}_f1"] = float(ap), float(ar), float(af)
    if spec["positive"] is not None:
        i = labels.index(spec["positive"])
        res["positive_class"] = names[i]
        res["positive_precision"], res["positive_recall"], res["positive_f1"] = float(p[i]), float(r[i]), float(f[i])
    return res


def macro_f1(y_true, y_pred, task: str) -> float:
    return float(precision_recall_fscore_support(y_true, y_pred, labels=TASKS[task]["labels"],
                                                 average="macro", zero_division=0)[2])


# --------------------------------------------------------------------------- misc
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def write_json(obj, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def write_predictions(df: pd.DataFrame, y_true, y_pred, task: str, path: str) -> None:
    out = pd.DataFrame({
        "topic_id": df["topic_id"].values,
        "initial_request": df["initial_request"].values,
        "true_level": df["level"].values,
        "true_label": np.asarray(y_true),
        "pred_label": np.asarray(y_pred),
    })
    out["task"] = task
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    out.to_csv(path, index=False)
