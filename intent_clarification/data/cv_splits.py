#Topic-level splits for the ambiguity task (Section 5-1 of the paper).

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterator, List, Sequence

import numpy as np

from intent_clarification.data.clariq_dataset import TopicLevelExample, verify_no_topic_overlap


@dataclass
class Split:
    repeat: int
    fold: int
    train: List[TopicLevelExample]
    val: List[TopicLevelExample]
    test: List[TopicLevelExample]


def _by_label(examples: Sequence[TopicLevelExample]) -> Dict[int, List[TopicLevelExample]]:
    groups: Dict[int, List[TopicLevelExample]] = defaultdict(list)
    for ex in examples:
        groups[ex.clarification_need].append(ex)
    return groups


def stratified_folds(
    examples: Sequence[TopicLevelExample], n_folds: int, rng: np.random.Generator
) -> List[List[TopicLevelExample]]:
    """Split into n_folds with (almost) the same class proportions in every fold.

    Each class is shuffled and dealt round-robin; the counter carries over from one class to the
    next, so the fold sizes also differ by at most one.
    """
    folds: List[List[TopicLevelExample]] = [[] for _ in range(n_folds)]
    counter = 0
    groups = _by_label(examples)
    for label in sorted(groups):
        members = list(groups[label])
        order = rng.permutation(len(members))
        for i in order:
            folds[counter % n_folds].append(members[i])
            counter += 1
    return folds


def stratified_holdout(
    examples: Sequence[TopicLevelExample], fraction: float, rng: np.random.Generator
):
    """Return (rest, held_out). A class with two or more topics gives at least one topic to the
    held-out part and keeps at least one."""
    rest: List[TopicLevelExample] = []
    held: List[TopicLevelExample] = []
    groups = _by_label(examples)
    for label in sorted(groups):
        members = list(groups[label])
        order = rng.permutation(len(members))
        n_held = int(round(fraction * len(members)))
        if len(members) >= 2:
            n_held = min(max(n_held, 1), len(members) - 1)
        else:
            n_held = 0
        for rank, i in enumerate(order):
            (held if rank < n_held else rest).append(members[i])
    return rest, held


def repeated_stratified_cv(
    examples: Sequence[TopicLevelExample],
    n_folds: int = 5,
    repeats: int = 3,
    val_fraction: float = 0.25,
    seed: int = 42,
) -> Iterator[Split]:
    """Yield n_folds * repeats splits. Within one repeat every topic is a test topic exactly
    once."""
    ids = [ex.topic_id for ex in examples]
    if len(set(ids)) != len(ids):
        raise ValueError("expected one example per topic; use data.clariq_dataset.load_topic_level")
    for repeat in range(repeats):
        rng = np.random.default_rng(seed + repeat)
        folds = stratified_folds(examples, n_folds, rng)
        for k in range(n_folds):
            test = folds[k]
            train_all = [ex for j, fold in enumerate(folds) if j != k for ex in fold]
            train, val = stratified_holdout(train_all, val_fraction, rng)
            verify_no_topic_overlap(train, val, test)
            yield Split(repeat=repeat, fold=k, train=train, val=val, test=test)


def official_split(
    train: List[TopicLevelExample], dev: List[TopicLevelExample], test: List[TopicLevelExample]
) -> Split:
    """Official ClariQ split: model selection on dev, evaluation on test."""
    verify_no_topic_overlap(train, dev, test)
    return Split(repeat=0, fold=0, train=train, val=dev, test=test)
