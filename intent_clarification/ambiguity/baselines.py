

from __future__ import annotations

from collections import Counter
from typing import List

import numpy as np

from intent_clarification.data.clariq_dataset import TopicLevelExample


def target(example: TopicLevelExample, binary: bool) -> str:
    level = example.clarification_need
    return ("1" if level == 1 else "2-4") if binary else str(level)


class Majority:
    name = "Majority"

    def __init__(self, binary: bool = False, seed: int = 0):
        self.binary = binary
        self.label = "2"

    def fit(self, train: List[TopicLevelExample], val: List[TopicLevelExample]):
        counts = Counter(target(ex, self.binary) for ex in train)
        self.label = counts.most_common(1)[0][0]
        return self

    def predict(self, examples: List[TopicLevelExample]) -> List[str]:
        return [self.label] * len(examples)


class Stratified:
    name = "Random-stratified"

    def __init__(self, binary: bool = False, seed: int = 0):
        self.binary = binary
        self.rng = np.random.default_rng(seed)
        self.labels: List[str] = []
        self.probs = np.array([])

    def fit(self, train: List[TopicLevelExample], val: List[TopicLevelExample]):
        counts = Counter(target(ex, self.binary) for ex in train)
        self.labels = sorted(counts)
        self.probs = np.array([counts[c] for c in self.labels], dtype=float)
        self.probs /= self.probs.sum()
        return self

    def predict(self, examples: List[TopicLevelExample]) -> List[str]:
        return [str(x) for x in self.rng.choice(self.labels, size=len(examples), p=self.probs)]


class TfidfLogReg:
    name = "TFIDF-LogReg"

    def __init__(self, binary: bool = False, seed: int = 0):
        self.binary = binary
        self.seed = seed

    def fit(self, train: List[TopicLevelExample], val: List[TopicLevelExample]):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression

        self.vec = TfidfVectorizer(ngram_range=(1, 2), lowercase=True, sublinear_tf=True)
        x = self.vec.fit_transform([ex.initial_request for ex in train])
        y = [target(ex, self.binary) for ex in train]
        self.clf = LogisticRegression(
            C=1.0, class_weight="balanced", max_iter=1000, random_state=self.seed
        )
        self.clf.fit(x, y)
        return self

    def predict(self, examples: List[TopicLevelExample]) -> List[str]:
        x = self.vec.transform([ex.initial_request for ex in examples])
        return [str(p) for p in self.clf.predict(x)]


BASELINES = {"majority": Majority, "stratified": Stratified, "tfidf": TfidfLogReg}
