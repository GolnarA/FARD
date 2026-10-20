# Reference implementation of the ambiguity classifiers of Section 3-3: the Siamese-BERT with
# triplet loss, and the fine-tuned BERT classifier it is compared with.


from __future__ import annotations

import os
import random
import re
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch import nn

from intent_clarification.ambiguity.baselines import target
from intent_clarification.data.clariq_dataset import TopicLevelExample
from intent_clarification.evaluation.stats import macro_f1


@dataclass
class TorchSettings:
    bert_name: str = "bert-base-uncased"
    bert_local_dir: Optional[str] = None
    tiny: bool = False  # random tiny BERT and a vocabulary from the data; wiring tests only
    device: Optional[str] = None
    max_length: int = 32
    batch_size: int = 16
    triplet_epochs: int = 5
    classifier_epochs: int = 5
    bert_epochs: int = 10
    margin: float = 0.5
    lr: float = 2e-5
    mlp_lr: float = 1e-3

    @classmethod
    def from_args(cls, args) -> "TorchSettings":
        fields = cls.__dataclass_fields__
        return cls(**{k: v for k, v in vars(args).items() if k in fields and v is not None})


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_backbone(settings: TorchSettings, texts: List[str]):
    """Return (tokenizer, bert). With settings.tiny a 2-layer random BERT is built."""
    from transformers import AutoModel, AutoTokenizer, BertConfig, BertModel, BertTokenizerFast

    if settings.tiny:
        words = sorted({w for t in texts for w in re.findall(r"\w+|[^\w\s]", t.lower())})
        vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + words
        path = os.path.join(tempfile.mkdtemp(prefix="tiny_vocab_"), "vocab.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(vocab) + "\n")
        tokenizer = BertTokenizerFast(vocab_file=path, do_lower_case=True)
        config = BertConfig(
            vocab_size=len(vocab),
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=128,
            max_position_embeddings=64,
        )
        return tokenizer, BertModel(config)
    source = settings.bert_local_dir or settings.bert_name
    return AutoTokenizer.from_pretrained(source), AutoModel.from_pretrained(source)


def mean_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * m).sum(1) / m.sum(1).clamp(min=1.0)


class QueryEncoder(nn.Module):
    """BERT -> mean pooling -> shared MLP (hidden size - 256 - 128, ReLU)."""

    def __init__(self, bert: nn.Module, mid: int = 256, out: int = 128):
        super().__init__()
        self.bert = bert
        h = bert.config.hidden_size
        self.projection = nn.Sequential(nn.Linear(h, mid), nn.ReLU(), nn.Linear(mid, out))
        self.out_dim = out

    def forward(self, input_ids, attention_mask):
        hidden = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return self.projection(mean_pool(hidden, attention_mask))


def make_head(in_dim: int, n_classes: int) -> nn.Module:
    return nn.Sequential(nn.Linear(in_dim, 64), nn.ReLU(), nn.Linear(64, n_classes))


def snapshot(*modules: nn.Module):
    return [{k: v.detach().cpu().clone() for k, v in m.state_dict().items()} for m in modules]


def restore(modules, states) -> None:
    for m, s in zip(modules, states):
        m.load_state_dict(s)


class _TorchModel:
    """Shared plumbing: tokenisation, batching, device handling."""

    def __init__(self, settings: TorchSettings, binary: bool = False, seed: int = 0):
        self.settings = settings
        self.binary = binary
        self.seed = seed
        self.device = torch.device(
            settings.device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

    def _tokenise(self, texts: List[str]):
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.settings.max_length,
            return_tensors="pt",
        )
        return enc["input_ids"].to(self.device), enc["attention_mask"].to(self.device)

    def _targets(self, examples: List[TopicLevelExample]) -> List[str]:
        return [target(ex, self.binary) for ex in examples]

    def _setup(self, train, val):
        set_seed(self.seed)
        self.classes = sorted(set(self._targets(train)))
        self.class_index = {c: i for i, c in enumerate(self.classes)}
        texts = [ex.initial_request for ex in list(train) + list(val)]
        self.tokenizer, bert = load_backbone(self.settings, texts)
        return bert

    def _labels_tensor(self, examples) -> torch.Tensor:
        return torch.tensor([self.class_index[t] for t in self._targets(examples)]).to(self.device)

    def _class_weights(self, train) -> torch.Tensor:
        counts = np.bincount([self.class_index[t] for t in self._targets(train)])
        weights = len(train) / (len(self.classes) * np.maximum(counts, 1))
        return torch.tensor(weights, dtype=torch.float).to(self.device)

    def _val_f1(self, true: List[str], pred: List[str]) -> float:
        return macro_f1(np.array(true), np.array(pred), self.classes)


class SiameseBert(_TorchModel):
    name = "Siamese"

    def fit(self, train: List[TopicLevelExample], val: List[TopicLevelExample]):
        s = self.settings
        bert = self._setup(train, val)
        self.encoder = QueryEncoder(bert).to(self.device)
        n_classes = len(self.classes)
        texts = [ex.initial_request for ex in train]
        labels = self._targets(train)
        rng = np.random.default_rng(self.seed)
        optim = torch.optim.AdamW(
            [
                {"params": self.encoder.bert.parameters(), "lr": s.lr},
                {"params": self.encoder.projection.parameters(), "lr": s.mlp_lr},
            ]
        )
        triplet = nn.TripletMarginLoss(margin=s.margin, p=2)
        best_f1, best_state = -1.0, None
        for epoch in range(s.triplet_epochs):
            self.encoder.train()
            triples = self._triplets(labels, rng)
            for start in range(0, len(triples), s.batch_size):
                batch = triples[start : start + s.batch_size]
                a, p, n = (
                    self.encoder(*self._tokenise([texts[t[k]] for t in batch])) for k in range(3)
                )
                loss = triplet(a, p, n)
                optim.zero_grad()
                loss.backward()
                optim.step()
            head = self._fit_head(train, n_classes)
            self.head = head
            f1 = self._val_f1(self._targets(val), self.predict(val)) if val else float(epoch)
            if f1 > best_f1:
                best_f1, best_state = f1, snapshot(self.encoder, head)
        restore([self.encoder, self.head], best_state)
        return self

    @staticmethod
    def _triplets(labels: List[str], rng: np.random.Generator) -> List[Tuple[int, int, int]]:
        by_class = defaultdict(list)
        for i, c in enumerate(labels):
            by_class[c].append(i)
        classes = sorted(by_class)
        out = []
        for a in rng.permutation(len(labels)):
            same = [i for i in by_class[labels[a]] if i != a] or [int(a)]
            others = [c for c in classes if c != labels[a]]
            neg = by_class[others[rng.integers(len(others))]]
            out.append(
                (int(a), int(same[rng.integers(len(same))]), int(neg[rng.integers(len(neg))]))
            )
        return out

    def _embed(self, examples: List[TopicLevelExample]) -> torch.Tensor:
        self.encoder.eval()
        chunks = []
        with torch.no_grad():
            for start in range(0, len(examples), 64):
                batch = [ex.initial_request for ex in examples[start : start + 64]]
                chunks.append(self.encoder(*self._tokenise(batch)))
        return torch.cat(chunks) if chunks else torch.zeros(0, self.encoder.out_dim)

    def _fit_head(self, train, n_classes: int) -> nn.Module:
        s = self.settings
        emb = self._embed(train)
        y = self._labels_tensor(train)
        head = make_head(self.encoder.out_dim, n_classes).to(self.device)
        optim = torch.optim.AdamW(head.parameters(), lr=s.mlp_lr)
        loss_fn = nn.CrossEntropyLoss(weight=self._class_weights(train))
        g = torch.Generator().manual_seed(self.seed)
        for _ in range(s.classifier_epochs):
            for idx in torch.randperm(len(y), generator=g).split(s.batch_size):
                idx = idx.to(self.device)
                loss = loss_fn(head(emb[idx]), y[idx])
                optim.zero_grad()
                loss.backward()
                optim.step()
        return head

    def predict(self, examples: List[TopicLevelExample]) -> List[str]:
        emb = self._embed(examples)
        self.head.eval()
        with torch.no_grad():
            idx = self.head(emb).argmax(-1).tolist() if len(examples) else []
        return [self.classes[i] for i in idx]


class BertClassifier(_TorchModel):
    name = "BERT"

    def fit(self, train: List[TopicLevelExample], val: List[TopicLevelExample]):
        s = self.settings
        bert = self._setup(train, val)
        self.bert = bert.to(self.device)
        self.out = nn.Linear(bert.config.hidden_size, len(self.classes)).to(self.device)
        optim = torch.optim.AdamW(
            list(self.bert.parameters()) + list(self.out.parameters()), lr=s.lr
        )
        loss_fn = nn.CrossEntropyLoss()
        y = self._labels_tensor(train)
        texts = [ex.initial_request for ex in train]
        g = torch.Generator().manual_seed(self.seed)
        best_f1, best_state = -1.0, None
        for epoch in range(s.bert_epochs):
            self.bert.train()
            for idx in torch.randperm(len(y), generator=g).split(s.batch_size):
                logits = self._logits([texts[i] for i in idx.tolist()])
                loss = loss_fn(logits, y[idx.to(self.device)])
                optim.zero_grad()
                loss.backward()
                optim.step()
            f1 = self._val_f1(self._targets(val), self.predict(val)) if val else float(epoch)
            if f1 > best_f1:
                best_f1, best_state = f1, snapshot(self.bert, self.out)
        restore([self.bert, self.out], best_state)
        return self

    def _logits(self, texts: List[str]) -> torch.Tensor:
        ids, mask = self._tokenise(texts)
        hidden = self.bert(input_ids=ids, attention_mask=mask).last_hidden_state
        return self.out(mean_pool(hidden, mask))

    def predict(self, examples: List[TopicLevelExample]) -> List[str]:
        self.bert.eval()
        out: List[str] = []
        with torch.no_grad():
            for start in range(0, len(examples), 64):
                texts = [ex.initial_request for ex in examples[start : start + 64]]
                out += [self.classes[i] for i in self._logits(texts).argmax(-1).tolist()]
        return out
