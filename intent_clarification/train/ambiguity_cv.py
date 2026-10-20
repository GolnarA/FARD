
from __future__ import annotations

import argparse
import csv
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [_ROOT, os.path.dirname(_ROOT)]  # modules are imported both flat (agents, models, ...) and as intent_clarification.*

from intent_clarification.ambiguity.baselines import BASELINES, target  # noqa: E402
from intent_clarification.data.clariq_dataset import load_topic_level  # noqa: E402
from intent_clarification.data.cv_splits import repeated_stratified_cv  # noqa: E402

TORCH_MODELS = ("siamese", "bert")


def build_model(name: str, binary: bool, seed: int, settings=None):
    if name in BASELINES:
        return BASELINES[name](binary=binary, seed=seed)
    from intent_clarification.ambiguity.siamese_bert import BertClassifier, SiameseBert

    cls = SiameseBert if name == "siamese" else BertClassifier
    return cls(settings, binary=binary, seed=seed)


def load_topics(paths):
    """One example per unique topic over all files; a topic in two files is an error."""
    topics, seen = [], {}
    for p in paths:
        for ex in load_topic_level(p):
            if ex.topic_id in seen:
                raise ValueError(f"topic {ex.topic_id} is in both {seen[ex.topic_id]} and {p}")
            seen[ex.topic_id] = p
            topics.append(ex)
    return topics


def main():
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0] or None)
    parser.add_argument("--data", nargs="+", required=True, help="ClariQ TSV files (train, dev, labelled test)")
    parser.add_argument("--model", choices=sorted(BASELINES) + list(TORCH_MODELS), required=True)
    parser.add_argument("--task", choices=["four", "binary"], default="four")
    parser.add_argument("--system", default=None, help="name written to the 'system' column")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--val-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42, help="seed of the first repeat; repeat r uses seed + r")
    parser.add_argument("--out", required=True)
    # settings of the torch models (defaults: Section 3-3 of the paper)
    parser.add_argument("--bert-name", dest="bert_name", default=None)
    parser.add_argument("--bert-local-dir", dest="bert_local_dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-length", dest="max_length", type=int, default=None)
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=None)
    parser.add_argument("--triplet-epochs", dest="triplet_epochs", type=int, default=None)
    parser.add_argument("--classifier-epochs", dest="classifier_epochs", type=int, default=None)
    parser.add_argument("--bert-epochs", dest="bert_epochs", type=int, default=None)
    parser.add_argument("--tiny", action="store_true", default=None, help="random tiny BERT; wiring tests only")
    args = parser.parse_args()

    binary = args.task == "binary"
    settings = None
    if args.model in TORCH_MODELS:
        from intent_clarification.ambiguity.siamese_bert import TorchSettings

        settings = TorchSettings.from_args(args)
    base = {
        "majority": "Majority", "stratified": "Random-stratified", "tfidf": "TFIDF-LogReg",
        "siamese": "Siamese", "bert": "BERT",
    }[args.model]  # fmt: skip
    system = args.system or base + ("-binary" if binary else "")

    topics = load_topics(args.data)
    print(f"{len(topics)} unique topics from {len(args.data)} file(s)")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    n_rows = 0
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["system", "seed", "topic_id", "gold", "pred"])
        splits = repeated_stratified_cv(
            topics, args.folds, args.repeats, args.val_fraction, args.seed
        )
        for sp in splits:
            model = build_model(args.model, binary, args.seed + sp.repeat * args.folds + sp.fold, settings)
            preds = model.fit(sp.train, sp.val).predict(sp.test)
            for ex, pred in zip(sp.test, preds):
                writer.writerow([system, args.seed + sp.repeat, ex.topic_id, target(ex, binary), pred])
                n_rows += 1
            print(f"repeat {sp.repeat} fold {sp.fold}: {len(sp.train)} train / {len(sp.val)} val / {len(sp.test)} test topics")
    print(f"Wrote {n_rows} predictions ({system}) to {args.out}")


if __name__ == "__main__":
    main()
