
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import tempfile
from collections import Counter

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from intent_clarification.ambiguity.baselines import Majority, Stratified  # noqa: E402
from intent_clarification.data.clariq_dataset import TopicLevelExample, load_topic_level  # noqa: E402
from intent_clarification.data.cv_splits import official_split, repeated_stratified_cv  # noqa: E402
from intent_clarification.evaluation.ambiguity import (  # noqa: E402
    FOUR_CLASS,
    average_reports,
    confusion_matrix,
    evaluate_records,
    harmonic_f1,
    run_report,
    to_binary,
)

LEVEL_COUNTS = {1: 37, 2: 126, 3: 96, 4: 39}  # all 298 ClariQ topics (Table R1)
LEVEL_WORD = {1: "uno", 2: "dos", 3: "tres", 4: "cuatro"}


def synthetic_topics(counts=None, seed=0):
    rng = np.random.default_rng(seed)
    out, tid = [], 0
    for level, n in (counts or LEVEL_COUNTS).items():
        for _ in range(n):
            words = " ".join(rng.choice(["alpha", "beta", "gamma", "delta", "eps"], size=3))
            out.append(
                TopicLevelExample(str(tid), f"tell me about {words} {LEVEL_WORD[level]}", level)
            )
            tid += 1
    return out


def test_metrics_by_hand():
    gold = ["1", "1", "2", "2", "2", "3", "4", "4"]
    pred = ["1", "2", "2", "2", "3", "3", "4", "1"]
    rep = run_report(gold, pred, FOUR_CLASS)
    pc = rep["per_class"]
    # class 1: tp=1 fp=1 fn=1 -> P=R=F1=0.5; class 2: tp=2 fp=1 fn=1 -> 2/3; class 3: tp=1 fp=1 fn=0
    # -> P=0.5 R=1 F1=2/3; class 4: tp=1 fp=0 fn=1 -> P=1 R=0.5 F1=2/3
    assert abs(pc["1"]["f1"] - 0.5) < 1e-12
    assert abs(pc["2"]["precision"] - 2 / 3) < 1e-12
    assert abs(pc["3"]["recall"] - 1.0) < 1e-12
    assert abs(pc["4"]["f1"] - 2 / 3) < 1e-12
    assert abs(rep["macro_f1"] - (0.5 + 2 / 3 + 2 / 3 + 2 / 3) / 4) < 1e-12
    assert [pc[c]["support"] for c in FOUR_CLASS] == [2, 3, 1, 2]
    cm = confusion_matrix(np.array(gold), np.array(pred), FOUR_CLASS)
    assert cm.sum() == 8 and cm[0, 0] == 1 and cm[0, 1] == 1 and cm[3, 0] == 1


def test_macro_f1_is_not_the_harmonic_mean():
    rng = np.random.default_rng(1)
    differs = 0
    for _ in range(200):
        gold = rng.choice(FOUR_CLASS, size=60, p=[0.1, 0.4, 0.3, 0.2])
        pred = np.where(rng.random(60) < 0.5, gold, rng.choice(FOUR_CLASS, size=60))
        rep = run_report(gold, pred, FOUR_CLASS)
        assert rep["macro_f1"] <= rep["harmonic_f1"] + 1e-12  # concavity of the harmonic mean
        differs += abs(rep["macro_f1"] - rep["harmonic_f1"]) > 1e-6
    assert differs > 150


def test_table_r2_arithmetic():
    # The first submission reported the harmonic mean of macro-P and macro-R (Table R2).
    assert round(harmonic_f1(0.727, 0.827), 3) == 0.774
    assert round(harmonic_f1(0.838, 0.922), 3) == 0.878


def test_binary_collapse():
    assert [to_binary(c) for c in "1234"] == ["1", "2-4", "2-4", "2-4"]


def test_average_over_runs():
    a = run_report(["1", "2", "2", "3"], ["1", "2", "3", "3"], FOUR_CLASS)
    b = run_report(["1", "2", "2", "3"], ["2", "2", "2", "3"], FOUR_CLASS)
    avg = average_reports([a, b])
    assert abs(avg["macro_f1"] - (a["macro_f1"] + b["macro_f1"]) / 2) < 1e-12
    assert np.allclose(
        avg["confusion_mean"], (np.array(a["confusion"]) + np.array(b["confusion"])) / 2
    )


def test_splits_are_topic_level_and_stratified():
    topics = synthetic_topics()
    splits = list(repeated_stratified_cv(topics, n_folds=5, repeats=3, val_fraction=0.25, seed=42))
    assert len(splits) == 15
    for repeat in range(3):
        tested = [ex.topic_id for sp in splits if sp.repeat == repeat for ex in sp.test]
        assert sorted(tested) == sorted(ex.topic_id for ex in topics)  # each topic tested once
    for sp in splits:
        ids = [ex.topic_id for part in (sp.train, sp.val, sp.test) for ex in part]
        assert len(ids) == len(set(ids)) == 298
        assert abs(len(sp.val) / (len(sp.train) + len(sp.val)) - 0.25) < 0.03
        tested = Counter(ex.clarification_need for ex in sp.test)
        for level, n in LEVEL_COUNTS.items():
            assert abs(tested[level] - n / 5) <= 1.0  # stratified
    folds_a = [sorted(ex.topic_id for ex in sp.test) for sp in splits if sp.repeat == 0]
    folds_b = [sorted(ex.topic_id for ex in sp.test) for sp in splits if sp.repeat == 1]
    assert folds_a != folds_b  # the repeats are different shuffles


def test_official_split_rejects_shared_topics():
    t = synthetic_topics({1: 4, 2: 6})
    try:
        official_split(t[:6], t[5:8], t[8:])
    except ValueError:
        return
    raise AssertionError("shared topic was not detected")


def test_baselines_and_evaluate_records():
    topics = synthetic_topics()
    sp = next(iter(repeated_stratified_cv(topics, seed=3)))
    assert set(Majority().fit(sp.train, sp.val).predict(sp.test)) == {"2"}
    preds = Stratified(seed=0).fit(sp.train, sp.val).predict(sp.test)
    assert len(preds) == len(sp.test) and set(preds) <= set(FOUR_CLASS)
    records = [
        {
            "system": "m",
            "seed": "0",
            "topic_id": ex.topic_id,
            "gold": str(ex.clarification_need),
            "pred": "2",
        }
        for ex in topics
    ]
    rep = evaluate_records(records, "m")
    assert rep["n"] == 298 and abs(rep["per_class"]["2"]["recall"] - 1.0) < 1e-12
    assert abs(rep["macro_f1"] - (2 * 126 / 298 / (126 / 298 + 1)) / 4) < 1e-9
    assert evaluate_records(records, "m", binary=True)["labels"] == ["1", "2-4"]
    for bad in (records + records[:1], records[:-1] + [dict(records[0], seed="1")]):
        try:
            evaluate_records(bad, "m")
        except ValueError:
            continue
        raise AssertionError("inconsistent records were accepted")


def test_tfidf_beats_chance_when_text_is_informative():
    try:
        import sklearn  # noqa: F401
    except ImportError:
        print("    (scikit-learn missing: skipped)")
        return
    from intent_clarification.ambiguity.baselines import TfidfLogReg

    topics = synthetic_topics()
    sp = next(iter(repeated_stratified_cv(topics, seed=3)))
    pred = TfidfLogReg(seed=0).fit(sp.train, sp.val).predict(sp.test)
    acc = np.mean([p == str(ex.clarification_need) for p, ex in zip(pred, sp.test)])
    assert acc > 0.9  # the level is written into the synthetic query


def test_cli_round_trip():
    with tempfile.TemporaryDirectory() as tmp:
        tsv = os.path.join(tmp, "toy.tsv")
        header = "topic_id\tinitial_request\ttopic_desc\tclarification_need\tfacet_id\tfacet_desc\tquestion_id\tquestion\tanswer\n"
        with open(tsv, "w", encoding="utf-8") as f:
            f.write(header)
            for ex in synthetic_topics({1: 10, 2: 20, 3: 15, 4: 10}):
                for k in range(3):  # three repeated rows per topic, as in ClariQ
                    f.write(
                        f"{ex.topic_id}\t{ex.initial_request}\tdesc\t{ex.clarification_need}\tF{ex.topic_id}{k}\t"
                        f"facet\tQ{k}\tquestion {k}\tyes\n"
                    )
        pred = os.path.join(tmp, "pred.csv")
        out = os.path.join(tmp, "eval")

        def run(*a):
            return subprocess.run(
                [sys.executable, "-m", *a], check=True, cwd=ROOT, capture_output=True
            )

        run("train.ambiguity_cv", "--data", tsv, "--model", "majority", "--out", pred)
        run(
            "train.ambiguity_cv",
            "--data",
            tsv,
            "--model",
            "stratified",
            "--out",
            pred + "2.csv",
            "--system",
            "Rand",
        )
        rows = list(csv.DictReader(open(pred)))
        assert len(rows) == 55 * 3 and len({r["topic_id"] for r in rows}) == 55
        run("train.evaluate_ambiguity", "--pred", pred, pred + "2.csv", "--out", out)
        run("train.evaluate_ambiguity", "--pred", pred, "--binary", "--out", out + "_bin")
        data = json.load(open(out + ".json"))
        assert {r["system"] for r in data["reports"]} == {"Majority", "Rand"}
        both = os.path.join(tmp, "both.csv")
        with open(both, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
            w.writerows(csv.DictReader(open(pred + "2.csv")))
        run(
            "train.significance", "--task", "classification", "--runs", both, "--reference", "Rand",
            "--against", "Majority", "--n-perm", "200", "--n-boot", "200", "--out", os.path.join(tmp, "sig"),
        )  # fmt: skip
        assert os.path.exists(os.path.join(tmp, "sig.json"))


def test_compare_to_paper_runs():
    rng = np.random.default_rng(0)
    means = {"FARD": 0.334, "QL": 0.257, "BERT-NeuQS": 0.301, "MI-Clf": 0.308, "MMR-BERT": 0.315}
    with tempfile.TemporaryDirectory() as tmp:
        runs = os.path.join(tmp, "runs.jsonl")
        with open(runs, "w") as f:
            for name, mean in means.items():
                for t in range(40):
                    v = float(np.clip(rng.normal(mean, 0.2), 0, 1))
                    rec = {"system": name, "seed": "0", "topic_id": str(t), "mrr": v}
                    rec.update({"ndcg@3": 0.8 * v, "ndcg@5": 0.83 * v})
                    f.write(json.dumps(rec) + "\n")
        run = lambda *a: subprocess.run(  # noqa: E731
            [sys.executable, "-m", *a], check=True, cwd=ROOT, capture_output=True, text=True
        )
        sig = os.path.join(tmp, "sig")
        run(
            "train.significance", "--runs", runs, "--reference", "FARD", "--against", "QL",
            "BERT-NeuQS", "MI-Clf", "MMR-BERT", "--unit", "topic", "--n-perm", "200",
            "--n-boot", "200", "--out", sig,
        )  # fmt: skip
        out = run("train.compare_to_paper", "--ranking", sig + ".json").stdout
        assert "FARD mrr" in out and "vs MMR-BERT: delta" in out
        topics = synthetic_topics({1: 10, 2: 20, 3: 15, 4: 10})
        recs = [
            {"system": "Siamese-binary", "seed": "0", "topic_id": e.topic_id,
             "gold": to_binary(e.clarification_need), "pred": "2-4"}
            for e in topics
        ]  # fmt: skip
        csv_path = os.path.join(tmp, "p.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(recs[0]))
            w.writeheader()
            w.writerows(recs)
        run("train.evaluate_ambiguity", "--pred", csv_path, "--binary", "--out", sig + "_amb")
        out = run("train.compare_to_paper", "--ambiguity", sig + "_amb.json").stdout
        assert "Siamese-binary: macro f1" in out and "class 2-4 f1" in out


def test_paper_numbers_are_consistent():
    r = subprocess.run(
        [
            sys.executable,
            "-m",
            "train.check_paper_numbers",
            "--out",
            os.path.join(tempfile.mkdtemp(), "c"),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stdout[-2000:]


def test_real_clariq_counts():
    base = os.path.join(ROOT, "data_files", "clariq_real")
    files = [os.path.join(base, n) for n in ("train.tsv", "dev.tsv", "test_with_labels.tsv")]
    if not all(os.path.exists(p) for p in files):
        print("    (ClariQ files not found in data_files/clariq_real: skipped)")
        return
    expected = [(187, [25, 74, 62, 26]), (50, [4, 21, 16, 9]), (61, [8, 31, 18, 4])]
    for path, (n, levels) in zip(files, expected):
        exs = load_topic_level(path)
        counts = Counter(e.clarification_need for e in exs)
        assert len(exs) == n and [counts[k] for k in (1, 2, 3, 4)] == levels, path


def test_siamese_bert_wiring():
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        print("    (torch / transformers missing: skipped)")
        return
    from intent_clarification.ambiguity.siamese_bert import BertClassifier, SiameseBert, TorchSettings

    topics = synthetic_topics({1: 12, 2: 24, 3: 18, 4: 10})
    sp = next(iter(repeated_stratified_cv(topics, seed=5)))
    settings = TorchSettings(
        tiny=True, device="cpu", triplet_epochs=2, classifier_epochs=2, bert_epochs=2
    )
    for cls in (SiameseBert, BertClassifier):
        first = cls(settings, seed=1).fit(sp.train, sp.val).predict(sp.test)
        again = cls(settings, seed=1).fit(sp.train, sp.val).predict(sp.test)
        assert first == again and len(first) == len(sp.test) and set(first) <= set(FOUR_CLASS)
    binary = SiameseBert(settings, binary=True, seed=1).fit(sp.train, sp.val).predict(sp.test)
    assert set(binary) <= {"1", "2-4"}


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("all ambiguity tests passed")
