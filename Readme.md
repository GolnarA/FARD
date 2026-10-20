# FARD: Asking Clarifying Questions in Conversational Search

Code for *Multi-Agent Model for Asking Clarifying Questions Based on User Response in Conversational Search*, built on [ClariQ](https://github.com/aliannejadi/ClariQ).

| Folder | Purpose |
|---|---|
| `ambiguity_detection/` | Decide whether a query needs clarification (binary and four-level). Siamese-BERT vs. BERT 5 seeds. |
| `intent_clarification/` | FARD: a multi-agent actor-critic that picks the next clarifying question. A frozen Llama-2-13B-Chat writes pseudo-turns and per-question response documents. Evaluated with MRR / NDCG@3 / NDCG@5 (5-fold topic CV, significance tests). |

## Ambiguity detection

```bash
cd ambiguity_detection
pip install -r requirements.txt
python download_models.py      # bert-base-uncased -> models/
python run_all.py --quick      # 1-seed smoke test
python run_all.py              # full run -> results/
```

ClariQ files are already in `data/` (`download_data.py` re-fetches them). Saved predictions and metrics are in `results/`.

**Protocol note.** This folder evaluates on the official train/dev/test split with five seeds. The paper's Tables 2-5 use a different protocol: stratified 5-fold cross-validation x 3 repeats over all 298 topics, 25 % of the training topics for model selection. That protocol is implemented in `intent_clarification/train/ambiguity_cv.py` (see below); the numbers in `ambiguity_detection/results/` and in the paper are not interchangeable.

## Intent clarification

```bash
cd intent_clarification
pip install -r requirements.txt
# put ClariQ train.tsv, dev.tsv and the LABELLED test file test_with_labels.tsv in data_files/clariq_real/
EXTRA="--llm-local-dir ./llama-2-13b-chat-local" bash scripts/run_all.sh
```

A GPU is needed (Llama-2-13B: ~26 GB in fp16, ~14 GB with `--llm-8bit`). For one model and one seed: `ARMS="FARD" SEEDS="42" bash scripts/run_arms.sh`.

Paper results (MRR / NDCG@3 / NDCG@5): FARD 0.333 / 0.270 / 0.278 vs. best baseline MMR-BERT 0.316 / 0.256 / 0.263. All tables are in `intent_clarification/results/summary_results/`.

**Statistics (Section 4-2).** The unit of analysis is the topic: all episodes of a topic are averaged into one score and then averaged over the five training seeds (42-46). `train/significance.py` reports the mean difference, a paired sign-flip randomization test (100,000 permutations), a two-level percentile bootstrap CI over topics *and* training seeds (10,000 replicates), d_z, and Holm-adjusted p-values:

```bash
python -m train.significance --runs runs/*.jsonl --reference FARD \
    --against QL BERT-NeuQS MI-Clf MMR-BERT --metrics mrr ndcg@3 ndcg@5 \
    --expect-topics 298 --out results/table7_sig
python -m train.compare_to_paper --ranking results/table7_sig.json     # code output vs. the printed Tables 6-7
```

**Topics.** ClariQ has 298 topics (187/50/61 in train/dev/test). `scripts/check_eval_topics.py` reports two numbers: the topics in the files (298) and the topics that yield at least one facet episode (294 with the released files; topics 6, 229, 266 and 284 have no facet with an affirmative recorded answer). Check which of the two your runs use before you quote an n.

## Checks that need no GPU

```bash
cd intent_clarification
bash scripts/run_validation.sh
```

This runs the unit tests (`tests/test_stats.py`, `tests/test_ambiguity.py`), checks the ClariQ files against the counts of the paper (`train/check_data.py`), checks that the numbers printed in the paper agree with each other (`train/check_paper_numbers.py`) and scores three trivial baselines (majority, stratified random, TF-IDF + logistic regression) under the paper's protocol, to show which scores are easy to reach.

Ambiguity detection under the paper's protocol (stratified 5-fold x 3 repeats at topic level, 25 % of the training topics for model selection):

```bash
cd intent_clarification
D=data_files/clariq_real
python -m train.ambiguity_cv --data $D/train.tsv $D/dev.tsv $D/test_with_labels.tsv --model siamese --out runs/ambig_siamese.csv
python -m train.ambiguity_cv --data ... --model bert --out runs/ambig_bert.csv
python -m train.ambiguity_cv --data ... --model siamese --task binary --out runs/ambig_siamese_binary.csv
python -m train.evaluate_ambiguity --pred runs/ambig_siamese.csv runs/ambig_bert.csv --out results/ambiguity_four_class
python -m train.compare_to_paper --ambiguity results/ambiguity_four_class.json
```

(`train.ambiguity_cv` needs `torch` and `transformers` for `siamese` / `bert`.) Macro-F1 is the mean of the per-class F1 values, computed from the predictions of one run per repeat and averaged over the repeats.

## What this release does not contain

The following are imported by the training scripts but are not in this repository: `agents/`, `models/`, `rag/facet_rag.py`, `utils/seeding.py`, `docret/`, `scripts/make_tables.sh`, `train/evaluate_docret.py`, `evaluation/check_table.py`, and all run and prediction files (`runs/`, `predictions/`). Training and the Tables 6-10 cannot be reproduced from this repository until they are added. See `CHANGES.md` for what was aligned with the paper and what is still open.
