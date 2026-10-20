# FARD: Asking Clarifying Questions in Conversational Search

Code for *Multi-Agent Model for Asking Clarifying Questions Based on User Response in Conversational Search*, built on [ClariQ](https://github.com/aliannejadi/ClariQ).

| Folder | Purpose |
|---|---|
| `ambiguity_detection/` | Decide whether a query needs clarification (binary and four-level). Siamese-BERT vs. BERT, RoBERTa, TF-IDF and majority baselines, 5 seeds. |
| `intent_clarification/` | FARD: a multi-agent actor-critic that picks the next clarifying question. A frozen Llama-2-13B-Chat writes pseudo-turns and per-question response documents. Evaluated with MRR / NDCG@3 / NDCG@5 (5-fold topic CV, significance tests). |

## Ambiguity detection

```bash
cd ambiguity_detection
pip install -r requirements.txt
python download_models.py      # bert-base-uncased, roberta-base -> models/
python run_all.py --quick      # 1-seed smoke test
python run_all.py              # full run -> results/
```

ClariQ files are already in `data/` (`download_data.py` re-fetches them). Saved predictions and metrics are in `results/`.

## Intent clarification

```bash
cd intent_clarification
pip install -r requirements.txt
# put ClariQ train.tsv, dev.tsv, test.tsv in data_files/clariq_real/
EXTRA="--llm-local-dir ./llama-2-13b-chat-local" bash scripts/run_all.sh
```

A GPU is needed (Llama-2-13B: ~26 GB in fp16, ~14 GB with `--llm-8bit`). For one model and one seed: `ARMS="FARD" SEEDS="42" bash scripts/run_arms.sh`.

Paper results (MRR / NDCG@3 / NDCG@5): FARD 0.333 / 0.270 / 0.278 vs. best baseline MMR-BERT 0.316 / 0.256 / 0.263. All tables are in `intent_clarification/results/summary_results/`.

**Note:** this release does not yet contain `agents/`, `models/`, `rag/facet_rag.py`, `utils/seeding.py`, `docret/` and `scripts/make_tables.sh`. The training scripts import them, so add them before training.