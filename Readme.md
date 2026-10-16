# CLARIQ-MARL: Multi-Agent Clarifying-Question Selection

Code for the paper *Multi-Agent Model for Asking Clarifying Questions Based on User Response in
Conversational Search*.

Given an ambiguous query, the system decides **which clarification question to ask next**. It is
trained and evaluated on [ClariQ](https://github.com/aliannejadi/ClariQ).

- **Agents.** A team of agents (default: 5). Each agent sees a different slice of the dialogue
  history: the last 1, 2, 3 or 4 turns, or the full history. The single-agent baseline is one agent
  with the full history.
- **Frozen LLM.** Llama-2-13B-Chat (default) is never trained. It does two things: (P1) rewrites the
  last asked question and writes a hypothetical user response (the "pseudo-turn"), and (P2) writes
  one short *response document* per clarification question, offline, via RAG over the topic's facet
  descriptions.
- **FARD encoder.** Facet-Aware Retrieval & Dialogue-turn generation. It encodes the (enriched)
  dialogue and each candidate question + document (LaBSE in the paper; `bert-base-uncased` for the
  FARD-BERT variant).
- **Selection.** Every agent scores every remaining candidate. Scores are averaged and the next
  question is sampled (training) or taken by argmax (inference).
- **Training.** Centralised-critic actor-critic (CTDE). Reward is +1 if the asked question is
  relevant to the user's hidden target facet, otherwise -1.
- **Evaluation.** MRR, NDCG@3 and NDCG@5 of the asked-question list, using 5-fold topic-level
  cross-validation, with paired randomisation tests, topic-clustered bootstrap, Cohen's d_z and
  Holm correction.

---

## 1. Status of this snapshot (read first)

This archive (`clariq_marl_code_upto_comment8`) contains **only the files changed up to reviewer
comment 8**. Several modules that the code imports are **not in it**. Copy these files over your
full `clariq_marl/` project (same-named files are complete replacements) before running anything
that trains or evaluates a model:

| Missing from this archive | Needed by |
|---|---|
| `agents/` (`agent.py`, `marl_system.py`, `windowing.py`) | `train_marl`, `evaluate_ranking`, `run_ablation`, `tune_question_docs`, smoke test |
| `models/` (`encoder.py`, `fard.py`, `critic.py`) | the same scripts, and `env/clarification_env.py` |
| `rag/facet_rag.py` | `rag/question_docs.py`, `build_question_docs`, all training/eval scripts |
| `utils/seeding.py` | all training/eval scripts |
| `tests/fake_encoder.py` | `tests/test_smoke.py` |

Also not in this archive:

- the ambiguity classifier (Siamese-BERT, `AmbiguityConfig` in `config.py`);
- the document-retrieval (KL-divergence) evaluation;
- baseline implementations (QL, BERT-NeuQS, MI-Clf, MMR-BERT). Only the scoring of their output is
  included.

**What was tested on this snapshot alone** (Python 3.13, numpy 2.5, scikit-learn 1.9, no torch):

| Check | Result |
|---|---|
| `python -m tests.test_stats` | all 8 tests pass |
| `train.score_asked_lists` then `train.significance` on synthetic data | run end to end |
| `bash -n scripts/run_comment8.sh` | syntax OK |
| `python -m tests.test_smoke` | **not runnable here** (needs torch and the missing modules) |

Run the smoke test on your complete project before the long experiments.

---

## 2. Requirements

- Python 3.9 or newer.
- `pip install -r requirements.txt` (torch, transformers, accelerate, sentence-transformers,
  scikit-learn, numpy, pandas, ...). `bitsandbytes` is optional, for `--llm-8bit`.
- A GPU for real runs. `config.py` uses CUDA when available, otherwise CPU. Llama-2-13B in fp16 needs
  about 26 GB of GPU memory, or about 14 GB with `--llm-8bit`.
- `bash` to run `scripts/run_comment8.sh` (on Windows use WSL or Git Bash).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The statistics and baseline-scoring tools need only `numpy` (and optionally `scikit-learn` for a
test).

Run every command from the project root (`clariq_marl/`).

---

## 3. Data

Download ClariQ (`train.tsv`, `dev.tsv`) from <https://github.com/aliannejadi/ClariQ> and place them
at:

```
data_files/clariq_real/train.tsv
data_files/clariq_real/dev.tsv
```

Expected tab-separated columns: `topic_id, initial_request, topic_desc, clarification_need, facet_id,
facet_desc, question_id, question, answer, conversation_context`.

How the data is used (see `data/clariq_dataset.py`):

- **Episode** = one (topic, target facet) pair. The candidate set is all questions of the topic.
  Target facets with no affirmative answer are skipped.
- **Relevance** of question *q* for target facet *f\** is 1 iff the recorded answer for (*q*, *f\**)
  starts with an affirmative word (`yes`, `yeah`, `sure`, `correct`, ...).
- **Simulated user** replies with the recorded answer. Reward is +1 for a relevant question and -1
  otherwise. An episode ends on a relevant question, after 5 turns, or when no candidate is left.
- **Folds**: `fold = topic_id % 5`. In rotation *k* the test fold is *k*, the validation fold is
  *(k+1) % 5*, and the other three folds are used for training, so each topic is tested exactly once.

Without `--data`, the scripts write a small synthetic ClariQ-shaped file for smoke runs.

---

## 4. Quick checks

```bash
python -m tests.test_stats    # metrics + significance tests (numpy only)
python -m tests.test_smoke    # full pipeline wiring, mock LLM + tiny encoder (needs the full project + torch)
```

`test_smoke` reads some files by relative path, so run it from the project root. It writes scratch
data to `/tmp/clariq_smoke/`.

A model-free smoke run of the training script (full project required):

```bash
python -m train.train_marl --llm-provider mock --no-pretrained --episodes 20
```

---

## 5. Pipeline, step by step

All steps accept `--llm-provider` and `--llm-local-dir` (see section 7).

### 5.1 Build per-question response documents (offline)

```bash
python -m train.build_question_docs \
    --data data_files/clariq_real/train.tsv data_files/clariq_real/dev.tsv \
    --llm-local-dir ./llama-2-13b-chat-local --dense-local-dir ./labse-local
```

Writes `data_files/question_documents.jsonl` plus `.stats.json` (fallback rate, for Table A1) and
`.manifest.json`. Defaults: LaBSE retriever, top-k 2, length preset 60.

Optional: pick retriever, top-k and length on the **validation** fold only (the test fold is
removed before anything is built):

```bash
python -m train.tune_question_docs --data data_files/clariq_real/train.tsv data_files/clariq_real/dev.tsv \
    --val-fold 3 --test-fold 4 --episodes 2000 --llm-local-dir ./llama-2-13b-chat-local
```

This writes `data_files/question_doc_tuning/tuning_results.json` and prints the
`build_question_docs` command for the best setting.

### 5.2 Train

The example below is rotation 4: test fold 4, validation fold 0, so it trains on folds 1, 2, 3.

```bash
python -m train.train_marl \
    --data data_files/clariq_real/train.tsv data_files/clariq_real/dev.tsv \
    --folds 1 2 3 --seed 42 --encoder-model sentence-transformers/LaBSE \
    --llm-local-dir ./llama-2-13b-chat-local --checkpoint-dir ckpt/FARD/f4_s42
```

Writes `marl_system.pt` (agents, critic, and the architecture and seed needed to evaluate it) and
`marl_run.manifest.json` into the checkpoint directory. Training stops after
`num_training_episodes` (20000) or when the 500-episode average reward reaches 0.8.

### 5.3 Evaluate a checkpoint on its test fold

```bash
python -m train.evaluate_ranking \
    --data data_files/clariq_real/train.tsv data_files/clariq_real/dev.tsv --folds 4 \
    --checkpoint ckpt/FARD/f4_s42/marl_system.pt --system FARD --seed-label 42 \
    --llm-local-dir ./llama-2-13b-chat-local --out runs/FARD_f4_s42.jsonl
```

Writes one JSON line per test episode: `system, seed, test_fold, topic_id, facet_id, asked, rels,
n_relevant, turns, mrr, ndcg@3, ndcg@5, seconds` (plus `<out>.manifest.json`).
`--ndcg-ideal all` (default) uses the standard IDCG over all relevant questions; `single` uses
IDCG = 1.

### 5.4 Score baselines with the same labels and metrics

Save each baseline's output as JSONL, one line per test episode:

```json
{"topic_id": "12", "facet_id": "F0123", "asked": ["question 1", "question 2"]}
```

(or `"ranking": [...]` for a static ranker; it is cut after the first relevant question or 5 turns).
Then convert it:

```bash
python -m train.score_asked_lists --data data_files/clariq_real/train.tsv data_files/clariq_real/dev.tsv \
    --input baselines/mmr_bert_f4.jsonl --system MMR-BERT --out runs/MMR-BERT_f4.jsonl
```

Use `--seed-label` for trainable baselines run with several seeds (default `0`).

### 5.5 Significance tests

```bash
python -m train.significance --runs "runs/*.jsonl" --reference FARD \
    --against QL BERT-NeuQS MI-Clf MMR-BERT --metrics mrr ndcg@3 ndcg@5 --out results/table6_significance
```

Writes `<out>.md` (paper-ready tables) and `<out>.json` (all numbers). It reports mean ± sd over
training seeds, then for each pair: difference, relative difference, 95% CI, randomisation p, Holm
p, Cohen's d_z with CI, and win/tie/loss counts. The `--against` list forms one Holm family.

- `--unit facet` (**default**): the metric is computed per (topic, facet) episode and the tests
  cluster by topic.
- `--unit topic`: episodes are averaged within each topic first ("topic-macro" numbers).
- `--task classification` takes a CSV with `system, seed, topic_id, gold, pred` and reports
  macro-F1, per-class scores and McNemar.
- `--n-perm` (default 100000) and `--n-boot` (default 10000) set the resampling sizes.

### 5.6 Input-leakage / ablation evaluation

```bash
python -m train.run_ablation --data ... --folds 4 --episodes 0 --arm none:docs --checkpoint ckpt/.../marl_system.pt
python -m train.run_ablation --data ... --folds 4 --episodes 200 --full   # all 6 arms, diagnostic
```

`--episodes 0` evaluates every episode once. Writes `data_files/ablation_results.json`.

---

## 6. Run the whole reviewer-comment-8 experiment

```bash
EXTRA="--llm-local-dir ./llama-2-13b-chat-local" bash scripts/run_comment8.sh
```

For each arm, seed and rotation *k*, the script trains on the three training folds (skipped if
`ckpt/<arm>/f<k>_s<seed>/marl_system.pt` already exists), evaluates on test fold *k*, and finally
runs both significance tests.

| Variable | Default | Meaning |
|---|---|---|
| `D` | `data_files/clariq_real/train.tsv data_files/clariq_real/dev.tsv` | ClariQ files |
| `SEEDS` | `42 43 44 45 46` | training seeds |
| `ARMS` | `FARD FARD-BERT FARD-noPseudoTurn FARD-noDocs SingleAgent` | model variants |
| `LABSE` | `sentence-transformers/LaBSE` | FARD encoder |
| `EXTRA` | empty | extra flags for train and evaluate, e.g. `--llm-local-dir ... --llm-8bit` |
| `CKPT`, `RUNS`, `OUT` | `ckpt`, `runs`, `results` | output folders |
| `BASELINES` | `QL BERT-NeuQS MI-Clf MMR-BERT` | baseline names expected in `runs/` |

| Arm | Flags |
|---|---|
| `FARD` | `--encoder-model LaBSE` (proposed model) |
| `FARD-BERT` | `--encoder-model bert-base-uncased` |
| `FARD-noPseudoTurn` | `--response-mode none` (no LLM pseudo-turn) |
| `FARD-noDocs` | `--no-facet-documents` (raw question text only) |
| `SingleAgent` | `--window-sizes full` (one agent, full history) |

Examples:

```bash
SEEDS="42 43 44" ARMS="FARD" bash scripts/run_comment8.sh        # main model, 3 seeds
```

**Cost:** the default is 5 arms x 5 seeds x 5 folds = **125 training runs**, each with up to 20000
episodes of LLM-enriched training. Start with one arm and one seed.

**Before the final step:** the script ends with the two significance tests. They need the baseline
runfiles in `runs/` (section 5.4), otherwise `train.significance` exits with "systems not found". Because the script uses
`set -e`, a failure of the first test also prevents the second (ablation) table. If you have no
baseline files yet, run the ablation comparison by hand:

```bash
python -m train.significance --runs "runs/*.jsonl" --reference FARD \
    --against FARD-BERT FARD-noPseudoTurn FARD-noDocs SingleAgent --unit facet \
    --metrics mrr ndcg@3 ndcg@5 --out results/ablation_significance
```

**Outputs:** `results/table6_significance.{md,json}`, `results/ablation_significance.{md,json}`,
`runs/*.jsonl` (per-episode records), `ckpt/<arm>/f<k>_s<seed>/` (checkpoints and manifests).

---

## 7. LLM backends

Select with `--llm-provider`:

| Provider | Notes |
|---|---|
| `llama_local` (default) | Llama-2-Chat via transformers, greedy decoding. Pass `--llm-local-dir`, or log in to Hugging Face (gated model). `--llm-8bit` needs `bitsandbytes`. |
| `anthropic` | Needs `ANTHROPIC_API_KEY`. **Also pass `--llm-model <model id>`**; the default model name in `config.py` is the Llama id. |
| `openai_compatible` | Any OpenAI-SDK gateway (GapGPT by default). `export GAPGPT_API_KEY=...`, optionally `--llm-base-url`, and pass `--llm-model`. |
| `mock` | Offline placeholder outputs for tests. **Never report results from it.** |

All LLM outputs are cached in `data_files/llm_cache.jsonl` (disable with `--no-llm-cache`). The key
includes the model id and a SHA-256 of the prompt, so changing a model or prompt never reuses stale
entries. Each process loads the cache once at start-up and then appends to it, so run arms
sequentially rather than in parallel, or disable the cache.

---

## 8. Offline model download

Llama 2 is gated: accept the licence on Hugging Face and log in first. Then download once and point
the scripts at the folders:

```python
from huggingface_hub import snapshot_download
snapshot_download("meta-llama/Llama-2-13b-chat-hf", local_dir="./llama-2-13b-chat-local")
snapshot_download("sentence-transformers/LaBSE",    local_dir="./labse-local")
snapshot_download("bert-base-uncased",              local_dir="./bert-base-uncased-local")
```

| Model | Flag |
|---|---|
| Llama 2 (LLM) | `--llm-local-dir ./llama-2-13b-chat-local` |
| LaBSE for the document retriever | `--dense-local-dir ./labse-local` (`build_question_docs`, `tune_question_docs`) |
| FARD encoder | `--local-dir <folder>` (`train_marl`, `evaluate_ranking`, `run_ablation`, `tune_question_docs`) |

A local encoder folder must contain `config.json`, tokenizer files (e.g. `vocab.txt`,
`tokenizer_config.json`) and `model.safetensors` or `pytorch_model.bin`. If `--encoder-model` names
a hub model, it is downloaded unless a local folder is given.

---

## 9. Inputs and leakage audit

The model must never see the target facet, relevance labels or the simulated user's answers to
questions that were not yet asked.

- The LLM prompts (`llm/prompts.py`) receive only: P1, the dialogue so far and the *last asked*
  question; P2, the topic description, one question and passages retrieved from the topic's facet
  descriptions. The corpus is the same for every target facet.
- `env/clarification_env.py::_resolve` is the **only** place that reads the simulator's target-facet
  information (`relevant_questions`, `answers_under_gold`); `step()` uses it for the real
  transition.
- `--response-mode`:
  - `hypothetical` (main model): observed history plus an LLM-generated hypothetical response.
  - `none`: observed history only, no generated response.
  - `oracle_actual`: diagnostic upper bound that peeks at the true answer to each candidate. It is
    **not deployable**. Use it only to check that the main model is not suspiciously close to it.
- `tests/test_smoke.py::test_no_leakage_audit` checks the LLM call signatures, the single read site,
  and that `oracle_peek` does not change environment state.

---

## 10. Training note: online updates

Training is online: one gradient update per dialogue turn (see `run_episode` in
`train/train_marl.py`). `MARLConfig.batch_size_episodes` is **not used**; do not cite it as the
batch size behind any result.

---

## 11. Reproducibility

- Seeds: `--seed` sets the training seed. `significance --seed` only seeds the tests.
- Every run writes a `*.manifest.json` with models, decoding parameters, seeds, retrieval
  settings, prompt hashes, library versions and SHA-256 of the input files.
- Greedy LLM decoding plus the cache means all seeds see identical LLM outputs, so seed variance
  comes only from RL training.

---

## 12. Known gotchas

- `config.py` imports `torch`, so every `train.*` script that imports it needs torch. Only
  `train.score_asked_lists`, `train.significance`, `evaluation/*` and `data/*` run without it.
- `topic_fold` uses `topic_id % 5` for numeric IDs (true for ClariQ and the synthetic data). For
  non-numeric IDs it falls back to Python's `hash()`, which changes between runs; use numeric IDs.
- `CHANGES.md` says the unit of the tests is the topic ("topic-macro"). The code default is
  `--unit facet` (per-facet episodes, clustered by topic). Pass `--unit topic` for topic-macro numbers
  and state the unit under each paper table.
- `config.py` documents `RAGConfig` / `facet_rag.py` as the legacy per-facet document path; the
  paper pipeline uses the per-question documents (`QuestionDocConfig`).

---

## 13. Project layout

```
config.py                    all hyper-parameters (dataclasses, one CONFIG object)
data/clariq_dataset.py       ClariQ loading, question bank, episodes, folds, synthetic data
env/clarification_env.py     user simulator + reward; the only reader of target-facet info
llm/prompts.py               every LLM prompt (versioned, hashed)
llm/llm_client.py            Llama-local / Anthropic / OpenAI-compatible / mock clients + disk cache
rag/question_docs.py         per-question response documents (retrieve, generate, filter, fallback)
evaluation/metrics.py        MRR, NDCG@k from the asked-question list
evaluation/stats.py          randomisation test, topic bootstrap, d_z, Holm, macro-F1, McNemar
utils/manifest.py            run manifests
train/build_question_docs.py offline document builder
train/tune_question_docs.py  document-setting selection on the validation fold
train/model_args.py          --window-sizes, --encoder-model (architecture variants)
train/train_marl.py          CTDE actor-critic training
train/evaluate_ranking.py    checkpoint -> per-episode runfile
train/score_asked_lists.py   baseline output -> runfile
train/significance.py        runfiles -> paper tables (.md/.json)
train/run_ablation.py        response-mode / document ablations
scripts/run_comment8.sh      full 5-fold x seeds x arms experiment
tests/test_stats.py          statistics tests (numpy only)
tests/test_smoke.py          pipeline smoke test (needs the full project)
CHANGES.md                   change notes up to comment 8
```