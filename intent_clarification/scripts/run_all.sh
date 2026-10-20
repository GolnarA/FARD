#!/usr/bin/env bash

#   EXTRA="--llm-local-dir ./llama-2-13b-chat-local" bash scripts/run_all.sh
#   SEEDS="42" bash scripts/run_all.sh              # a one-seed dry run (the tables then have no SD)
#   DOCS=docs.jsonl POOL_PKL=data_files/clariq_real/top10k_docs_dict.pkl bash scripts/run_all.sh   # + Table 10
#                                                   # (retrieval options: DOCRET_ARGS, MU, POOL_SIZE ... see scripts/run_docret.sh)
set -euo pipefail
cd "$(dirname "$0")/.."

D=${D:-"data_files/clariq_real/train.tsv data_files/clariq_real/dev.tsv data_files/clariq_real/test.tsv"}
export D
python scripts/check_eval_topics.py $D --expect-topics 298

# run_comment8.sh and run_table9.sh repeat the check; it is already done
export SKIP_TOPIC_CHECK=1
ARMS="FARD FARD-BERT" bash scripts/run_arms.sh
ARMS="T9-noHyp T9-noRewrite T9-noPseudoTurn T9-noDocs T9-2win T9-single T9-5xfull T9-singleMatched T9-supervised T9-textOnly" \
  bash scripts/run_arms.sh

# Table 10 needs the text of the judged documents, which ClariQ does not ship: DOCS=<file> turns the step on.
# The baselines' runfiles (runs/QL*.jsonl, ...) come from train/score_asked_lists.py; a system without
# runs is left out with a warning.
if [ -n "${DOCS:-}" ]; then
  ALLOW_MISSING=1 bash scripts/run_docret.sh
else
  echo "[skip] document retrieval (Table 10): set DOCS=<file with the document texts>, see docret/README.md"
fi
bash scripts/make_tables.sh
