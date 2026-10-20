#!/usr/bin/env bash
# Document retrieval : the dialogue of every system (initial request +
# asked questions + recorded answers) is one query; the documents of the topic are ranked with the
# KL-divergence model (Dirichlet smoothing) and scored with the facet-level ClariQ judgments.
# One runfile per system is written to $OUT; scripts/make_tables.sh turns them into Table 10 and
# scripts/verify_docret_release.sh compares them with the paper.
#
# NEEDS what ClariQ does not ship: the TEXT of the judged ClueWeb09/ClueWeb12 documents (your licensed
# copy or ChatNoir), one document per line: JSONL {"id","text"} or TSV id<TAB>text (docret/store.py).
#
#   DOCS=docs.jsonl POOL_PKL=data_files/clariq_real/top10k_docs_dict.pkl bash scripts/run_docret.sh
#
set -euo pipefail
cd "$(dirname "$0")/.."

CLARIQ=${CLARIQ:-data_files/clariq_real}
D=${D:-"$CLARIQ/train.tsv $CLARIQ/dev.tsv $CLARIQ/test.tsv"}
QRELS=${QRELS:-"$CLARIQ/train.qrel $CLARIQ/dev.qrel $CLARIQ/test.qrel"}
RUNS=${RUNS:-runs}
SYSTEMS=${SYSTEMS:-"FARD QL BERT-NeuQS MI-Clf MMR-BERT"}
MODE=${MODE:-kl}
MU=${MU:-2000}
DOCRET_ARGS=${DOCRET_ARGS:-}
FLAGS=""
[ -n "${ALLOW_MISSING:-}" ] && FLAGS="--allow-missing"

if [ "$MODE" = "lookup" ]; then
  OUT=${OUT:-runs_docret_official}
  : "${EVAL_PKL:?MODE=lookup needs EVAL_PKL=<single_turn_train_eval.pkl of ClariQ>}"
  python -m train.evaluate_docret --mode lookup --eval-pkl "$EVAL_PKL" --data $D --runs "$RUNS/*.jsonl" \
    --systems $SYSTEMS $FLAGS --reference-systems --out-dir "$OUT" $DOCRET_ARGS
  echo "Official single-turn lookup written to $OUT/ (first question only, train+dev topics only; NOT the Table 10 protocol)."
  exit 0
fi

OUT=${OUT:-runs_docret}
: "${DOCS:?set DOCS=<file with the document texts> (JSONL {id,text} or TSV id<TAB>text), see docret/README.md}"
[ -f "$DOCS" ] || { echo "DOCS=$DOCS not found" >&2; exit 2; }
mkdir -p "$OUT"
if [ -z "${POOL_FILE:-}" ] && [ -n "${POOL_PKL:-}" ]; then      # untar top10k_docs_dict.pkl.tar.gz first
  mkdir -p "$OUT/aux"                                             # (not in $OUT itself: its *.jsonl are the runfiles)
  POOL_FILE="$OUT/aux/pool.json"
  python -m docret.lookup --pool-pkl "$POOL_PKL" --out "$POOL_FILE"
fi
POOL_ARGS=""
[ -n "${POOL_FILE:-}" ] && POOL_ARGS="--pool-file $POOL_FILE"
[ -n "${POOL_SIZE:-}" ] && POOL_ARGS="$POOL_ARGS --pool-size $POOL_SIZE"
[ -n "${ADD_JUDGED:-}" ] && POOL_ARGS="$POOL_ARGS --add-judged"

# what the pool and the text file can reach at most (printed for the record; does not stop the run)
python -m docret.pool_coverage --data $D --qrels $QRELS --docs "$DOCS" $POOL_ARGS || true

python -m train.evaluate_docret --data $D --runs "$RUNS/*.jsonl" --systems $SYSTEMS $FLAGS \
  --qrels $QRELS --docs "$DOCS" $POOL_ARGS --mu "$MU" --include-no-question --out-dir "$OUT" $DOCRET_ARGS
echo
echo "Next:  bash scripts/make_tables.sh        (Table 10 -> results/table10_descriptive.md)"
echo "       bash scripts/verify_docret_release.sh   (compare with the values printed in the paper)"
