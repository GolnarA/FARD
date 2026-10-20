#!/usr/bin/env bash
# Everything that can be checked without a GPU, a model download or an API key:
#   1. the unit tests;
#   2. the ClariQ files against the counts in the paper;
#   3. the numbers reported in the paper against each other;
#   4. trivial baselines (majority, random, TF-IDF) under the paper's ambiguity protocol, which
#      show what scores are easy to reach;
#   5. if you have produced your own Siamese / BERT predictions (runs/ambig_*.csv), their metrics,
#      the significance test between them, and the comparison with the paper.
#
#   bash scripts/run_validation.sh
#   DATA_DIR=/path/to/clariq PY=python3 bash scripts/run_validation.sh
set -euo pipefail

DATA_DIR=${DATA_DIR:-data_files/clariq_real}
OUT=${OUT:-results/validation}
PY=${PY:-python}
DATA="$DATA_DIR/train.tsv $DATA_DIR/dev.tsv $DATA_DIR/test_with_labels.tsv"
mkdir -p "$OUT/predictions"

$PY -m tests.test_stats
$PY -m tests.test_ambiguity

$PY -m train.check_data --train "$DATA_DIR/train.tsv" --dev "$DATA_DIR/dev.tsv" \
  --test "$DATA_DIR/test_with_labels.tsv" --out "$OUT/data_check"
$PY -m train.check_paper_numbers --out "$OUT/paper_numbers_check"

for m in majority stratified tfidf; do
  $PY -m train.ambiguity_cv --data $DATA --model "$m" --out "$OUT/predictions/ambig_$m.csv"
done
$PY -m train.ambiguity_cv --data $DATA --model tfidf --task binary \
  --out "$OUT/predictions/ambig_tfidf_binary.csv"

PRED="$OUT/predictions/ambig_majority.csv $OUT/predictions/ambig_stratified.csv $OUT/predictions/ambig_tfidf.csv"
$PY -m train.evaluate_ambiguity --pred $PRED --out "$OUT/ambiguity_baselines_four_class"
$PY -m train.evaluate_ambiguity --pred $PRED "$OUT/predictions/ambig_tfidf_binary.csv" --binary \
  --out "$OUT/ambiguity_baselines_binary"

# Your own results (see section 13 of the README for how to produce them).
if [[ -f runs/ambig_siamese.csv && -f runs/ambig_bert.csv ]]; then
  $PY -m train.evaluate_ambiguity --pred runs/ambig_siamese.csv runs/ambig_bert.csv \
    --out results/ambiguity_four_class
  $PY -m train.significance --task classification --runs runs/ambig_siamese.csv runs/ambig_bert.csv \
    --reference Siamese --against BERT --out results/ambiguity_four_class_significance
  $PY -m train.compare_to_paper --ambiguity results/ambiguity_four_class.json \
    --out results/ambiguity_four_class_vs_paper.md
fi
if [[ -f runs/ambig_siamese_binary.csv && -f runs/ambig_bert_binary.csv ]]; then
  $PY -m train.evaluate_ambiguity --pred runs/ambig_siamese_binary.csv runs/ambig_bert_binary.csv \
    --binary --out results/ambiguity_binary
  $PY -m train.significance --task classification \
    --runs runs/ambig_siamese_binary.csv runs/ambig_bert_binary.csv \
    --reference Siamese-binary --against BERT-binary --out results/ambiguity_binary_significance
  $PY -m train.compare_to_paper --ambiguity results/ambiguity_binary.json \
    --out results/ambiguity_binary_vs_paper.md
fi
echo "Done. Reports are in $OUT/"
