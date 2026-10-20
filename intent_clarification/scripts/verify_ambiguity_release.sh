#!/usr/bin/env bash
# Compares the ambiguity-detection prediction files with the values printed in Tables 2-5 of the paper
# (macro-averaged precision / recall / F1 and per-class values, 3 decimals; class supports 37/126/96/39
# and 37/261). Exit status 1 if anything differs.
#
#   predictions/ambiguity/*.csv   columns: system, seed, topic_id, gold, pred
#   (written by `python -m train.ambiguity_cv`, one file per system and task)
#
# System names: Siamese, BERT (four-class); Siamese-binary, BERT-binary, MuSIc-binary (binary).
# MuSIc is an external model: its predictions must come from your own run of MuSIc.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
P=${PRED:-predictions/ambiguity}; OUT=${OUT:-results}; mkdir -p "$OUT"
status=0

# Tables 2 and 3, Figure 5 (four-class). Table 2 = Siamese; Table 3 = BERT vs Siamese.
$PY -m train.evaluate_ambiguity --pred $P/*.csv --systems Siamese BERT --expect-topics 298 \
  --out "$OUT/ambiguity_four_class" || status=$?
$PY -m train.compare_to_paper --ambiguity "$OUT/ambiguity_four_class.json" --strict \
  --out "$OUT/ambiguity_four_class_vs_paper.md" || status=$?

# Tables 4 and 5, Figure 6 (binary; positive class = ambiguous = levels 2-4)
$PY -m train.evaluate_ambiguity --pred $P/*.csv --binary --systems Siamese-binary BERT-binary MuSIc-binary \
  --expect-topics 298 --out "$OUT/ambiguity_binary" || status=$?
$PY -m train.compare_to_paper --ambiguity "$OUT/ambiguity_binary.json" --strict \
  --out "$OUT/ambiguity_binary_vs_paper.md" || status=$?

echo
[ "$status" -eq 0 ] && echo "ALL VALUES IDENTICAL TO THE PAPER (3 decimals)" || echo "SOME VALUES DIFFER FROM THE PAPER (exit status $status) -- see DIFFERS lines above"
exit "$status"
