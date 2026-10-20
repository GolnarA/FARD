#!/usr/bin/env bash

#   predictions/ambiguity/*.csv   columns: system, seed, topic_id, gold, pred   (see predictions/README.md)
#
# System names: Siamese, BERT (four-class); Siamese-binary, BERT-binary, MuSIc-binary (binary).
set -uo pipefail
cd "$(dirname "$0")/.."
P=${PRED:-predictions/ambiguity}; OUT=${OUT:-results}; mkdir -p "$OUT"
RUN=("$P/*.csv" "$P/*.jsonl")
status=0

# Tables 2 and 3, Figure 5 (four-class). Table 2 = Siamese; Table 3 = BERT vs Siamese (macro P / R / F1).
python -m evaluation.evaluate_ambiguity --runs "${RUN[@]}" --systems Siamese BERT \
  --expect-topics 298 --expect-support 1=37 2=126 3=96 4=39 \
  --expect Siamese:1:0.213:0.234:0.222 Siamese:2:0.463:0.418:0.438 Siamese:3:0.395:0.392:0.392 \
           Siamese:4:0.275:0.333:0.301 Siamese:macro:0.336:0.344:0.338 \
           BERT:macro:0.344:0.282:0.254 BERT:1:-:0.000:- BERT:4:-:0.043:- \
  --out "$OUT/ambiguity_four_class" || status=$?

# Tables 4 and 5, Figure 6 (binary; positive class = ambiguous = levels 2-4)
python -m evaluation.evaluate_ambiguity --runs "${RUN[@]}" --systems Siamese-binary BERT-binary MuSIc-binary \
  --expect-topics 298 --expect-support unambiguous=37 ambiguous=261 \
  --expect Siamese-binary:unambiguous:0.203:0.189:0.195 Siamese-binary:ambiguous:0.886:0.895:0.891 \
           Siamese-binary:macro:0.545:0.542:0.543 BERT-binary:macro:0.509:0.511:0.497 \
           MuSIc-binary:macro:0.541:0.506:0.520 \
  --out "$OUT/ambiguity_binary" || status=$?

# Figures 5-6 and the "recovers only 18.9% of level-1 topics" sentence
python -m evaluation.confusion_check --runs "${RUN[@]}" --systems Siamese BERT \
  --out "$OUT/figure5_confusion" || status=$?

echo
[ "$status" -eq 0 ] && echo "ALL VALUES IDENTICAL TO THE PAPER (3 decimals)" || echo "SOME VALUES DIFFER FROM THE PAPER (exit status $status) -- see MISMATCH lines above"
exit "$status"
