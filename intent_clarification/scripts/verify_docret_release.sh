#!/usr/bin/env bash
# Compares the document-retrieval runfiles ($DOCRET/*.jsonl, written by scripts/run_docret.sh) with the
# values printed in Table 10 of the paper (MRR / P@1 / NDCG@1 / NDCG@5; mean over the topics, then over
# the seeds). Exit status 1 if any value differs after rounding to 3 decimals; 2 if a system or metric
# is missing from the runfiles.
#
#   N_TOPICS=294 bash scripts/verify_docret_release.sh     # also require this number of evaluated topics
#
# A difference means that your retrieval set-up (document pool, Dirichlet mu, tokenization, query
# construction; see docret/README.md) is not the one that produced the paper's numbers -- it is a finding
# about the set-up, not a bug of the comparison.
set -uo pipefail
cd "$(dirname "$0")/.."
DOCRET=${DOCRET:-runs_docret}
python -m evaluation.check_table --runs "$DOCRET/*.jsonl" ${N_TOPICS:+--expect-topics "$N_TOPICS"} --expect \
  QL:mrr:0.293 QL:p@1:0.211 QL:ndcg@1:0.148 QL:ndcg@5:0.147 \
  BERT-NeuQS:mrr:0.303 BERT-NeuQS:p@1:0.216 BERT-NeuQS:ndcg@1:0.151 BERT-NeuQS:ndcg@5:0.153 \
  MMR-BERT:mrr:0.308 MMR-BERT:p@1:0.220 MMR-BERT:ndcg@1:0.154 MMR-BERT:ndcg@5:0.157 \
  MI-Clf:mrr:0.310 MI-Clf:p@1:0.219 MI-Clf:ndcg@1:0.152 MI-Clf:ndcg@5:0.158 \
  FARD:mrr:0.314 FARD:p@1:0.225 FARD:ndcg@1:0.156 FARD:ndcg@5:0.158
status=$?
echo
case "$status" in
  0) echo "ALL VALUES OF TABLE 10 IDENTICAL TO THE PAPER (3 decimals)";;
  1) echo "SOME VALUES DIFFER FROM TABLE 10 (exit status 1) -- see MISMATCH lines above";;
  *) echo "A SYSTEM OR METRIC IS MISSING (exit status $status) -- see the EXPECT lines above";;
esac
exit "$status"
