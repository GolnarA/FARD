#!/usr/bin/env bash
# 5-fold topic-level CV x several training seeds for the arms in $ARMS; one
#
# Rotation k: test fold = k, validation fold = (k+1) mod 5, training = the other 3.
# Each topic is therefore tested exactly once per seed.
#
#   ARMS="FARD T9-single" SEEDS="42 43" bash scripts/run_arms.sh
set -euo pipefail

D=${D:-"data_files/clariq_real/train.tsv data_files/clariq_real/dev.tsv data_files/clariq_real/test_with_labels.tsv"}
SEEDS=${SEEDS:-"42 43 44 45 46"}
ARMS=${ARMS:?set ARMS}
LABSE=${LABSE:-"sentence-transformers/LaBSE"}
EXTRA=${EXTRA:-""}            # e.g. "--llm-local-dir ./llama-2-13b-chat-local --llm-8bit"
CKPT=${CKPT:-ckpt}; RUNS=${RUNS:-runs}; OUT=${OUT:-results}
mkdir -p "$RUNS" "$OUT"

# Pre-flight: the paper uses 298 ClariQ topics. Verify BEFORE spending GPU time.
if [[ -z "${SKIP_TOPIC_CHECK:-}" ]]; then
  python scripts/check_eval_topics.py $D --expect-topics 298 || {
    echo "The files in D do not contain 298 topics (see above). Fix D, or set SKIP_TOPIC_CHECK=1 to continue anyway." >&2
    exit 2; }
fi

# Table 9 numbering in comments. All arms use LaBSE unless stated.
arm_flags () {
  case "$1" in
    FARD)             echo "--encoder-model $LABSE" ;;                                   # Table 6 / 8 / 9 row 1 (full model)
    FARD-BERT)        echo "--encoder-model bert-base-uncased" ;;                        # Table 8
    T9-noHyp)         echo "--encoder-model $LABSE --input-variant no_hypothetical" ;;   # row 2
    T9-noRewrite)     echo "--encoder-model $LABSE --input-variant no_rewrite" ;;        # row 3
    T9-noPseudoTurn)  echo "--encoder-model $LABSE --response-mode none" ;;              # row 4
    T9-noDocs)        echo "--encoder-model $LABSE --no-facet-documents" ;;              # row 5
    T9-2win)          echo "--encoder-model $LABSE --window-sizes 1 full" ;;             # row 6
    T9-single)        echo "--encoder-model $LABSE --window-sizes full" ;;               # row 7
    T9-5xfull)        echo "--encoder-model $LABSE --window-sizes full full full full full" ;;   # row 8
    T9-singleMatched) # row 9: needs the parameter count of the full model (arm FARD must have been trained)
      local ref=${REF_MANIFEST:-"$CKPT/FARD/f${REF_FOLD:-0}_s${REF_SEED:-42}/marl_run.manifest.json"}
      [[ -f "$ref" ]] || { echo "row 9 needs $ref: train arm FARD first (or set REF_MANIFEST)" >&2; exit 1; }
      echo "--encoder-model $LABSE --window-sizes full --match-params-to $ref" ;;
    T9-supervised)    echo "--encoder-model $LABSE --objective supervised" ;;             # row 10
    T9-textOnly)      echo "--encoder-model $LABSE --window-sizes full --response-mode none --no-facet-documents --objective supervised" ;;  # row 11
    *) echo "unknown arm $1" >&2; exit 1 ;;
  esac
}

for arm in $ARMS; do
  flags=$(arm_flags "$arm")
  for s in $SEEDS; do
    for k in 0 1 2 3 4; do
      v=$(( (k + 1) % 5 )); train=""
      for f in 0 1 2 3 4; do [[ $f -ne $k && $f -ne $v ]] && train="$train $f"; done
      dir="$CKPT/$arm/f${k}_s${s}"
      if [[ ! -f "$dir/marl_system.pt" ]]; then
        python -m train.train_marl --data $D --folds $train --seed "$s" --checkpoint-dir "$dir" \
          --arm-name "$arm" $flags $EXTRA
      fi
      # the architecture, objective and input variant are restored from the checkpoint
      python -m train.evaluate_ranking --data $D --folds "$k" --checkpoint "$dir/marl_system.pt" \
        --system "$arm" --seed-label "$s" --out "$RUNS/${arm}_f${k}_s${s}.jsonl" $EXTRA
    done
  done
done
