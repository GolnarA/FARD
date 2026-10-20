
# Run every model x task x seed, then the evaluation script.
#
#   python run_all.py                       # full run (5 seeds), ~2-4 hours on a laptop CPU
#   python run_all.py --quick               # 2-minute smoke test of the whole pipeline
#   python run_all.py --seeds 1 2 3         # fewer seeds

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--bert", default=None, help="Path/id of bert-base-uncased (default: models/bert-base-uncased if it exists)")
    ap.add_argument("--roberta", default=None, help="Path/id of roberta-base (default: models/roberta-base if it exists)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--tasks", nargs="+", default=["binary", "four"])
    ap.add_argument("--models", nargs="+", default=["majority", "tfidf_lr", "bert", "roberta", "siamese_bert"])
    ap.add_argument("--quick", action="store_true", help="1 seed, 1 epoch: checks that everything works")
    args = ap.parse_args()

    def pick(arg, local, hub):
        if arg:
            return arg
        full = os.path.join(HERE, local)
        return full if os.path.isdir(full) else hub

    bert = pick(args.bert, "models/bert-base-uncased", "bert-base-uncased")
    roberta = pick(args.roberta, "models/roberta-base", "roberta-base")
    out = args.out_dir or ("results_quick" if args.quick else "results")
    seeds = [1] if args.quick else args.seeds
    quick = ["--triplet-epochs", "1", "--triplets-per-epoch", "32", "--head-epochs", "5", "--ft-epochs", "1"] if args.quick else []

    specs = {
        "majority": ["--model", "majority"],
        "tfidf_lr": ["--model", "tfidf_lr"],
        "bert": ["--model", "finetune", "--encoder", bert],
        "roberta": ["--model", "finetune", "--encoder", roberta],
        "siamese_bert": ["--model", "siamese", "--encoder", bert],
    }
    jobs = [(t, m, s) for t in args.tasks for m in args.models for s in seeds]
    t_start = time.time()
    for i, (task, model, seed) in enumerate(jobs, 1):
        done = os.path.join(out, task, model, f"seed{seed}", "predictions_test.csv")
        if os.path.exists(done):
            print(f"[{i}/{len(jobs)}] skip (already done): {task} {model} seed{seed}")
            continue
        print(f"\n[{i}/{len(jobs)}] {task} {model} seed{seed}   (elapsed {(time.time() - t_start) / 60:.1f} min)", flush=True)
        cmd = [sys.executable, os.path.join(HERE, "train_models.py"), *specs[model], "--name", model, "--task", task,
               "--seed", str(seed), "--data-dir", args.data_dir, "--out-dir", out, *quick]
        subprocess.run(cmd, check=True)

    subprocess.run([sys.executable, os.path.join(HERE, "evaluate.py"), "--results-dir", out,
                    "--reference", "siamese_bert"], check=True)
    print(f"\nAll done in {(time.time() - t_start) / 60:.1f} min. Send back the folder: {out}")


if __name__ == "__main__":
    main()
