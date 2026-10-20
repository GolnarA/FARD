
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

from cn_common import TASKS, compute_metrics, macro_f1, write_json


def load_runs(results_dir):
    runs = {}
    for path in sorted(glob.glob(os.path.join(results_dir, "*", "*", "seed*", "predictions_test.csv"))):
        seed_dir = os.path.dirname(path)
        model_dir = os.path.dirname(seed_dir)
        task = os.path.basename(os.path.dirname(model_dir))
        if task not in TASKS:
            continue
        model, seed = os.path.basename(model_dir), os.path.basename(seed_dir)
        df = pd.read_csv(path).sort_values("topic_id").reset_index(drop=True)
        runs.setdefault(task, {}).setdefault(model, {})[seed] = df
    return runs


def bootstrap(task_runs, task, reference, n_boot, rng_seed):
    """Topic-level bootstrap of the seed-averaged macro-F1 for every model, using the SAME resamples for all."""
    models = list(task_runs)
    any_df = next(iter(task_runs[models[0]].values()))
    topics = any_df["topic_id"].values
    n = len(topics)
    for m in models:
        for s, df in task_runs[m].items():
            if not np.array_equal(df["topic_id"].values, topics):
                raise ValueError(f"{task}/{m}/{s}: test topics differ from the other runs")
    y_true = any_df["true_label"].values
    preds = {m: [df["pred_label"].values for df in task_runs[m].values()] for m in models}
    rng = np.random.default_rng(rng_seed)
    dist = {m: np.empty(n_boot) for m in models}
    for b in range(n_boot):
        ix = rng.integers(0, n, n)
        for m in models:
            dist[m][b] = np.mean([macro_f1(y_true[ix], p[ix], task) for p in preds[m]])
    point = {m: float(np.mean([macro_f1(y_true, p, task) for p in preds[m]])) for m in models}
    ci = {m: [float(np.percentile(dist[m], 2.5)), float(np.percentile(dist[m], 97.5))] for m in models}

    tests = {}
    if reference in models:
        for m in models:
            if m == reference:
                continue
            d = dist[reference] - dist[m]
            p = float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))
            tests[m] = {"diff_macro_f1": point[reference] - point[m],
                        "diff_ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                        "p_value": p}
        # Holm correction
        order = sorted(tests, key=lambda k: tests[k]["p_value"])
        running = 0.0
        for i, m in enumerate(order):
            adj = min(1.0, (len(order) - i) * tests[m]["p_value"])
            running = max(running, adj)
            tests[m]["p_holm"] = running
    return point, ci, tests


def fmt(mean, std):
    return f"{mean:.3f} ± {std:.3f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--reference", default="siamese_bert", help="Model the baselines are tested against")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--boot-seed", type=int, default=12345)
    args = ap.parse_args()

    runs = load_runs(args.results_dir)
    if not runs:
        raise SystemExit(f"No predictions found under {args.results_dir}/<task>/<model>/seed*/")
    summary, md = {}, []

    ds_path = os.path.join(args.results_dir, "data_summary.json")
    if os.path.exists(ds_path):
        ds = json.load(open(ds_path, encoding="utf-8"))
        summary["data"] = ds
        md += ["## Topic-level class counts (one instance = one unique topic)", "",
               "| Split | Topics | Rows in original file | Level 1 | Level 2 | Level 3 | Level 4 | Clear (1) | Needs clarification (2-4) |",
               "|---|---|---|---|---|---|---|---|---|"]
        for sp, d in ds["splits"].items():
            lc = d["level_counts"]
            bc = list(d["binary_counts"].values())
            md.append(f"| {sp} | {d['n_topics']} | {d['n_rows_in_original_file']} | "
                      + " | ".join(str(lc.get(str(k), lc.get(k, 0))) for k in (1, 2, 3, 4))
                      + f" | {bc[0]} | {bc[1]} |")
        md.append("")

    for task in ("binary", "four"):
        if task not in runs:
            continue
        task_runs = runs[task]
        summary[task] = {}
        names = TASKS[task]["names"]
        point, ci, tests = bootstrap(task_runs, task, args.reference, args.n_boot, args.boot_seed)

        md += [f"## Task: {task} — official ClariQ test topics", "",
               "Mean ± std over seeds. Macro-F1 = mean of per-class F1. 95% CI from topic-level bootstrap.", "",
               "| Model | Seeds | Macro-P | Macro-R | Macro-F1 | Macro-F1 95% CI | Weighted-F1 | Accuracy |",
               "|---|---|---|---|---|---|---|---|"]
        per_class_rows = []
        for model, seeds in task_runs.items():
            ms = []
            for df in seeds.values():
                ms.append(compute_metrics(df["true_label"].values, df["pred_label"].values, task))
            agg = lambda k: (float(np.mean([m[k] for m in ms])), float(np.std([m[k] for m in ms])))
            entry = {k: dict(zip(("mean", "std"), agg(k))) for k in
                     ("macro_precision", "macro_recall", "macro_f1", "weighted_precision", "weighted_recall",
                      "weighted_f1", "accuracy")}
            if task == "binary":
                for k in ("positive_precision", "positive_recall", "positive_f1"):
                    entry[k] = dict(zip(("mean", "std"), agg(k)))
            entry["per_class"] = []
            for i, nm in enumerate(names):
                pc = {k: dict(zip(("mean", "std"), (float(np.mean([m["per_class"][i][k] for m in ms])),
                                                     float(np.std([m["per_class"][i][k] for m in ms])))))
                      for k in ("precision", "recall", "f1")}
                pc["name"], pc["support"] = nm, ms[0]["per_class"][i]["support"]
                entry["per_class"].append(pc)
                per_class_rows.append(f"| {model} | {nm} | {fmt(**pc['precision'])} | {fmt(**pc['recall'])} | "
                                      f"{fmt(**pc['f1'])} | {pc['support']} |")
            cm = np.sum([m["confusion_matrix"] for m in ms], axis=0)
            entry["confusion_matrix_summed_over_seeds"] = cm.tolist()
            entry["per_seed"] = {s: m for s, m in zip(seeds, ms)}
            entry["n_seeds"] = len(ms)
            entry["bootstrap_macro_f1_seed_avg"] = {"point": point[model], "ci95": ci[model]}
            if model in tests:
                entry["paired_bootstrap_vs_reference"] = tests[model]
            summary[task][model] = entry
            pd.DataFrame(cm, index=[f"true_{n}" for n in names], columns=[f"pred_{n}" for n in names]).to_csv(
                os.path.join(args.results_dir, f"confusion_{task}_{model}.csv"))
            md.append(f"| {model} | {len(ms)} | {fmt(**entry['macro_precision'])} | {fmt(**entry['macro_recall'])} | "
                      f"{fmt(**entry['macro_f1'])} | [{ci[model][0]:.3f}, {ci[model][1]:.3f}] | "
                      f"{fmt(**entry['weighted_f1'])} | {fmt(**entry['accuracy'])} |")
        md.append("")
        if task == "binary":
            md += ["Positive class = needs_clarification (levels 2-4).", "",
                   "| Model | Positive-P | Positive-R | Positive-F1 |", "|---|---|---|---|"]
            for model, e in summary[task].items():
                md.append(f"| {model} | {fmt(**e['positive_precision'])} | {fmt(**e['positive_recall'])} | "
                          f"{fmt(**e['positive_f1'])} |")
            md.append("")
        md += [f"### Per-class results ({task})", "", "| Model | Class | Precision | Recall | F1 | Support |",
               "|---|---|---|---|---|---|"] + per_class_rows + [""]
        if tests:
            md += [f"### Paired topic-level bootstrap vs. {args.reference} ({task}, {args.n_boot} resamples)", "",
                   "| Baseline | Δ macro-F1 (ref − baseline) | 95% CI of Δ | p | p (Holm) |", "|---|---|---|---|---|"]
            for m, t in tests.items():
                md.append(f"| {m} | {t['diff_macro_f1']:+.3f} | [{t['diff_ci95'][0]:+.3f}, {t['diff_ci95'][1]:+.3f}] | "
                          f"{t['p_value']:.3f} | {t['p_holm']:.3f} |")
            md.append("")

    write_json(summary, os.path.join(args.results_dir, "summary.json"))
    with open(os.path.join(args.results_dir, "summary_tables.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print("\n".join(md))
    print(f"\nWrote {args.results_dir}/summary.json and {args.results_dir}/summary_tables.md")


if __name__ == "__main__":
    main()
