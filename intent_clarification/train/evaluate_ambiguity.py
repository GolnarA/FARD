# Score ambiguity-level predictions: per-class precision / recall / F1 / support, macro averages,
# confusion matrices (Tables 2-5, Figures 5-6 of the revised manuscript).
#
#     python -m train.evaluate_ambiguity --pred runs/ambig_siamese.csv runs/ambig_bert.csv \
#         --out results/ambiguity_siamese_vs_bert


from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from intent_clarification.evaluation.ambiguity import evaluate_records, format_confusion  # noqa: E402
from intent_clarification.evaluation.stats import load_records, systems_in  # noqa: E402


def per_class_table(report: dict) -> str:
    lines = [
        "| Class | Precision | Recall | F1 | Support |",
        "|---|---|---|---|---|",
    ]
    for c in report["labels"]:
        v = report["per_class"][c]
        lines.append(
            f"| {c} | {v['precision']:.3f} | {v['recall']:.3f} | {v['f1']:.3f} | {v['support']} |"
        )
    lines.append(
        f"| **Macro average** | {report['macro_precision']:.3f} | {report['macro_recall']:.3f} | "
        f"{report['macro_f1']:.3f} | {report['n']} |"
    )
    return "\n".join(lines)


def summary_table(reports: list, with_harmonic: bool) -> str:
    head = "| System | Macro precision | Macro recall | Macro F1 | Accuracy |"
    sep = "|---|---|---|---|---|"
    if with_harmonic:
        head += " Harmonic mean of macro P and R |"
        sep += "---|"
    lines = [head, sep]
    for r in reports:
        row = (
            f"| {r['system']} | {r['macro_precision']:.3f} | {r['macro_recall']:.3f} | "
            f"{r['macro_f1']:.3f} ± {r['macro_f1_sd']:.3f} | {r['accuracy']:.3f} |"
        )
        if with_harmonic:
            row += f" {r['harmonic_f1']:.3f} |"
        lines.append(row)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--pred", nargs="+", required=True, help="prediction CSV / JSONL files")
    parser.add_argument("--systems", nargs="+", default=None, help="default: every system found")
    parser.add_argument("--binary", action="store_true", help="collapse to level 1 vs 2-4")
    parser.add_argument("--reproduce-r2", action="store_true")
    parser.add_argument("--out", required=True, help="output prefix (.json and .md are written)")
    args = parser.parse_args()

    records = load_records(args.pred)
    systems = args.systems or systems_in(records)
    reports = [evaluate_records(records, s, binary=args.binary) for s in systems]
    task = "binary (level 1 vs 2-4, positive class = 2-4)" if args.binary else "four-class"

    md = [f"## Ambiguity detection, {task}", "", summary_table(reports, args.reproduce_r2), ""]
    md.append(
        f"Macro-F1 is the mean of the per-class F1 values, averaged over {reports[0]['n_runs']} "
        f"run(s) of {reports[0]['n']} topics (± = sd over runs)."
    )
    for r in reports:
        md += ["", f"### {r['system']}: per class", "", per_class_table(r)]
        md += ["", "Confusion matrix (row-normalised %, mean counts in parentheses)", ""]
        md += [format_confusion(r)]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out + ".json", "w", encoding="utf-8") as f:
        json.dump({"task": task, "reports": reports}, f, indent=2)
    with open(args.out + ".md", "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\nWrote {args.out}.json and {args.out}.md")


if __name__ == "__main__":
    main()
