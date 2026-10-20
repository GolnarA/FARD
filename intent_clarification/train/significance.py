# Significance tests for the comparisons reported in the paper. Methods: evaluation/stats.py.
#
#
#     python -m train.significance --runs runs/*.jsonl --reference FARD \
#         --against QL BERT-NeuQS MI-Clf MMR-BERT --metrics mrr ndcg@3 ndcg@5 \
#         --out results/table6_sig
#
#     # ablations as a family of their own (Tables 7, 8, 10)
#     python -m train.significance --runs runs/*.jsonl --reference FARD \
#         --against FARD-BERT FARD-noPseudoTurn FARD-noDocs SingleAgent --out results/ablation_sig


from __future__ import annotations

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from intent_clarification.evaluation.stats import (  # noqa: E402
    compare_classification,
    holm,
    load_records,
    run_family,
    seed_summary,
    systems_in,
)


def fmt_p(p: float) -> str:
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


def ranking_markdown(results, records, systems, metrics, unit="facet") -> str:
    from intent_clarification.evaluation.stats import seed_summary_facet

    summ = seed_summary_facet if unit == "facet" else seed_summary
    lines = [
        "### Seed variability (mean ± sd over training seeds)",
        "",
        "| System | " + " | ".join(m.upper() for m in metrics) + " | Seeds | Topics |",
        "|---|" + "---|" * (len(metrics) + 2),
    ]
    for s in systems:
        cells, n_seeds, n_topics = [], 0, 0
        for m in metrics:
            ss = summ(records, s, m)
            n_seeds, n_topics = ss["n_seeds"], ss["n_topics"]
            cells.append(
                f"{ss['mean']:.3f} ± {ss['sd']:.3f}" if ss["n_seeds"] > 1 else f"{ss['mean']:.3f}"
            )
        lines.append(f"| {s} | " + " | ".join(cells) + f" | {n_seeds} | {n_topics} |")
    lines += [
        "",
        "### Paired comparisons (per facet, clustered by topic)",
        "",
        "| A vs B | Metric | A | B | Δ (A−B) | Δ% | 95% CI of Δ | p (rand.) | p (Holm) | d_z [95% CI] | W/T/L | n |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r['system_a']} vs {r['system_b']} | {r['metric'].upper()} | {r['mean_a']:.3f} | {r['mean_b']:.3f} | "
            f"{r['diff']:+.4f} | {r['rel_diff_pct']:+.1f}% | [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}] | "
            f"{fmt_p(r['p_value'])} | {fmt_p(r['p_holm'])} | {r['cohens_dz']:.2f} "
            f"[{r['dz_ci_low']:.2f}, {r['dz_ci_high']:.2f}] | {r['wins']}/{r['ties']}/{r['losses']} | "
            f"{r.get('n_facets', r['n_topics'])} facets / {r['n_topics']} topics |"
        )
    return "\n".join(lines) + "\n"


def classification_markdown(results) -> str:
    lines = [
        "| A vs B | Macro-F1 A | Macro-F1 B | Δ | 95% CI of Δ | p (rand.) | p (Holm) | McNemar b/c | p (McNemar) | n topics | seeds |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r['system_a']} vs {r['system_b']} | {r['macro_f1_a']:.3f} | {r['macro_f1_b']:.3f} | {r['diff']:+.3f} | "
            f"[{r['ci_low']:+.3f}, {r['ci_high']:+.3f}] | {fmt_p(r['p_value'])} | {fmt_p(r['p_holm'])} | "
            f"{r['mcnemar_b']}/{r['mcnemar_c']} | {fmt_p(r['p_mcnemar'])} | {r['n_topics']} | {r['n_seeds']} |"
        )
    for r in results:
        for side in ("a", "b"):
            name = r[f"system_{side}"]
            lines += [
                "",
                f"Per-class results — {name}",
                "",
                "| Class | Precision | Recall | F1 | Support |",
                "|---|---|---|---|---|",
            ]
            for c, v in r[f"per_class_{side}"].items():
                lines.append(
                    f"| {c} | {v['precision']:.3f} | {v['recall']:.3f} | {v['f1']:.3f} | {v['support']} |"
                )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True, help="runfiles (globs allowed)")
    parser.add_argument("--task", choices=["ranking", "classification"], default="ranking")
    parser.add_argument(
        "--reference", required=True, help="system A in every comparison (the proposed model)"
    )
    parser.add_argument(
        "--against", nargs="+", required=True, help="systems B; together they form one Holm family"
    )
    parser.add_argument("--metrics", nargs="+", default=["mrr", "ndcg@3", "ndcg@5"])
    parser.add_argument(
        "--unit",
        choices=["facet", "topic"],
        default="facet",
        help="facet: metric per facet episode, tests clustered by topic (default)",
    )
    parser.add_argument("--n-perm", type=int, default=100_000)
    parser.add_argument("--n-boot", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0, help="seed of the tests themselves")
    parser.add_argument("--out", required=True, help="output prefix (.json and .md are written)")
    args = parser.parse_args()

    paths = sorted({p for pat in args.runs for p in (glob.glob(pat) or [pat])})
    records = load_records(paths)
    available = systems_in(records)
    missing = [s for s in [args.reference, *args.against] if s not in available]
    if missing:
        raise SystemExit(f"systems not found in runfiles: {missing}; available: {available}")
    pairs = [(args.reference, b) for b in args.against]

    if args.task == "ranking":
        results = run_family(
            records, pairs, args.metrics, args.n_perm, args.n_boot, args.seed, args.unit
        )
        md = ranking_markdown(
            results, records, [args.reference, *args.against], args.metrics, args.unit
        )
    else:
        n_perm = min(args.n_perm, 10_000)
        results = [
            compare_classification(records, a, b, n_perm, args.n_boot, args.seed) for a, b in pairs
        ]
        for r, p in zip(results, holm([r["p_value"] for r in results])):
            r["p_holm"] = p
        md = classification_markdown(results)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out + ".json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "runfiles": paths,
                "task": args.task,
                "n_perm": args.n_perm,
                "n_boot": args.n_boot,
                "results": results,
            },
            f,
            indent=2,
        )
    with open(args.out + ".md", "w", encoding="utf-8") as f:
        f.write(md)
    print(md)
    print(f"Wrote {args.out}.json and {args.out}.md")


if __name__ == "__main__":
    main()
