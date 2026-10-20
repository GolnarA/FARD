
# Turn a baseline's output into a runfile for train/significance.py,

from __future__ import annotations

import argparse
import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [_ROOT, os.path.dirname(_ROOT)]  # modules are imported both flat (agents, models, ...) and as intent_clarification.*

from data.clariq_dataset import QuestionBank, build_episodes, load_tsv, topic_fold  # noqa: E402
from evaluation.metrics import episode_metrics  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", nargs="+", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--system", required=True)
    parser.add_argument("--seed-label", default="0", help="Deterministic baselines: leave 0")
    parser.add_argument("--max-turns", type=int, default=5)
    parser.add_argument("--ndcg-ideal", default="all", choices=["all", "single"])
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    rows = [r for p in args.data for r in load_tsv(p)]
    episodes = {(ep.topic_id, ep.gold_facet_id): ep for ep in build_episodes(rows, QuestionBank.from_rows(rows))}
    n_in = n_out = 0
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.input, encoding="utf-8") as fin, open(args.out, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            n_in += 1
            rec = json.loads(line)
            ep = episodes.get((str(rec["topic_id"]), str(rec["facet_id"])))
            if ep is None:  # (topic, facet) with no relevant question: not an evaluation episode
                continue
            asked, rels = [], []
            for q in rec.get("asked") or rec.get("ranking") or []:
                if len(asked) >= args.max_turns:
                    break
                asked.append(q)
                rels.append(int(q in ep.relevant_questions))
                if rels[-1]:
                    break
            n_rel = len(set(ep.relevant_questions) & set(ep.candidate_questions))
            fout.write(json.dumps({"system": args.system, "seed": args.seed_label, "topic_id": ep.topic_id,
                                   "facet_id": ep.gold_facet_id, "test_fold": topic_fold(ep.topic_id),
                                   "asked": asked, "rels": rels, "n_relevant": n_rel, "turns": len(asked),
                                   **episode_metrics(rels, n_rel, args.ndcg_ideal)}, ensure_ascii=False) + "\n")
            n_out += 1
    print(f"{args.system}: scored {n_out} of {n_in} input episodes -> {args.out}")


if __name__ == "__main__":
    main()
