# Check which ClariQ topics actually yield intent-clarification episodes.

import argparse
import csv
import re
import sys
from collections import Counter, defaultdict

NO_CLARIFICATION_FACET = "F0000"
AFFIRMATIVE = {"yes", "yeah", "yep", "yup", "yea", "sure", "correct",
               "exactly", "absolutely", "definitely", "certainly"}
FIRST_WORD = re.compile(r"[a-z]+")


def is_affirmative(answer):
    m = FIRST_WORD.search((answer or "").lower())
    return bool(m) and m.group(0) in AFFIRMATIVE


def main(paths, expect=None, expect_episodes=None):
    rows = []
    for p in paths:
        with open(p, encoding="utf-8", newline="") as f:
            rows += list(csv.DictReader(f, delimiter="\t"))

    need = {}
    by_topic = defaultdict(list)
    for r in rows:
        need[r["topic_id"]] = int(r.get("clarification_need") or 1)
        by_topic[r["topic_id"]].append(r)

    ep_topics, facets_per_topic = set(), {}
    for tid, trs in by_topic.items():
        facets = {r["facet_id"] for r in trs if r["facet_id"] != NO_CLARIFICATION_FACET}
        facets_per_topic[tid] = len(facets)
        if not any(r["question"] for r in trs):
            continue
        for f in facets:
            answers = {r["question"]: r["answer"] for r in trs if r["facet_id"] == f and r["question"]}
            if any(is_affirmative(a) for a in answers.values()):
                ep_topics.add(tid)
                break

    n_all = Counter(need.values())
    n_ep = Counter(need[t] for t in ep_topics)
    print(f"topics in the files: {len(need)}")
    print(f"topics with >= 1 evaluation episode: {len(ep_topics)}")
    print("level | topics | with episode | missing")
    for lvl in (1, 2, 3, 4):
        print(f"  {lvl}   | {n_all[lvl]:6d} | {n_ep[lvl]:12d} | {n_all[lvl] - n_ep[lvl]:7d}")
    lvl1 = [t for t in need if need[t] == 1]
    multi = sum(1 for t in lvl1 if facets_per_topic[t] >= 2)
    only_f0 = sum(1 for t in lvl1 if facets_per_topic[t] == 0)
    print(f"level-1 topics with >= 2 real facets: {multi} of {len(lvl1)}")
    print(f"level-1 topics with only the F0000 sentinel (no episode possible): {only_f0}")
    no_ep = sorted(set(need) - ep_topics, key=lambda t: (len(t), t))
    if no_ep:
        print(f"topics without an episode ({len(no_ep)}): {', '.join(no_ep)}")
        print("  -> they cannot enter a facet-episode metric; state in the paper whether the reported means")
        print("     are over the topics with episodes or over all topics (e.g. with score 0 for these).")

    status = 0
    if expect is not None and len(need) != expect:
        print(f"MISMATCH: the paper reports {expect} topics, these files contain {len(need)}")
        status = 2
    if expect_episodes is not None and len(ep_topics) != expect_episodes:
        print(f"MISMATCH: expected {expect_episodes} topics with an episode, these files give {len(ep_topics)}")
        status = 2
    return status


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="ClariQ TSV files (train, dev and the labelled test file)")
    ap.add_argument("--expect-topics", type=int, default=None,
                    help="number of topics that must be in the files (the paper: 298)")
    ap.add_argument("--expect-episode-topics", type=int, default=None,
                    help="number of topics that must yield at least one evaluation episode")
    a = ap.parse_args()
    sys.exit(main(a.paths, a.expect_topics, a.expect_episode_topics))
