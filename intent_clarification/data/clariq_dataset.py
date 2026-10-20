"""ClariQ loading, question bank and episode construction.
The simulated user answers q with that recorded answer. `build_episodes` stores this in
`answers_under_gold` and `relevant_questions`; env/clarification_env.py uses it for the
reward (Eq. 5).
"""

from __future__ import annotations

import csv
import json
import logging
import random
import re
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Set

logger = logging.getLogger(__name__)

NO_CLARIFICATION_FACET = "F0000"  # ClariQ's marker for topics that are already clear

# An answer counts as affirmative (rel = 1) if its first word is one of these. Anything else,
# including "no ..." and answers that add information without confirming, counts as rel = 0.
AFFIRMATIVE_FIRST_WORDS = {
    "yes",
    "yeah",
    "yep",
    "yup",
    "yea",
    "sure",
    "correct",
    "exactly",
    "absolutely",
    "definitely",
    "certainly",
}
_FIRST_WORD = re.compile(r"[a-z]+")


def is_affirmative(answer: str) -> bool:
    m = _FIRST_WORD.search((answer or "").lower())
    return bool(m) and m.group(0) in AFFIRMATIVE_FIRST_WORDS


@dataclass
class ClariQRow:
    topic_id: str
    initial_request: str
    topic_desc: str
    clarification_need: int
    facet_id: str
    facet_desc: str
    question_id: str
    question: str
    answer: str
    conversation_context: List[Dict[str, str]] = field(default_factory=list)


def _parse_context(raw: str) -> List[Dict[str, str]]:
    if not raw:
        return []
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return []


def load_tsv(path: str) -> List[ClariQRow]:
    rows: List[ClariQRow] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for r in reader:
            rows.append(
                ClariQRow(
                    topic_id=r["topic_id"],
                    initial_request=r["initial_request"],
                    topic_desc=r.get("topic_desc", ""),
                    clarification_need=int(r.get("clarification_need", 1) or 1),
                    facet_id=r.get("facet_id", NO_CLARIFICATION_FACET),
                    facet_desc=r.get("facet_desc", ""),
                    question_id=r.get("question_id", ""),
                    question=r.get("question", ""),
                    answer=r.get("answer", ""),
                    conversation_context=_parse_context(r.get("conversation_context", "")),
                )
            )
    return rows


@dataclass
class QuestionBank:
    """Topics, facets and questions (T, F, Q in Section 3-2)."""

    topics: Dict[str, str] = field(default_factory=dict)  # topic_id -> topic_desc
    facets_by_topic: Dict[str, Dict[str, str]] = field(
        default_factory=dict
    )  # topic_id -> {facet_id: facet_desc}
    questions_by_facet: Dict[str, List[str]] = field(default_factory=dict)  # facet_id -> [question]
    # A question is asked under many facets, so this keeps only the last facet it was seen
    # with. Only the old per-facet document path uses it; never use it as a relevance label.
    question_to_facet: Dict[str, str] = field(default_factory=dict)
    answers_by_facet: Dict[str, List[str]] = field(default_factory=dict)  # facet_id -> [answer]
    question_ids: Dict[str, str] = field(default_factory=dict)  # question -> question_id

    @classmethod
    def from_rows(cls, rows: List[ClariQRow]) -> "QuestionBank":
        bank = cls()
        for r in rows:
            bank.topics[r.topic_id] = r.topic_desc
            bank.facets_by_topic.setdefault(r.topic_id, {})[r.facet_id] = r.facet_desc
            if r.question:
                bank.questions_by_facet.setdefault(r.facet_id, [])
                if r.question not in bank.questions_by_facet[r.facet_id]:
                    bank.questions_by_facet[r.facet_id].append(r.question)
                bank.question_to_facet[r.question] = r.facet_id
                bank.question_ids.setdefault(r.question, r.question_id)
            if r.answer:
                bank.answers_by_facet.setdefault(r.facet_id, []).append(r.answer)
        return bank

    def candidate_questions_for_topic(self, topic_id: str) -> List[str]:
        """All candidate questions across the facets of a topic, without duplicates."""
        facet_ids = self.facets_by_topic.get(topic_id, {})
        cands: List[str] = []
        for fid in facet_ids:
            cands.extend(self.questions_by_facet.get(fid, []))
        seen = set()
        out = []
        for q in cands:
            if q not in seen:
                seen.add(q)
                out.append(q)
        return out


@dataclass
class DialogueEpisode:
    """One (topic, target facet) episode for the environment."""

    topic_id: str
    initial_request: str
    gold_facet_id: str
    candidate_questions: List[str]
    facet_of_question: Dict[str, str]
    canned_answer_for_facet: Dict[str, str]
    # Read only by env/clarification_env.py::_resolve (the user simulator):
    #   answers_under_gold[q] = answer recorded for (q, target facet)
    #   relevant_questions    = {q : rel(q, target facet) = 1}
    answers_under_gold: Dict[str, str] = field(default_factory=dict)
    relevant_questions: Set[str] = field(default_factory=set)


def build_episodes(rows: List[ClariQRow], bank: QuestionBank) -> List[DialogueEpisode]:
    """One episode per (topic, target facet). A facet with no relevant question can never be
    resolved, so it is skipped (the count is logged)."""
    episodes: List[DialogueEpisode] = []
    by_topic: Dict[str, List[ClariQRow]] = {}
    for r in rows:
        by_topic.setdefault(r.topic_id, []).append(r)

    canned = {
        fid: (answers[0] if answers else "Yes, that is correct.")
        for fid, answers in bank.answers_by_facet.items()
    }
    skipped = 0
    for topic_id, topic_rows in by_topic.items():
        gold_facets = {r.facet_id for r in topic_rows if r.facet_id != NO_CLARIFICATION_FACET}
        candidates = bank.candidate_questions_for_topic(topic_id)
        if not candidates:
            continue
        for gold_facet in sorted(gold_facets):
            answers = {
                r.question: r.answer for r in topic_rows if r.facet_id == gold_facet and r.question
            }
            relevant = {q for q, a in answers.items() if is_affirmative(a)}
            if not relevant:
                skipped += 1
                continue
            episodes.append(
                DialogueEpisode(
                    topic_id=topic_id,
                    initial_request=topic_rows[0].initial_request,
                    gold_facet_id=gold_facet,
                    candidate_questions=candidates,
                    facet_of_question={q: bank.question_to_facet.get(q, "") for q in candidates},
                    canned_answer_for_facet=canned,
                    answers_under_gold=answers,
                    relevant_questions=relevant,
                )
            )
    if skipped:
        logger.info(
            "build_episodes: skipped %d (topic, facet) pairs with no relevant question", skipped
        )
    return episodes


def topic_fold(topic_id: str, num_folds: int = 5) -> int:
    """Fold of a topic: topic_id modulo 5, as in the paper."""
    return int(topic_id) % num_folds if str(topic_id).isdigit() else abs(hash(topic_id)) % num_folds


def write_synthetic_clariq_tsv(path: str, num_topics: int = 6, seed: int = 0) -> None:
    """Write a small ClariQ-shaped file for smoke tests (no download needed).

    Like ClariQ, every question of a topic is answered under every facet; the answer is
    affirmative only when the question is about that facet.
    """
    rng = random.Random(seed)
    topics = [
        ("camera", "best camera", ["price range", "brand", "camera type", "intended use"]),
        (
            "laptop",
            "best laptop",
            ["budget", "screen size", "operating system", "gaming vs office"],
        ),
        (
            "headphones",
            "best headphones",
            ["wired or wireless", "noise cancelling", "budget", "brand"],
        ),
        ("bicycle", "best bicycle", ["terrain", "budget", "frame size", "electric or manual"]),
        ("coffee maker", "best coffee maker", ["capacity", "budget", "espresso or drip", "brand"]),
        ("running shoes", "best running shoes", ["terrain", "arch support", "budget", "brand"]),
    ][:num_topics]

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(
            [
                "topic_id",
                "initial_request",
                "topic_desc",
                "clarification_need",
                "facet_id",
                "facet_desc",
                "question_id",
                "question",
                "answer",
                "conversation_context",
            ]
        )
        for t_idx, (slug, desc, facets) in enumerate(topics):
            topic_id = str(t_idx + 1)
            need = rng.choice([2, 3, 4])
            questions = [
                (f"Q{t_idx:03d}{q_idx}", f"Are you looking for a specific {fd} of {slug}?", q_idx)
                for q_idx, fd in enumerate(facets)
            ]
            for f_idx, facet_desc in enumerate(facets):
                facet_id = f"F{t_idx:02d}{f_idx}"
                for qid, question, q_facet in questions:
                    target = facets[q_facet]
                    answer = (
                        f"Yes, I care about the {facet_desc}."
                        if q_facet == f_idx
                        else f"No, {target} doesn't matter to me."
                    )
                    writer.writerow(
                        [
                            topic_id,
                            f"What's the best {slug}?",
                            desc,
                            need,
                            facet_id,
                            facet_desc,
                            qid,
                            question,
                            answer,
                            json.dumps([]),
                        ]
                    )


# Topic-level view for the ambiguity classifier.
#
# `clarification_need` is constant within a topic (0 of 298 topics in the official
# train/dev/test files have more than one value): ambiguity belongs to the initial query, not
# to a (facet, question) pair. Counting each of a topic's ~54 rows as a sample would repeat
# the same label and mostly the same text, so the ambiguity task uses one instance per topic,
# built before any split, batch or triplet.
@dataclass
class TopicLevelExample:
    topic_id: str
    initial_request: str
    clarification_need: int  # 1..4, constant within a topic


def load_topic_level(path: str) -> List[TopicLevelExample]:
    """One example per topic_id (first row wins). Raises if the label differs between rows of
    the same topic. The ambiguity classifier should read ClariQ only through this function."""
    rows = load_tsv(path)
    seen: "OrderedDict[str, TopicLevelExample]" = OrderedDict()
    multi_label_topics = set()
    for r in rows:
        if r.topic_id in seen:
            if seen[r.topic_id].clarification_need != r.clarification_need:
                multi_label_topics.add(r.topic_id)
            continue
        seen[r.topic_id] = TopicLevelExample(
            topic_id=r.topic_id,
            initial_request=r.initial_request,
            clarification_need=r.clarification_need,
        )
    if multi_label_topics:
        raise ValueError(
            f"{len(multi_label_topics)} topic(s) have inconsistent clarification_need "
            f"across their rows (expected a single topic-level label): {sorted(multi_label_topics)[:5]}..."
        )
    return list(seen.values())


def verify_no_topic_overlap(*splits: List[TopicLevelExample]) -> None:
    """Raise if a topic_id appears in more than one split."""
    seen: Dict[str, int] = {}
    for split_idx, split in enumerate(splits):
        for ex in split:
            if ex.topic_id in seen and seen[ex.topic_id] != split_idx:
                raise ValueError(f"topic_id {ex.topic_id} appears in more than one split")
            seen[ex.topic_id] = split_idx
