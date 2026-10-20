# Offline construction of one response document per clarification question


from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from rag.facet_rag import (
    BaseDenseEmbedder,
    DenseChunkRetriever,
    TfidfPassageRetriever,
    grounding_score,
)

from intent_clarification.llm.llm_client import BaseLLMClient
from intent_clarification.llm.prompts import QUESTION_DOC_PROMPT_VERSION, prompt_registry

logger = logging.getLogger(__name__)

# length_preset -> words requested in the prompt / words accepted by the filter
LENGTH_PRESETS: Dict[int, Dict[str, int]] = {
    60: {"min_words": 40, "max_words": 70, "accept_min_words": 40, "accept_max_words": 80},
    120: {"min_words": 90, "max_words": 130, "accept_min_words": 80, "accept_max_words": 150},
}


@dataclass
class QuestionDocumentBuilder:
    llm_client: BaseLLMClient
    retriever_mode: str = "dense"  # "dense" | "tfidf"
    dense_embedder: Optional[BaseDenseEmbedder] = None
    top_k: int = 2
    length_preset: int = 60
    grounding_threshold: float = 0.5
    stop_words: str = "english"

    def __post_init__(self):
        if self.retriever_mode not in ("dense", "tfidf"):
            raise ValueError(
                f"retriever_mode must be 'dense' or 'tfidf', got {self.retriever_mode!r}"
            )
        if self.retriever_mode == "dense" and self.dense_embedder is None:
            logger.warning("retriever_mode='dense' but no dense_embedder was given; using TF-IDF.")
            self.retriever_mode = "tfidf"
        if self.length_preset not in LENGTH_PRESETS:
            raise ValueError(f"length_preset must be one of {sorted(LENGTH_PRESETS)}")
        self.lengths = LENGTH_PRESETS[self.length_preset]

    def _retriever(self, passages: List[str]):
        if self.retriever_mode == "dense":
            return DenseChunkRetriever(passages, self.dense_embedder)
        return TfidfPassageRetriever(passages, stop_words=self.stop_words)

    def _filter(self, doc: str, evidence: List[str]) -> Tuple[bool, float, str]:
        n = len(doc.split())
        g = grounding_score(doc, evidence)
        if n < self.lengths["accept_min_words"]:
            return False, g, "too_short"
        if n > self.lengths["accept_max_words"]:
            return False, g, "too_long"
        if g < self.grounding_threshold:
            return False, g, "low_grounding"
        return True, g, "ok"

    def build_for_topic(
        self,
        topic_id: str,
        topic_desc: str,
        facet_descriptions: List[str],
        questions: List[Tuple[str, str]],
    ) -> List[dict]:
        """`questions` is a list of (question_id, question_text) of one topic.
        Returns one record per question."""
        passages: List[str] = []
        for d in facet_descriptions:
            d = (d or "").strip()
            if d and d not in passages:
                passages.append(d)
        retriever = self._retriever(passages) if passages else None
        prompt_sha = prompt_registry()[QUESTION_DOC_PROMPT_VERSION]["sha256"]

        records = []
        for question_id, question in questions:
            hits = retriever.retrieve(question, top_k=self.top_k) if retriever else []
            retrieved = [passages[i] for i, _ in hits]
            generated = self.llm_client.generate_question_document(
                topic_desc=topic_desc,
                question=question,
                retrieved_passages=retrieved,
                min_words=self.lengths["min_words"],
                max_words=self.lengths["max_words"],
            )
            ok, g, reason = self._filter(generated, retrieved + [topic_desc, question])
            if ok:
                document, status = generated.strip(), "generated"
            else:
                document = " ".join(retrieved) or topic_desc
                status = f"fallback:{reason}"
            records.append(
                {
                    "question_id": question_id,
                    "question": question,
                    "topic_id": topic_id,
                    "document": document,
                    "status": status,
                    "grounding_score": round(g, 4),
                    "retrieval": {
                        "retriever_mode": self.retriever_mode,
                        "top_k": self.top_k,
                        "passages": retrieved,
                        "scores": [round(s, 4) for _, s in hits],
                    },
                    "length_preset": self.length_preset,
                    "raw_generation": generated,
                    "prompt_version": QUESTION_DOC_PROMPT_VERSION,
                    "prompt_sha256": prompt_sha,
                }
            )
        return records

    def build_and_cache(
        self, bank, output_path: str, topic_ids: Optional[List[str]] = None
    ) -> List[dict]:
        """Build documents for every question of the given topics (default: all topics in
        `bank`, a data.clariq_dataset.QuestionBank) and write them to `output_path`."""
        all_records: List[dict] = []
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for topic_id in topic_ids or list(bank.topics):
                questions = [
                    (bank.question_ids.get(q, ""), q)
                    for q in bank.candidate_questions_for_topic(topic_id)
                ]
                facet_descs = list(bank.facets_by_topic.get(topic_id, {}).values())
                for rec in self.build_for_topic(
                    topic_id, bank.topics[topic_id], facet_descs, questions
                ):
                    all_records.append(rec)
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return all_records


def load_question_documents(path: str) -> Dict[Tuple[str, str], str]:
    """Return {(topic_id, question text): document}, the lookup used for the candidate texts.
    The key includes the topic because a few generic questions (e.g. "is this an emergency")
    occur in several topics and get a different document in each."""
    docs: Dict[Tuple[str, str], str] = {}
    p = Path(path)
    if not p.exists():
        return docs
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                docs[(rec["topic_id"], rec["question"])] = rec["document"]
    return docs
