

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from models.fard import FARD, DialogueTurn

from intent_clarification.data.clariq_dataset import DialogueEpisode


@dataclass
class EnvState:
    initial_request: str
    turns: List[DialogueTurn]
    remaining_candidates: List[str]


class ClarificationEnv:
    def __init__(
        self,
        facet_documents: Optional[Dict[str, str]] = None,
        max_turns: int = 5,
        use_facet_documents: bool = True,
        question_documents: Optional[Dict[Tuple[str, str], str]] = None,
    ):
        self.facet_documents = facet_documents or {}  # old: facet_id -> document
        self.question_documents = question_documents or {}  # (topic_id, question) -> document
        self.max_turns = max_turns
        self.use_facet_documents = use_facet_documents
        self._episode: Optional[DialogueEpisode] = None
        self._state: Optional[EnvState] = None
        self._turn_count = 0

    def reset(self, episode: DialogueEpisode) -> EnvState:
        self._episode = episode
        self._turn_count = 0
        self._state = EnvState(
            initial_request=episode.initial_request,
            turns=[],
            remaining_candidates=list(episode.candidate_questions),
        )
        return self._state

    def candidate_texts_for_fard(self) -> List[str]:
        """Remaining candidates, each followed by its response document. With
        use_facet_documents=False only the question text is used (ablation)."""
        assert self._state is not None
        out = []
        for q in self._state.remaining_candidates:
            doc = None
            if self.use_facet_documents:
                doc = self.question_documents.get((self._episode.topic_id, q))
                if doc is None:  # fall back to the old per-facet documents
                    facet_id = self._episode.facet_of_question.get(q, "")
                    doc = self.facet_documents.get(facet_id)
            out.append(FARD.build_candidate_text(q, doc))
        return out

    def _resolve(self, candidate_question: str) -> Tuple[str, bool, str]:
        """The only place that reads the simulator's target-facet information
        (answers_under_gold, relevant_questions). Used by step() and oracle_peek()."""
        facet_id = self._episode.facet_of_question.get(candidate_question, "")
        confirmed = candidate_question in self._episode.relevant_questions
        answer_text = self._episode.answers_under_gold.get(
            candidate_question, "Yes, exactly." if confirmed else "No, that's not it."
        )
        return facet_id, confirmed, answer_text

    def oracle_peek(self, candidate_question: str) -> Tuple[str, bool, str]:
        """Return what step() would return for this question, without changing any state.

        Diagnostic only: it exposes target-facet information, so it is used just for the
        response_mode='oracle_actual' upper-bound arm and never by the main model."""
        return self._resolve(candidate_question)

    def step(self, action_index: int):
        """Ask the candidate at `action_index`. Returns (reward, done, info)."""
        assert self._state is not None and self._episode is not None
        chosen_question = self._state.remaining_candidates[action_index]
        facet_id, confirmed, answer_text = self._resolve(chosen_question)
        reward = 1.0 if confirmed else -1.0

        self._state.turns.append(DialogueTurn(question=chosen_question, answer=answer_text))
        self._state.remaining_candidates = [
            q for q in self._state.remaining_candidates if q != chosen_question
        ]
        self._turn_count += 1

        done = (
            confirmed or self._turn_count >= self.max_turns or not self._state.remaining_candidates
        )
        info = {"facet_id": facet_id, "confirmed": confirmed, "chosen_question": chosen_question}
        return reward, done, info

    @property
    def state(self) -> EnvState:
        assert self._state is not None
        return self._state
