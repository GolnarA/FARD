# Every prompt sent to the frozen LLM (quoted in the paper's appendix, Table A3).

import hashlib

# P1: pseudo-turn. The LLM turns the last asked question and the user's reply into a more
# specific, self-contained question, plus the answer the user would most likely give. The
# pseudo-turn is internal: it is never asked to the user and is not a question from the bank.
# The output is two plain-text lines rather than JSON, because Llama 2-Chat often breaks JSON
# and every parse failure would silently drop the enrichment. llm_client.parse_rewrite_output
# still accepts the earlier JSON format.
REWRITE_PROMPT_VERSION = "rewrite-v4"
REWRITE_SYSTEM_PROMPT = (
    "Given the conversation so far, rewrite the most recently asked "
    "clarification question into a self-contained, more specific question that "
    "reflects what the user's reply suggests they are looking for (resolve "
    "pronouns and ellipsis using the conversation). Then write the answer the "
    "user would most likely give to the rewritten question, consistent with "
    "what the user has already said. Use only information from the "
    "conversation; do not add needs that the user has not expressed or implied. "
    "Answer in exactly two lines:\n"
    "Rewritten question: <text>\n"
    "User response: <text>"
)
REWRITE_USER_TEMPLATE = (
    "Conversation so far:\n{dialogue_history}\n\n"
    "Most recently asked clarification question, to rewrite:\n{question_to_rewrite}"
)

# P2: response document for one clarification question (offline). {min_words} and
# {max_words} come from rag/question_docs.LENGTH_PRESETS (preset 60 -> "40 to 70 words").
QUESTION_DOC_PROMPT_VERSION = "question-doc-v1"
QUESTION_DOC_SYSTEM_PROMPT = (
    "You write a short, factual reference document of {min_words} to {max_words} "
    "words describing the information need that a clarification question asks "
    "about, within a broad topic. Focus on what distinguishes the information "
    "need behind this question from other aspects of the topic. Use only the "
    "supplied passages and the topic description as grounding; do not invent "
    "facts. Respond with plain text only, no preamble."
)
QUESTION_DOC_USER_TEMPLATE = (
    "Topic: {topic}\nClarification question: {question}\n\nRetrieved passages:\n{passages}"
)

# Old per-facet document prompt (rag/facet_rag.py), kept to reproduce earlier runs.
FACET_DOC_PROMPT_VERSION = "facet-doc-v1"
FACET_DOC_SYSTEM_PROMPT = (
    "You write a short, factual, facet-specific reference document (max "
    "~150 words) that helps disambiguate a specific aspect (facet) of a "
    "broad topic, using only the supplied passages as grounding. Do not "
    "invent facts not supported by the passages; if passages are sparse, "
    "stay generic and clearly hedge. Respond with plain text only, no "
    "preamble."
)
FACET_DOC_USER_TEMPLATE = (
    "Topic: {topic}\nFacet to document: {facet}\n\nRetrieved passages:\n{passages}"
)


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def prompt_registry() -> dict:
    return {
        REWRITE_PROMPT_VERSION: {
            "system": REWRITE_SYSTEM_PROMPT,
            "user_template": REWRITE_USER_TEMPLATE,
            "sha256": sha256(REWRITE_SYSTEM_PROMPT + REWRITE_USER_TEMPLATE),
        },
        QUESTION_DOC_PROMPT_VERSION: {
            "system": QUESTION_DOC_SYSTEM_PROMPT,
            "user_template": QUESTION_DOC_USER_TEMPLATE,
            "sha256": sha256(QUESTION_DOC_SYSTEM_PROMPT + QUESTION_DOC_USER_TEMPLATE),
        },
        FACET_DOC_PROMPT_VERSION: {
            "system": FACET_DOC_SYSTEM_PROMPT,
            "user_template": FACET_DOC_USER_TEMPLATE,
            "sha256": sha256(FACET_DOC_SYSTEM_PROMPT + FACET_DOC_USER_TEMPLATE),
        },
    }
