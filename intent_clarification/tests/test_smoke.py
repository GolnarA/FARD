

from __future__ import annotations

import os
import random
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from agents.agent import RLAgent
from agents.marl_system import MultiAgentSystem
from agents.windowing import apply_window, validate_window_sizes
from models.critic import CentralizedCritic
from models.fard import FARD, DialogueEnricher, DialogueTurn, format_history
from rag.facet_rag import FacetDocumentBuilder, TfidfPassageRetriever, chunk_text
from tests.fake_encoder import FakeTinyEncoder
from torch.optim import AdamW

from intent_clarification.config import CONFIG
from intent_clarification.data.clariq_dataset import (
    QuestionBank,
    build_episodes,
    is_affirmative,
    load_topic_level,
    load_tsv,
    topic_fold,
    verify_no_topic_overlap,
    write_synthetic_clariq_tsv,
)
from intent_clarification.env.clarification_env import ClarificationEnv
from intent_clarification.llm.llm_client import BaseLLMClient, CachedLLMClient, MockLLMClient, parse_rewrite_output
from intent_clarification.rag.question_docs import QuestionDocumentBuilder, load_question_documents
from intent_clarification.train.train_marl import run_episode


def test_windowing():
    turns = [DialogueTurn(f"Q{i}", f"A{i}") for i in range(5)]
    assert len(apply_window(turns, 1)) == 1
    assert len(apply_window(turns, 3)) == 3
    assert len(apply_window(turns, None)) == 5
    validate_window_sizes([1, 2, 3, 4, None])

    # Figure 4 invariant: every agent sees the most recent turn, regardless
    # of window size (including the defensive window_size=0 edge case).
    for w in [0, 1, 2, 3, 4, None, 100]:
        windowed = apply_window(turns, w)
        assert windowed[-1] == turns[-1], f"last turn missing for window_size={w}"
    assert apply_window([], 1) == [], "empty history must stay empty regardless of window size"
    print("[ok] windowing (incl. Figure 4 last-turn-always-visible invariant)")


def test_dataset_and_rag():
    data_path = "/tmp/clariq_smoke/synthetic.tsv"
    write_synthetic_clariq_tsv(data_path, num_topics=2, seed=0)
    rows = load_tsv(data_path)
    bank = QuestionBank.from_rows(rows)
    episodes = build_episodes(rows, bank)
    assert len(episodes) > 0
    assert all(ep.candidate_questions for ep in episodes)

    chunks = chunk_text("word " * 500, chunk_size_words=140, overlap_words=25)
    assert len(chunks) > 1
    retriever = TfidfPassageRetriever(
        ["camera price is low", "camera brand is known", "unrelated text"]
    )
    top = retriever.retrieve("price", top_k=1)
    assert top and top[0][0] == 0 and top[0][1] > 0

    builder = FacetDocumentBuilder(llm_client=MockLLMClient(), top_k=2)
    facet_docs = builder.build_for_topic(
        topic_id=episodes[0].topic_id,
        topic_desc=bank.topics[episodes[0].topic_id],
        source_document=bank.topics[episodes[0].topic_id] * 5,
        facets_by_id=bank.facets_by_topic[episodes[0].topic_id],
    )
    assert facet_docs
    print(
        f"[ok] dataset + RAG ({len(episodes)} episodes, {len(facet_docs)} facet docs for first topic)"
    )
    return rows, bank, episodes, facet_docs


def test_simulator_relevance(rows, bank, episodes):
    """In ClariQ every question is asked under every facet, so relevance depends on the target
    facet (affirmative recorded answer), not on a question -> facet mapping."""
    assert is_affirmative("Yes, I care about it.") and is_affirmative("yeah sure")
    assert not is_affirmative("No, not that.") and not is_affirmative("I want the price")
    ep = episodes[0]
    env = ClarificationEnv(max_turns=len(ep.candidate_questions))
    env.reset(ep)
    for q in ep.candidate_questions:
        _f, confirmed, answer = env.oracle_peek(q)
        assert confirmed == (q in ep.relevant_questions)
        assert answer == ep.answers_under_gold[q]
    # the same question is relevant for one target facet and not for another
    same_topic = [e for e in episodes if e.topic_id == ep.topic_id]
    assert (
        len(same_topic) > 1 and same_topic[0].relevant_questions != same_topic[1].relevant_questions
    )
    assert topic_fold("17") == 2
    print("[ok] user simulator: relevance and answers depend on the target facet")


def test_grounding_filter_and_manifest():
    from intent_clarification.utils.manifest import build_manifest

    class FakeLLM(BaseLLMClient):
        def generate_facet_document(self, topic_desc, facet_desc, retrieved_passages):
            if "mirrorless" in facet_desc:
                return (
                    "Mirrorless cameras use interchangeable lenses and compact bodies, "
                    "which many photographers prefer for travel and everyday shooting today."
                )
            return "Quantum blockchain toasters revolutionize interstellar agriculture logistics worldwide overnight."

    src = (
        "Mirrorless cameras use interchangeable lenses and compact bodies. Many photographers "
        "prefer mirrorless cameras for travel and everyday shooting because they are light. "
    ) * 3
    b = FacetDocumentBuilder(
        llm_client=FakeLLM(), top_k=2, grounding_threshold=0.6, min_doc_words=8
    )
    recs = b.build_for_topic(
        "T", "best camera", src, {"F1": "mirrorless cameras lenses", "F2": "blockchain toasters"}
    )
    assert recs["F1"]["status"] == "generated", recs["F1"]
    assert recs["F2"]["status"].startswith("fallback"), recs["F2"]
    assert recs["F2"]["raw_generation"] and recs["F2"]["prompt_sha256"]
    m = build_manifest(CONFIG)
    assert "facet-doc-v1" in m["prompts"] and m["rag"]["top_k_chunks"] == 5
    assert "question-doc-v1" in m["prompts"] and "rewrite-v4" in m["prompts"]
    assert m["question_docs"]["top_k"] == 2 and m["question_docs"]["retriever"] == "dense"
    print("[ok] grounding filter + fallback + manifest")


def test_question_documents():
    """One document per question; corpus = facet descriptions; filter and fallback."""
    long_ok = ("Mirrorless cameras are compact cameras with interchangeable lenses. " * 6).strip()

    class FakeLLM(BaseLLMClient):
        def generate_question_document(
            self, topic_desc, question, retrieved_passages, min_words=40, max_words=70
        ):
            return long_ok if "mirrorless" in question else "Too short."

    b = QuestionDocumentBuilder(llm_client=FakeLLM(), retriever_mode="tfidf", top_k=1)
    recs = b.build_for_topic(
        "1",
        "best camera",
        ["mirrorless cameras with interchangeable lenses", "camera price range", "camera brand"],
        [("Q1", "do you want a mirrorless camera"), ("Q2", "what price range do you have in mind")],
    )
    assert [r["question_id"] for r in recs] == ["Q1", "Q2"]
    assert recs[0]["status"] == "generated", recs[0]
    assert recs[0]["retrieval"]["passages"] == ["mirrorless cameras with interchangeable lenses"]
    assert recs[1]["status"] == "fallback:too_short"
    assert recs[1]["document"] == "camera price range"  # extractive fallback = retrieved passage
    # dense without an embedder falls back to TF-IDF instead of crashing
    assert (
        QuestionDocumentBuilder(llm_client=FakeLLM(), retriever_mode="dense").retriever_mode
        == "tfidf"
    )

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "qdocs.jsonl")
        data_path = os.path.join(d, "syn.tsv")
        write_synthetic_clariq_tsv(data_path, num_topics=2, seed=0)
        bank = QuestionBank.from_rows(load_tsv(data_path))
        all_recs = QuestionDocumentBuilder(
            llm_client=MockLLMClient(), retriever_mode="tfidf"
        ).build_and_cache(bank, path)
        docs = load_question_documents(path)
        assert (
            len(docs)
            == len(all_recs)
            == sum(len(bank.candidate_questions_for_topic(t)) for t in bank.topics)
        )
    print(f"[ok] per-question documents ({len(all_recs)} questions, filter + extractive fallback)")
    return docs


def test_llm_parsing_and_cache():
    r, ok = parse_rewrite_output(
        "Sure! Here you go:\nRewritten question: Do you want a compact camera?\n"
        "User response: Yes, something small.",
        "orig",
    )
    assert (
        ok
        and r.rewritten_question == "Do you want a compact camera?"
        and r.hypothetical_answer == "Yes, something small."
    )
    r, ok = parse_rewrite_output('{"rewritten_question": "A?", "hypothetical_answer": "B"}', "orig")
    assert ok and r.rewritten_question == "A?" and r.hypothetical_answer == "B"
    r, ok = parse_rewrite_output("I cannot help with that.", "orig")
    assert not ok and r.rewritten_question == "orig" and r.hypothetical_answer == ""

    class CountingLLM(BaseLLMClient):
        model_id = "counting"

        def _call(self, system, user):
            return "Rewritten question: X?\nUser response: Y."

    inner = CountingLLM()
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "cache.jsonl")
        c = CachedLLMClient(inner, path)
        a = c.rewrite_question_and_hypothesize_answer("hist", "q")
        b = c.rewrite_question_and_hypothesize_answer("hist", "q")
        assert a == b and inner.num_calls == 1 and c.hits == 1
        c2 = CachedLLMClient(inner, path)  # reloaded from disk
        c2.rewrite_question_and_hypothesize_answer("hist", "q")
        assert inner.num_calls == 1 and c2.hits == 1
        c2.generate_question_document("topic", "question", ["p1"])
        c2.generate_question_document("topic", "question", ["p1"])
        assert inner.num_calls == 2
    print("[ok] P1 two-line/JSON parsing + disk cache (identical inputs never call the LLM twice)")


class FakeDenseEmbedder:
    """Offline stand-in for SentenceTransformerEmbedder: hashed bag of words averaged into a
    small L2-normalised vector. Only for testing the retrieval wiring."""

    def __init__(self, dim: int = 32, vocab_size: int = 4096):
        self.dim = dim
        self.vocab_size = vocab_size
        rng = __import__("numpy").random.RandomState(0)
        self._table = rng.normal(size=(vocab_size, dim)).astype("float32")

    def encode(self, texts):
        import numpy as np

        vecs = []
        for t in texts:
            words = t.lower().split() or ["<empty>"]
            ids = [abs(hash(w)) % self.vocab_size for w in words]
            v = self._table[ids].mean(axis=0)
            norm = np.linalg.norm(v) or 1.0
            vecs.append(v / norm)
        return np.stack(vecs, axis=0)


def test_hybrid_retrieval():
    from rag.facet_rag import DenseChunkRetriever, reciprocal_rank_fusion

    passages = [
        "The camera body is compact and weighs under 500 grams.",
        "This model is priced around four hundred dollars at most retailers.",
        "Available colors include black, silver and red.",
    ]
    tfidf = TfidfPassageRetriever(passages)
    dense = DenseChunkRetriever(passages, FakeDenseEmbedder())

    tfidf_hits = tfidf.retrieve("price range", top_k=3)
    dense_hits = dense.retrieve("price range", top_k=3)
    fused = reciprocal_rank_fusion([tfidf_hits, dense_hits], rrf_k=60)
    assert fused, "fused ranking should not be empty"
    assert all(0 <= idx < len(passages) for idx, _ in fused)

    fallback_builder = FacetDocumentBuilder(
        llm_client=MockLLMClient(), retriever_mode="dense", dense_embedder=None
    )
    assert fallback_builder.retriever_mode == "tfidf"

    hybrid_builder = FacetDocumentBuilder(
        llm_client=MockLLMClient(),
        retriever_mode="hybrid",
        dense_embedder=FakeDenseEmbedder(),
        top_k=2,
    )
    recs = hybrid_builder.build_for_topic(
        "T", "best camera", " ".join(passages) * 3, {"F1": "price range"}
    )
    rec = recs["F1"]
    assert rec["retrieval"]["retriever_mode"] == "hybrid"
    assert "tfidf_hits" in rec["retrieval"] and "dense_hits" in rec["retrieval"]

    qb = QuestionDocumentBuilder(
        llm_client=MockLLMClient(),
        retriever_mode="dense",
        dense_embedder=FakeDenseEmbedder(),
        top_k=2,
    )
    qrecs = qb.build_for_topic("1", "best camera", passages, [("Q1", "what price range")])
    assert (
        qrecs[0]["retrieval"]["retriever_mode"] == "dense"
        and len(qrecs[0]["retrieval"]["passages"]) == 2
    )
    print("[ok] hybrid / dense retrieval (RRF fusion + graceful fallback when no dense embedder)")


def build_fake_agents(window_sizes, projection_dim=16, temperature=1.0):
    agents = []
    for i, w in enumerate(window_sizes):
        enc = FakeTinyEncoder(projection_dim=projection_dim)
        fard = FARD(enc, dense_hidden=32)
        agents.append(RLAgent(agent_id=i, window_size=w, fard=fard, temperature=temperature))
    return agents


def test_observed_history_kept():
    """Every ablation arm (Table 9) keeps the observed history (questions and the user's real answers);
    only the generated part differs."""
    turns = [DialogueTurn("Do you mean a DSLR?", "No, something compact.")]
    agent = build_fake_agents([None])[0]
    enricher = DialogueEnricher(MockLLMClient())
    observed = format_history("best camera", turns)
    assert agent.build_observation("none", enricher, "best camera", turns) == observed
    assert agent.build_observation("hypothetical", enricher, "best camera", turns).startswith(
        observed
    )
    print("[ok] observed history (incl. real user responses) kept in every arm")


def test_fard_and_marl_scoring():
    agents = build_fake_agents([1, 2, None])
    enricher = DialogueEnricher(MockLLMClient())
    system = MultiAgentSystem(agents, enricher, temperature=1.0)

    turns = [DialogueTurn("Are you looking for a DSLR?", "No, something compact.")]
    obs = system.per_agent_observations("hypothetical", "What's the best camera?", turns)
    assert len(obs) == 3

    candidates = ["Do you want it lightweight?", "What's your budget?", "Do you need 4K video?"]
    S_t, z_list, logits_list = system.score_all_agents(obs, candidates)
    assert S_t.shape == (3,)
    assert len(z_list) == 3 and all(z.shape == (3,) for z in z_list)
    assert torch.all(S_t <= 1.0) and torch.all(S_t >= -1.0)

    inference_action = MultiAgentSystem.select_action_inference(S_t)
    train_action, logp = system.sample_action_training(S_t)
    assert 0 <= inference_action < 3 and 0 <= train_action < 3
    print("[ok] FARD + multi-agent aggregation (S_t=", S_t.tolist(), ")")


def test_critic():
    enc = FakeTinyEncoder(projection_dim=16)
    critic = CentralizedCritic(enc)
    v = critic("some history text", ["cand a", "cand b"])
    assert v.dim() == 0
    print(f"[ok] centralized critic (V={v.item():.4f})")


def test_full_ctde_step(rows, bank, episodes, facet_docs, question_docs):
    marl_cfg = CONFIG.marl
    marl_cfg.num_agents = 3
    marl_cfg.window_sizes = [1, 2, None]
    marl_cfg.max_turns_per_episode = 3

    for response_mode in ["hypothetical", "none", "oracle_actual"]:
        for use_facet_documents in [True, False]:
            marl_cfg.response_mode = response_mode
            marl_cfg.use_facet_documents = use_facet_documents

            agents = build_fake_agents(
                marl_cfg.window_sizes, projection_dim=16, temperature=marl_cfg.temperature
            )
            enricher = DialogueEnricher(MockLLMClient())
            system = MultiAgentSystem(agents, enricher, temperature=marl_cfg.temperature)
            critic = CentralizedCritic(FakeTinyEncoder(projection_dim=16))
            agent_optim = AdamW([p for a in agents for p in a.parameters()], lr=1e-3)
            critic_optim = AdamW(critic.parameters(), lr=1e-3)
            env = ClarificationEnv(
                facet_documents=facet_docs,
                question_documents=question_docs,
                max_turns=marl_cfg.max_turns_per_episode,
                use_facet_documents=use_facet_documents,
            )
            env.reset(episodes[0])
            texts = env.candidate_texts_for_fard()
            if use_facet_documents:
                assert all(
                    t.startswith(q + "\n") for t, q in zip(texts, episodes[0].candidate_questions)
                )
            else:
                assert texts == episodes[0].candidate_questions

            total_reward, turns_used = run_episode(
                episodes[0], env, system, critic, marl_cfg, agent_optim, critic_optim, train=True
            )
            print(
                f"[ok] CTDE episode ran (response_mode={response_mode}, "
                f"use_facet_documents={use_facet_documents}, reward={total_reward}, turns={turns_used})"
            )

    marl_cfg.response_mode = "hypothetical"
    marl_cfg.use_facet_documents = True


def test_no_leakage_audit():
    """Leakage checks: the frozen LLM calls never see the target facet or relevance labels,
    and the only code that reads the simulator's target-facet information is env._resolve
    (used by step() and by oracle_peek())."""
    import inspect

    sig = inspect.signature(BaseLLMClient.rewrite_question_and_hypothesize_answer)
    assert set(sig.parameters) == {"self", "dialogue_history_text", "question_to_rewrite"}, sig
    sig2 = inspect.signature(BaseLLMClient.generate_facet_document)
    assert set(sig2.parameters) == {"self", "topic_desc", "facet_desc", "retrieved_passages"}, sig2
    sig3 = inspect.signature(BaseLLMClient.generate_question_document)
    assert set(sig3.parameters) == {
        "self",
        "topic_desc",
        "question",
        "retrieved_passages",
        "min_words",
        "max_words",
    }, sig3

    src = open("env/clarification_env.py").read()
    assert src.count(".gold_facet_id") == 0, "env must not read gold_facet_id directly"
    assert src.count(".relevant_questions") == 1, (
        "relevance must be read in exactly one code site (_resolve)"
    )
    assert src.count(".answers_under_gold") == 1, (
        "simulated answers must be read in exactly one code site (_resolve)"
    )
    for path in ["rag/question_docs.py", "llm/llm_client.py", "models/fard.py"]:
        s = open(path).read()
        for name in ["gold_facet_id", "relevant_questions", "answers_under_gold"]:
            assert name not in s, f"{path} must not access {name}"

    # oracle_peek must not mutate env state
    data_path = "/tmp/clariq_smoke/synthetic.tsv"
    write_synthetic_clariq_tsv(data_path, num_topics=1, seed=0)
    rows = load_tsv(data_path)
    bank = QuestionBank.from_rows(rows)
    episodes = build_episodes(rows, bank)
    env = ClarificationEnv(max_turns=3)
    state = env.reset(episodes[0])
    before = (list(state.remaining_candidates), list(state.turns))
    for q in state.remaining_candidates:
        env.oracle_peek(q)
    after = (list(env.state.remaining_candidates), list(env.state.turns))
    assert before == after, "oracle_peek must not mutate environment state"
    print(
        "[ok] leakage audit (LLM call signatures, single simulator read site, oracle_peek is read-only)"
    )


def test_ablation_runner(rows, bank, episodes, facet_docs, question_docs):
    from intent_clarification.train.run_ablation import evaluate_arm

    marl_cfg = CONFIG.marl
    marl_cfg.num_agents = 3
    marl_cfg.window_sizes = [1, 2, None]
    marl_cfg.max_turns_per_episode = 3

    agents = build_fake_agents(
        marl_cfg.window_sizes, projection_dim=16, temperature=marl_cfg.temperature
    )
    for a in agents:
        a.eval()
    enricher = DialogueEnricher(MockLLMClient())
    critic = CentralizedCritic(FakeTinyEncoder(projection_dim=16))
    critic.eval()
    rng = random.Random(0)

    results = []
    for response_mode in ["none", "hypothetical", "oracle_actual"]:
        for use_fd in [True, False]:
            r = evaluate_arm(
                episodes,
                facet_docs,
                agents,
                critic,
                enricher,
                marl_cfg,
                response_mode,
                use_fd,
                num_episodes=3,
                rng=rng,
                question_docs=question_docs,
            )
            assert 0.0 <= r["success_rate"] <= 1.0
            assert r["response_mode"] == response_mode and r["use_facet_documents"] == use_fd
            results.append(r)
    assert len(results) == 6
    r_all = evaluate_arm(
        episodes,
        facet_docs,
        agents,
        critic,
        enricher,
        marl_cfg,
        "hypothetical",
        True,
        num_episodes=None,
        rng=rng,
        question_docs=question_docs,
    )
    assert r_all["num_episodes"] == len(episodes)
    print(
        f"[ok] ablation runner (6/6 arms produced valid results; "
        f"e.g. hypothetical/with-docs success_rate="
        f"{[r['success_rate'] for r in results if r['response_mode'] == 'hypothetical' and r['use_facet_documents']][0]:.2f})"
    )


def test_ranking_evaluator(episodes, facet_docs, question_docs):
    """train.evaluate_ranking.evaluate_episode: MRR and NDCG from the asked list."""
    from intent_clarification.train.evaluate_ranking import evaluate_episode

    marl_cfg = CONFIG.marl
    marl_cfg.num_agents, marl_cfg.window_sizes = 3, [1, 2, None]
    agents = build_fake_agents(marl_cfg.window_sizes, projection_dim=16)
    for a in agents:
        a.eval()
    system = MultiAgentSystem(agents, DialogueEnricher(MockLLMClient()), temperature=1.0)
    env = ClarificationEnv(
        facet_documents=facet_docs, question_documents=question_docs, max_turns=5
    )
    for ep in episodes:
        rec = evaluate_episode(ep, env, system, "hypothetical", max_turns=5)
        assert 1 <= rec["turns"] <= 5 and len(rec["rels"]) == rec["turns"]
        assert sum(rec["rels"]) <= 1 and (rec["rels"][-1] == 1 or rec["mrr"] == 0.0)
        assert 0.0 <= rec["ndcg@3"] <= 1.0 and 0.0 <= rec["ndcg@5"] <= 1.0
        assert rec["topic_id"] == ep.topic_id
    print(f"[ok] ranking evaluator ({len(episodes)} episodes, per-episode MRR/NDCG records)")


def test_tuning_runner():
    """train.tune_question_docs: one tiny configuration end to end, without the test fold."""
    from intent_clarification.train.tune_question_docs import pick_best, run_config, split_episodes

    marl_cfg = CONFIG.marl
    marl_cfg.num_agents = 3
    marl_cfg.window_sizes = [1, 2, None]
    marl_cfg.max_turns_per_episode = 3
    with tempfile.TemporaryDirectory() as d:
        data_path = os.path.join(d, "syn.tsv")
        write_synthetic_clariq_tsv(
            data_path, num_topics=6, seed=0
        )  # topic ids 1..6 -> folds 1,2,3,4,0,1
        rows = [r for r in load_tsv(data_path) if topic_fold(r.topic_id) != 4]
        bank = QuestionBank.from_rows(rows)
        train_eps, val_eps = split_episodes(build_episodes(rows, bank), val_fold=3, test_fold=4)
        assert train_eps and val_eps
        assert all(topic_fold(e.topic_id) not in (3, 4) for e in train_eps)
        assert all(topic_fold(e.topic_id) == 3 for e in val_eps)

        def make_models():
            return (
                build_fake_agents(marl_cfg.window_sizes, projection_dim=16),
                CentralizedCritic(FakeTinyEncoder(projection_dim=16)),
            )

        results = [
            run_config(
                ret,
                k,
                60,
                bank,
                sorted(bank.topics),
                train_eps,
                val_eps,
                MockLLMClient(),
                FakeDenseEmbedder(),
                make_models,
                num_episodes=3,
                seed=0,
                docs_path=os.path.join(d, f"docs_{ret}_{k}.jsonl"),
            )
            for ret, k in [("tfidf", 1), ("dense", 2)]
        ]
        best = pick_best(results)
        assert best in results and all(r["val_episodes"] == len(val_eps) for r in results)
    marl_cfg.response_mode = "hypothetical"
    marl_cfg.use_facet_documents = True
    print("[ok] validation-fold tuning runner (test fold never used)")


def test_topic_level_loader():
    data_path = "/tmp/clariq_smoke/synthetic_topiclevel.tsv"
    write_synthetic_clariq_tsv(data_path, num_topics=6, seed=1)
    rows = load_tsv(data_path)
    topic_level = load_topic_level(data_path)
    # one row per unique topic, not per (topic, facet) row
    assert len(topic_level) == len(set(r.topic_id for r in rows))
    assert len(topic_level) < len(rows), "topic-level dedup should reduce row count"
    # split-overlap check: splitting the same topics into two "splits" must raise
    half = len(topic_level) // 2
    try:
        verify_no_topic_overlap(
            topic_level[:half], topic_level[half - 1 :]
        )  # deliberately overlapping
        raise AssertionError("verify_no_topic_overlap should have raised on overlapping splits")
    except ValueError:
        pass
    verify_no_topic_overlap(topic_level[:half], topic_level[half:])  # non-overlapping: must pass
    print(
        f"[ok] topic-level loader ({len(rows)} rows -> {len(topic_level)} unique-topic instances)"
    )


def main():
    test_windowing()
    test_topic_level_loader()
    rows, bank, episodes, facet_docs = test_dataset_and_rag()
    test_simulator_relevance(rows, bank, episodes)
    test_grounding_filter_and_manifest()
    question_docs = test_question_documents()
    test_llm_parsing_and_cache()
    test_hybrid_retrieval()
    test_no_leakage_audit()
    test_observed_history_kept()
    test_fard_and_marl_scoring()
    test_critic()
    test_full_ctde_step(rows, bank, episodes, facet_docs, question_docs)
    test_ablation_runner(rows, bank, episodes, facet_docs, question_docs)
    test_ranking_evaluator(episodes, facet_docs, question_docs)
    test_tuning_runner()
    print("\nALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
