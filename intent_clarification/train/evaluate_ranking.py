# Evaluate a trained checkpoint on its test fold: MRR, NDCG@3 and NDCG@5 (Section 4-3,
# Tables 6-10), :
#     python -m train.evaluate_ranking --data train.tsv dev.tsv --folds 4 \
#         --checkpoint ckpt/full/f4_s42/marl_system.pt --system FARD --seed-label 42 \
#         --out runs/FARD_f4_s42.jsonl


from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from agents.marl_system import MultiAgentSystem, build_agents
from models.fard import DialogueEnricher
from rag.facet_rag import load_facet_documents
from utils.seeding import set_global_seed

from intent_clarification.config import CONFIG
from intent_clarification.data.clariq_dataset import QuestionBank, build_episodes, load_tsv, topic_fold
from intent_clarification.env.clarification_env import ClarificationEnv
from intent_clarification.evaluation.metrics import episode_metrics
from intent_clarification.llm.llm_client import build_llm_client
from intent_clarification.rag.question_docs import load_question_documents
from intent_clarification.train.build_question_docs import add_llm_args, apply_llm_args
from intent_clarification.train.model_args import add_model_args, apply_model_args, restore_architecture
from intent_clarification.utils.manifest import build_manifest, write_manifest


@torch.no_grad()
def evaluate_episode(
    episode,
    env: ClarificationEnv,
    system: MultiAgentSystem,
    response_mode: str,
    max_turns: int,
    ideal: str = "all",
) -> dict:
    t0 = time.perf_counter()
    state = env.reset(episode)
    asked, rels = [], []
    done = False
    while not done and len(asked) < max_turns:
        candidate_texts = env.candidate_texts_for_fard()
        if not candidate_texts:
            break
        observations = system.per_agent_observations(
            response_mode,
            episode.initial_request,
            state.turns,
            candidate_questions=state.remaining_candidates,
            env=env,
        )
        S_t, _z, _logits = system.score_all_agents(observations, candidate_texts)
        action = MultiAgentSystem.select_action_inference(S_t)
        question = state.remaining_candidates[action]
        _reward, done, info = env.step(action)
        asked.append(question)
        rels.append(int(info["confirmed"]))
    n_relevant = len(set(episode.relevant_questions) & set(episode.candidate_questions))
    return {
        "topic_id": episode.topic_id,
        "facet_id": episode.gold_facet_id,
        "asked": asked,
        "rels": rels,
        "n_relevant": n_relevant,
        "turns": len(asked),
        **episode_metrics(rels, n_relevant, ideal),
        "seconds": time.perf_counter() - t0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, nargs="+", required=True)
    parser.add_argument(
        "--folds", type=int, nargs="+", required=True, help="test fold(s) (topic_id %% 5)"
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--system", type=str, required=True, help="name written to the runfile, e.g. FARD"
    )
    parser.add_argument(
        "--seed-label",
        type=str,
        default=None,
        help="training seed of the checkpoint, written to the runfile",
    )
    parser.add_argument(
        "--response-mode",
        type=str,
        default=None,
        choices=["hypothetical", "none", "oracle_actual"],
        help="default: the mode stored in the checkpoint",
    )
    parser.add_argument("--no-facet-documents", action="store_true")
    parser.add_argument("--question-docs", type=str, default=CONFIG.question_docs_path)
    parser.add_argument("--ndcg-ideal", type=str, default="all", choices=["all", "single"])
    parser.add_argument("--local-dir", type=str, default=None)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--out", type=str, required=True)
    add_llm_args(parser)
    add_model_args(parser)
    args = parser.parse_args()
    apply_llm_args(args)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    restore_architecture(ckpt.get("architecture"))
    apply_model_args(args)  # flags given on the command line override the stored architecture
    if args.local_dir:
        CONFIG.encoder.local_dir = args.local_dir
    arch = ckpt.get("architecture") or {}
    response_mode = args.response_mode or arch.get("response_mode", CONFIG.marl.response_mode)
    use_docs = (not args.no_facet_documents) and arch.get("use_facet_documents", True)
    set_global_seed(CONFIG.seed.seed, CONFIG.seed.deterministic_cudnn)
    device = torch.device(CONFIG.device)

    rows = [r for p in args.data for r in load_tsv(p)]
    bank = QuestionBank.from_rows(rows)
    episodes = [
        ep for ep in build_episodes(rows, bank) if topic_fold(ep.topic_id) in set(args.folds)
    ]
    question_docs = load_question_documents(args.question_docs)
    facet_docs = {} if question_docs else load_facet_documents(CONFIG.facet_docs_path)

    agents = build_agents(
        CONFIG.marl, CONFIG.encoder, share_bert_backbone=True, pretrained=not args.no_pretrained
    )
    for a, sd in zip(agents, ckpt["agents"]):
        a.load_state_dict(sd)
    for a in agents:
        a.to(device).eval()
    llm_client = build_llm_client(CONFIG.llm)
    system = MultiAgentSystem(
        agents, DialogueEnricher(llm_client), temperature=CONFIG.marl.temperature
    )
    env = ClarificationEnv(
        facet_documents=facet_docs,
        question_documents=question_docs,
        max_turns=CONFIG.marl.max_turns_per_episode,
        use_facet_documents=use_docs,
    )

    seed_label = args.seed_label or str(arch.get("seed", CONFIG.seed.seed))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    totals = {"mrr": 0.0, "ndcg@3": 0.0, "ndcg@5": 0.0}
    with open(args.out, "w", encoding="utf-8") as f:
        for ep in episodes:
            rec = evaluate_episode(
                ep, env, system, response_mode, CONFIG.marl.max_turns_per_episode, args.ndcg_ideal
            )
            rec.update(
                {"system": args.system, "seed": seed_label, "test_fold": topic_fold(ep.topic_id)}
            )
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            for k in totals:
                totals[k] += rec[k]
    n = len(episodes)
    n_topics = len({ep.topic_id for ep in episodes})
    print(
        f"{args.system} seed={seed_label} folds={args.folds}: {n} episodes, {n_topics} topics | "
        + "  ".join(f"{k}={v / max(n, 1):.4f}" for k, v in totals.items())
        + "  (episode means; "
        "the paper reports topic means, see train/significance.py)"
    )
    write_manifest(
        args.out + ".manifest.json",
        build_manifest(
            CONFIG,
            input_files={f"clariq_tsv_{i}": p for i, p in enumerate(args.data)},
            extra={
                "checkpoint": args.checkpoint,
                "test_folds": args.folds,
                "response_mode": response_mode,
                "use_facet_documents": use_docs,
                "ndcg_ideal": args.ndcg_ideal,
                "llm_backend_calls": getattr(llm_client, "num_calls", None),
            },
        ),
    )


if __name__ == "__main__":
    main()
