# Choose the document-construction settings (retriever, k, length) on the validation fold
# (sction 3-4-b).


from __future__ import annotations

import argparse
import itertools
import json
import os
import random
import sys
from typing import Callable, Dict, List

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [_ROOT, os.path.dirname(_ROOT)]  # modules are imported both flat (agents, models, ...) and as intent_clarification.*

import torch
from agents.marl_system import MultiAgentSystem, build_agents
from models.critic import CentralizedCritic
from models.encoder import TextEncoder
from models.fard import DialogueEnricher
from rag.facet_rag import build_dense_embedder
from torch.optim import AdamW
from utils.seeding import set_global_seed

from intent_clarification.config import CONFIG
from intent_clarification.data.clariq_dataset import (
    QuestionBank,
    build_episodes,
    load_tsv,
    topic_fold,
    write_synthetic_clariq_tsv,
)
from intent_clarification.env.clarification_env import ClarificationEnv
from intent_clarification.llm.llm_client import build_llm_client
from intent_clarification.rag.question_docs import LENGTH_PRESETS
from intent_clarification.train.build_question_docs import add_llm_args, apply_llm_args, make_builder, write_stats
from intent_clarification.train.run_ablation import evaluate_arm
from intent_clarification.train.train_marl import llm_usage, run_episode
from intent_clarification.utils.manifest import build_manifest, write_manifest


def split_episodes(episodes, val_fold: int, test_fold: int):
    train = [ep for ep in episodes if topic_fold(ep.topic_id) not in (val_fold, test_fold)]
    val = [ep for ep in episodes if topic_fold(ep.topic_id) == val_fold]
    return train, val


def run_config(
    retriever: str,
    top_k: int,
    length_preset: int,
    bank,
    topic_ids: List[str],
    train_eps,
    val_eps,
    llm_client,
    dense_embedder,
    make_models: Callable,
    num_episodes: int,
    seed: int,
    docs_path: str,
) -> Dict:
    builder = make_builder(
        llm_client,
        retriever,
        top_k,
        length_preset,
        dense_embedder if retriever == "dense" else None,
    )
    records = builder.build_and_cache(bank, docs_path, topic_ids=topic_ids)
    stats = write_stats(records, docs_path, builder)
    question_docs = {(r["topic_id"], r["question"]): r["document"] for r in records}

    set_global_seed(seed, CONFIG.seed.deterministic_cudnn)
    agents, critic = make_models()
    enricher = DialogueEnricher(llm_client)
    marl_cfg = CONFIG.marl
    system = MultiAgentSystem(agents, enricher, temperature=marl_cfg.temperature)
    env = ClarificationEnv(
        question_documents=question_docs,
        max_turns=marl_cfg.max_turns_per_episode,
        use_facet_documents=True,
    )
    agent_optim = AdamW([p for a in agents for p in a.parameters()], lr=marl_cfg.policy_lr)
    critic_optim = AdamW(critic.parameters(), lr=marl_cfg.critic_lr)
    marl_cfg.response_mode = "hypothetical"
    marl_cfg.use_facet_documents = True

    rng = random.Random(seed)
    for a in agents:
        a.train()
    critic.train()
    for _ in range(num_episodes):
        run_episode(
            rng.choice(train_eps),
            env,
            system,
            critic,
            marl_cfg,
            agent_optim,
            critic_optim,
            train=True,
        )

    for a in agents:
        a.eval()
    critic.eval()
    val = evaluate_arm(
        val_eps,
        {},
        agents,
        critic,
        enricher,
        marl_cfg,
        "hypothetical",
        True,
        None,
        rng,
        question_docs=question_docs,
    )
    return {
        "retriever": builder.retriever_mode,
        "top_k": top_k,
        "length_preset": length_preset,
        "val_mean_reward": val["mean_reward"],
        "val_success_rate": val["success_rate"],
        "val_episodes": val["num_episodes"],
        "fallback_rate": stats["fallback_rate"],
        "documents": docs_path,
    }


def pick_best(results: List[Dict]) -> Dict:
    return sorted(results, key=lambda r: (-r["val_mean_reward"], r["top_k"], r["length_preset"]))[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, nargs="+", default=None)
    parser.add_argument("--val-fold", type=int, default=3)
    parser.add_argument("--test-fold", type=int, default=4)
    parser.add_argument(
        "--episodes", type=int, default=2000, help="training episodes per configuration"
    )
    parser.add_argument(
        "--grid-retrievers",
        type=str,
        nargs="+",
        default=["tfidf", "dense"],
        choices=["tfidf", "dense"],
    )
    parser.add_argument("--grid-k", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument(
        "--grid-lengths", type=int, nargs="+", default=[60, 120], choices=sorted(LENGTH_PRESETS)
    )
    parser.add_argument("--dense-model", type=str, default=CONFIG.question_docs.dense_model_name)
    parser.add_argument("--dense-local-dir", type=str, default=CONFIG.question_docs.dense_local_dir)
    parser.add_argument(
        "--local-dir", type=str, default=None, help="local folder of the FARD encoder"
    )
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--out-dir", type=str, default="./data_files/question_doc_tuning")
    add_llm_args(parser)
    args = parser.parse_args()
    apply_llm_args(args)
    assert args.val_fold != args.test_fold, "validation and test folds must differ"
    if args.local_dir:
        CONFIG.encoder.local_dir = args.local_dir

    data_paths = args.data
    if not data_paths:
        data_paths = [os.path.join(CONFIG.data_dir, "synthetic_train.tsv")]
        write_synthetic_clariq_tsv(data_paths[0], seed=CONFIG.seed.seed)
        print(f"No --data given; wrote a synthetic ClariQ-shaped sample to {data_paths[0]}")

    # drop the test fold before building anything
    rows = [r for p in data_paths for r in load_tsv(p) if topic_fold(r.topic_id) != args.test_fold]
    bank = QuestionBank.from_rows(rows)
    train_eps, val_eps = split_episodes(build_episodes(rows, bank), args.val_fold, args.test_fold)
    if not train_eps or not val_eps:
        raise RuntimeError(
            "Empty training or validation split; check --val-fold / --test-fold and the data."
        )
    topic_ids = sorted(bank.topics)
    print(
        f"Training episodes: {len(train_eps)} | validation episodes: {len(val_eps)} | "
        f"topics used: {len(topic_ids)} (test fold {args.test_fold} excluded)"
    )

    llm_client = build_llm_client(CONFIG.llm)
    dense_embedder = (
        build_dense_embedder(args.dense_model, local_dir=args.dense_local_dir)
        if "dense" in args.grid_retrievers
        else None
    )

    def make_models():
        agents = build_agents(
            CONFIG.marl, CONFIG.encoder, share_bert_backbone=True, pretrained=not args.no_pretrained
        )
        critic = CentralizedCritic(
            TextEncoder(
                model_name=CONFIG.encoder.model_name,
                hidden_size=CONFIG.encoder.hidden_size,
                projection_dim=CONFIG.encoder.projection_dim,
                dropout=CONFIG.encoder.dropout,
                max_length=CONFIG.encoder.max_length,
                pretrained=not args.no_pretrained,
                local_dir=CONFIG.encoder.local_dir,
            )
        )
        device = torch.device(CONFIG.device)
        for a in agents:
            a.to(device)
        return agents, critic.to(device)

    os.makedirs(args.out_dir, exist_ok=True)
    results = []
    for retriever, k, length in itertools.product(
        args.grid_retrievers, args.grid_k, args.grid_lengths
    ):
        docs_path = os.path.join(args.out_dir, f"docs_{retriever}_k{k}_L{length}.jsonl")
        r = run_config(
            retriever,
            k,
            length,
            bank,
            topic_ids,
            train_eps,
            val_eps,
            llm_client,
            dense_embedder,
            make_models,
            args.episodes,
            CONFIG.seed.seed,
            docs_path,
        )
        results.append(r)
        print(
            f"{retriever:6s} k={k} L={length:3d} -> val mean return {r['val_mean_reward']:+.3f} "
            f"success {r['val_success_rate']:.3f} | fallback {r['fallback_rate']}"
        )

    best = pick_best(results)
    summary = {
        "val_fold": args.val_fold,
        "test_fold": args.test_fold,
        "episodes_per_config": args.episodes,
        "results": results,
        "best": best,
        "llm_usage": llm_usage(llm_client),
    }
    out = os.path.join(args.out_dir, "tuning_results.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    write_manifest(
        out + ".manifest.json",
        build_manifest(
            CONFIG, input_files={f"clariq_tsv_{i}": p for i, p in enumerate(data_paths)}
        ),
    )
    print(
        f"\nBest on validation: retriever={best['retriever']} top_k={best['top_k']} "
        f"length_preset={best['length_preset']}"
    )
    print(
        "Build the final documents for ALL topics with:\n"
        f"    python -m train.build_question_docs --data {' '.join(data_paths)} "
        f"--retriever {best['retriever']} --top-k {best['top_k']} --length-preset {best['length_preset']}"
    )
    print(f"Results: {out}")


if __name__ == "__main__":
    main()
