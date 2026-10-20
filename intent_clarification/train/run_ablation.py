# Input-leakage and ablation study.
#
#     python -m train.run_ablation --data train.tsv dev.tsv --folds 4 --episodes 200
#     python -m train.run_ablation --checkpoint checkpoints/marl_system.pt --episodes 200


from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from agents.marl_system import MultiAgentSystem, build_agents
from models.critic import CentralizedCritic
from models.encoder import TextEncoder
from models.fard import DialogueEnricher
from rag.facet_rag import load_facet_documents
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
from intent_clarification.rag.question_docs import load_question_documents
from intent_clarification.train.build_question_docs import add_llm_args, apply_llm_args
from intent_clarification.train.train_marl import run_episode
from intent_clarification.utils.manifest import build_manifest, write_manifest

RESPONSE_MODES = ["none", "hypothetical", "oracle_actual"]
FACET_DOC_SETTINGS = [True, False]

# Arms of Table 9 (rows 1, 4, 5), as (response_mode, use_facet_documents, label). The user's real
# answers to questions already asked are part of the history in every arm.
# "oracle_actual" is not reported: in ClariQ the recorded answer to a candidate is the
# relevance label itself (relevant = affirmative), so that arm measures label access, not a
# model. It stays available through --arm / --full for diagnostics.
FOCUSED_ARMS = [
    ("none", True, "FARD without the pseudo-turn (actual user responses only)"),
    ("hypothetical", False, "FARD without facet-aware response documents"),
    ("hypothetical", True, "FARD (full model)"),
]


def evaluate_arm(
    episodes,
    facet_docs,
    agents,
    critic,
    enricher,
    marl_cfg,
    response_mode,
    use_facet_documents,
    num_episodes,
    rng,
    question_docs=None,
):
    """num_episodes=None evaluates every episode once, in order; an integer samples that many
    episodes with `rng`."""
    marl_cfg = copy.copy(marl_cfg)
    marl_cfg.response_mode = response_mode
    marl_cfg.use_facet_documents = use_facet_documents

    system = MultiAgentSystem(agents, enricher, temperature=marl_cfg.temperature)
    env = ClarificationEnv(
        facet_documents=facet_docs,
        question_documents=question_docs,
        max_turns=marl_cfg.max_turns_per_episode,
        use_facet_documents=use_facet_documents,
    )
    # No optimiser step is taken (run_episode(train=False) skips backward and step), but
    # run_episode still needs optimiser objects.
    dummy_agent_optim = AdamW([p for a in agents for p in a.parameters()], lr=0.0)
    dummy_critic_optim = AdamW(critic.parameters(), lr=0.0)

    chosen = (
        list(episodes)
        if num_episodes is None
        else [rng.choice(episodes) for _ in range(num_episodes)]
    )
    rewards = []
    for ep in chosen:
        with torch.no_grad():
            total_reward, _turns = run_episode(
                ep,
                env,
                system,
                critic,
                marl_cfg,
                dummy_agent_optim,
                dummy_critic_optim,
                train=False,
            )
        rewards.append(total_reward)
    success_rate = sum(1 for r in rewards if r > 0) / len(rewards)
    return {
        "response_mode": response_mode,
        "use_facet_documents": use_facet_documents,
        "num_episodes": len(chosen),
        "mean_reward": sum(rewards) / len(rewards),
        "success_rate": success_rate,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, nargs="+", default=None)
    parser.add_argument(
        "--folds",
        type=int,
        nargs="+",
        default=None,
        help="evaluate only topics whose fold (topic_id %% 5) is in this list, e.g. the test fold",
    )
    parser.add_argument("--question-docs", type=str, default=CONFIG.question_docs_path)
    add_llm_args(parser)
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="marl_system.pt from train.train_marl; without it, freshly initialised "
        "weights are used (still a valid input-leakage check)",
    )
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--local-dir", type=str, default=None)
    parser.add_argument(
        "--episodes",
        type=int,
        default=200,
        help="episodes sampled per arm; 0 evaluates every episode once",
    )
    parser.add_argument(
        "--arm",
        type=str,
        default=None,
        help="evaluate one arm, as <response_mode>:<docs|nodocs> (use with its own --checkpoint)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="run all 6 arms (3 response modes x 2 document settings); "
        "default: the 3 arms of Table 9, rows 1, 4, 5 (FOCUSED_ARMS)",
    )
    parser.add_argument("--out", type=str, default="./data_files/ablation_results.json")
    args = parser.parse_args()
    apply_llm_args(args)

    set_global_seed(CONFIG.seed.seed, CONFIG.seed.deterministic_cudnn)
    if args.local_dir:
        CONFIG.encoder.local_dir = args.local_dir

    data_paths = args.data
    if not data_paths:
        data_paths = [os.path.join(CONFIG.data_dir, "synthetic_train.tsv")]
        write_synthetic_clariq_tsv(data_paths[0], seed=CONFIG.seed.seed)
        print(f"No --data given; wrote a synthetic ClariQ-shaped sample to {data_paths[0]}")

    rows = [r for p in data_paths for r in load_tsv(p)]
    bank = QuestionBank.from_rows(rows)
    episodes = build_episodes(rows, bank)
    if args.folds is not None:
        episodes = [ep for ep in episodes if topic_fold(ep.topic_id) in set(args.folds)]
    question_docs = load_question_documents(args.question_docs)
    facet_docs = {} if question_docs else load_facet_documents(CONFIG.facet_docs_path)

    llm_client = build_llm_client(CONFIG.llm)
    enricher = DialogueEnricher(llm_client)

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
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        for a, sd in zip(agents, ckpt["agents"]):
            a.load_state_dict(sd)
        critic.load_state_dict(ckpt["critic"])
        print(f"Loaded checkpoint: {args.checkpoint}")
    else:
        print(
            "No --checkpoint given: evaluating freshly initialized weights "
            "(valid for the leakage/input check; use --checkpoint for a trained-model ablation)."
        )

    for a in agents:
        a.eval()
    critic.eval()

    rng = random.Random(CONFIG.seed.seed)
    results = []
    if args.arm:
        mode, docs = args.arm.split(":")
        assert mode in RESPONSE_MODES and docs in ("docs", "nodocs"), args.arm
        arms = [(mode, docs == "docs", args.arm)]
    elif args.full:
        arms = [
            (rm, fd, f"{rm}/{'with' if fd else 'without'}_docs")
            for rm in RESPONSE_MODES
            for fd in FACET_DOC_SETTINGS
        ]
    else:
        arms = FOCUSED_ARMS
    num_eval = None if args.episodes == 0 else args.episodes
    for response_mode, use_fd, label in arms:
        r = evaluate_arm(
            episodes,
            facet_docs,
            agents,
            critic,
            enricher,
            CONFIG.marl,
            response_mode,
            use_fd,
            num_eval,
            rng,
            question_docs=question_docs,
        )
        r["label"] = label
        results.append(r)
        print(
            f"{label:32s} (response_mode={response_mode:14s} use_facet_documents={str(use_fd):5s}) "
            f"-> mean_reward={r['mean_reward']:+.3f}  success_rate={r['success_rate']:.3f}"
        )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    write_manifest(
        args.out + ".manifest.json",
        build_manifest(
            CONFIG, input_files={f"clariq_tsv_{i}": p for i, p in enumerate(data_paths)}
        ),
    )

    print(f"\nResults: {args.out}")
    print(f"Manifest: {args.out}.manifest.json")
    if args.full:
        print(
            "\nLeakage check: confirm 'hypothetical' is NOT close to 'oracle_actual' "
            "(oracle_actual is an upper bound that uses target-facet-derived answers and "
            "must not be reported as the method's result)."
        )
    else:
        print(
            "\nThese are the Table 8 arms. For the paper numbers, train one model per arm "
            "and evaluate each with --arm and its own --checkpoint (see the module docstring)."
        )


if __name__ == "__main__":
    main()
