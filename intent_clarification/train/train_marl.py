

from __future__ import annotations

import argparse
import os
import random
import sys
from collections import deque

import torch
from torch.optim import AdamW

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [_ROOT, os.path.dirname(_ROOT)]  # modules are imported both flat (agents, models, ...) and as intent_clarification.*

from agents.marl_system import MultiAgentSystem, build_agents
from models.critic import CentralizedCritic
from models.encoder import TextEncoder
from models.fard import DialogueEnricher, format_history
from rag.facet_rag import load_facet_documents
from utils.seeding import set_global_seed

from intent_clarification.config import CONFIG
from intent_clarification.data.clariq_dataset import (
    DialogueEpisode,
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
from intent_clarification.train.model_args import add_model_args, apply_model_args, architecture_dict
from intent_clarification.utils.manifest import build_manifest, write_manifest


def run_episode(
    episode: DialogueEpisode,
    env: ClarificationEnv,
    system: MultiAgentSystem,
    critic: CentralizedCritic,
    marl_cfg,
    agent_optim: AdamW,
    critic_optim: AdamW,
    train: bool = True,
):
    state = env.reset(episode)
    total_reward = 0.0
    turn = 0
    done = False

    while not done:
        candidate_texts = env.candidate_texts_for_fard()
        if not candidate_texts:
            break
        observations = system.per_agent_observations(
            marl_cfg.response_mode,
            episode.initial_request,
            state.turns,
            candidate_questions=state.remaining_candidates,
            env=env,
        )

        S_t, z_list, logits_list = system.score_all_agents(observations, candidate_texts)

        if train:
            action_index, _ = system.sample_action_training(S_t)
        else:
            action_index = MultiAgentSystem.select_action_inference(S_t)

        full_history_text_before = format_history(episode.initial_request, state.turns)
        V_st = critic(full_history_text_before, candidate_texts)

        reward, done, info = env.step(action_index)
        total_reward += reward
        turn += 1

        next_candidate_texts = env.candidate_texts_for_fard()
        if not done and next_candidate_texts:
            full_history_text_after = format_history(episode.initial_request, env.state.turns)
            with torch.no_grad():
                V_st1 = critic(full_history_text_after, next_candidate_texts)
        else:
            V_st1 = torch.zeros_like(V_st.detach())

        target = reward + marl_cfg.gamma * V_st1.detach()
        advantage = (target - V_st).detach()
        critic_loss = 0.5 * (target - V_st).pow(2)

        policy_losses = []
        for logits in logits_list:
            log_prob, entropy = _log_prob_and_entropy(logits, action_index, system.temperature)
            lpi = -log_prob * advantage
            policy_losses.append(lpi - marl_cfg.entropy_coef * entropy)
        total_policy_loss = torch.stack(policy_losses).sum()

        total_loss = marl_cfg.critic_loss_coef * critic_loss + total_policy_loss

        if train:
            agent_optim.zero_grad()
            critic_optim.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for a in system.agents for p in a.parameters()], marl_cfg.max_grad_norm
            )
            torch.nn.utils.clip_grad_norm_(critic.parameters(), marl_cfg.max_grad_norm)
            agent_optim.step()
            critic_optim.step()

        if turn >= marl_cfg.max_turns_per_episode:
            done = True

    return total_reward, turn


def _log_prob_and_entropy(logits: torch.Tensor, action_index: int, temperature: float):
    import torch.nn.functional as F

    log_probs = F.log_softmax(logits / temperature, dim=-1)
    probs = log_probs.exp()
    entropy = -(probs * log_probs).sum(dim=-1)
    return log_probs[action_index], entropy


def llm_usage(llm_client) -> dict:
    """LLM call counts, stored in the run manifest."""
    return {
        "backend_calls": getattr(llm_client, "num_calls", None),
        "cache_hits": getattr(llm_client, "hits", None),
        "p1_parse_failures": getattr(llm_client, "parse_failures", None),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=str,
        nargs="+",
        default=None,
        help="ClariQ TSV file(s), e.g. train.tsv dev.tsv",
    )
    parser.add_argument(
        "--folds",
        type=int,
        nargs="+",
        default=None,
        help="train only on topics whose fold (topic_id %% 5) is in this list "
        "(default: all topics)",
    )
    parser.add_argument(
        "--question-docs",
        type=str,
        default=CONFIG.question_docs_path,
        help="per-question documents from train.build_question_docs",
    )
    add_llm_args(parser)
    add_model_args(parser)
    parser.add_argument(
        "--no-pretrained", action="store_true", help="random tiny backbone (offline smoke test)"
    )
    parser.add_argument(
        "--local-dir",
        type=str,
        default=None,
        help="local folder with a downloaded encoder (loads fully offline)",
    )
    parser.add_argument("--share-backbone", action="store_true", default=True)
    parser.add_argument(
        "--episodes", type=int, default=None, help="override CONFIG.marl.num_training_episodes"
    )
    parser.add_argument(
        "--response-mode",
        type=str,
        default=CONFIG.marl.response_mode,
        choices=["hypothetical", "none", "oracle_actual"],
        help="hypothetical: main model; none: observed history only; "
        "oracle_actual: diagnostic upper bound (see README)",
    )
    parser.add_argument(
        "--no-facet-documents",
        action="store_true",
        help="ablation: score candidates on the question text only (no response documents)",
    )
    parser.add_argument("--seed", type=int, default=None, help="override CONFIG.seed.seed")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="override CONFIG.checkpoint_dir (use one folder per arm and seed)",
    )
    args = parser.parse_args()
    apply_llm_args(args)
    apply_model_args(args)
    if args.checkpoint_dir:
        CONFIG.checkpoint_dir = args.checkpoint_dir

    if args.seed is not None:
        CONFIG.seed.seed = args.seed
    set_global_seed(CONFIG.seed.seed, CONFIG.seed.deterministic_cudnn)
    device = torch.device(CONFIG.device)
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
    if not episodes:
        raise RuntimeError(
            "No episodes could be built from the given data (check facet/question coverage)."
        )
    print(
        f"Loaded {len(episodes)} clarification episodes across "
        f"{len({ep.topic_id for ep in episodes})} topics."
    )

    question_docs = load_question_documents(args.question_docs)
    facet_docs = {} if question_docs else load_facet_documents(CONFIG.facet_docs_path)
    if not question_docs and not facet_docs:
        print(
            "WARNING: no response documents found at "
            f"{args.question_docs}. Run `python -m train.build_question_docs` first for "
            "full fidelity to the paper's RAG-enriched candidate pathway. "
            "Continuing with un-enriched candidate questions."
        )

    llm_client = build_llm_client(CONFIG.llm)
    enricher = DialogueEnricher(llm_client)

    agents = build_agents(
        CONFIG.marl,
        CONFIG.encoder,
        share_bert_backbone=args.share_backbone,
        pretrained=not args.no_pretrained,
    )
    for a in agents:
        a.to(device)
    system = MultiAgentSystem(agents, enricher, temperature=CONFIG.marl.temperature)

    critic_encoder = TextEncoder(
        model_name=CONFIG.encoder.model_name,
        hidden_size=CONFIG.encoder.hidden_size,
        projection_dim=CONFIG.encoder.projection_dim,
        dropout=CONFIG.encoder.dropout,
        max_length=CONFIG.encoder.max_length,
        pretrained=not args.no_pretrained,
        local_dir=CONFIG.encoder.local_dir,
    ).to(device)
    critic = CentralizedCritic(critic_encoder).to(device)

    agent_params = [p for a in agents for p in a.parameters()]
    agent_optim = AdamW(agent_params, lr=CONFIG.marl.policy_lr)
    critic_optim = AdamW(critic.parameters(), lr=CONFIG.marl.critic_lr)

    CONFIG.marl.response_mode = args.response_mode
    if args.no_facet_documents:
        CONFIG.marl.use_facet_documents = False
    env = ClarificationEnv(
        facet_documents=facet_docs,
        question_documents=question_docs,
        max_turns=CONFIG.marl.max_turns_per_episode,
        use_facet_documents=CONFIG.marl.use_facet_documents,
    )

    rng = random.Random(CONFIG.seed.seed)
    num_episodes = args.episodes or CONFIG.marl.num_training_episodes
    reward_window = deque(maxlen=CONFIG.marl.eval_window)

    for ep_idx in range(1, num_episodes + 1):
        episode = rng.choice(episodes)
        total_reward, turns_used = run_episode(
            episode, env, system, critic, CONFIG.marl, agent_optim, critic_optim, train=True
        )
        reward_window.append(total_reward)

        if ep_idx % max(1, num_episodes // 20 or 1) == 0 or ep_idx == num_episodes:
            avg_reward = sum(reward_window) / len(reward_window)
            print(
                f"episode {ep_idx}/{num_episodes} turns={turns_used} reward={total_reward:.1f} avg_reward={avg_reward:.3f}"
            )
            if (
                len(reward_window) == reward_window.maxlen
                and avg_reward >= CONFIG.marl.reward_stop_threshold
            ):
                print(
                    f"Stopping criterion met: avg reward {avg_reward:.3f} >= "
                    f"{CONFIG.marl.reward_stop_threshold} over the last {len(reward_window)} episodes."
                )
                break

    usage = llm_usage(llm_client)
    print(f"LLM usage: {usage}")
    os.makedirs(CONFIG.checkpoint_dir, exist_ok=True)
    write_manifest(
        os.path.join(CONFIG.checkpoint_dir, "marl_run.manifest.json"),
        build_manifest(
            CONFIG,
            input_files={
                **{f"clariq_tsv_{i}": p for i, p in enumerate(data_paths)},
                "question_documents": args.question_docs,
            },
            extra={
                "train_folds": args.folds,
                "llm_usage": usage,
                "response_mode": CONFIG.marl.response_mode,
                "use_facet_documents": CONFIG.marl.use_facet_documents,
            },
        ),
    )
    torch.save(
        {
            "agents": [a.state_dict() for a in agents],
            "critic": critic.state_dict(),
            # train/evaluate_ranking.py restores this, so a checkpoint is evaluated with the
            # architecture it was trained with
            "architecture": {
                **architecture_dict(),
                "seed": CONFIG.seed.seed,
                "train_folds": args.folds,
            },
        },
        os.path.join(CONFIG.checkpoint_dir, "marl_system.pt"),
    )
    print(f"Saved MARL checkpoint to {os.path.join(CONFIG.checkpoint_dir, 'marl_system.pt')}")


if __name__ == "__main__":
    main()
