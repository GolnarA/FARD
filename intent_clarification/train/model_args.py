

from __future__ import annotations

import argparse
from typing import List, Optional

from intent_clarification.config import CONFIG


def parse_windows(values: Optional[List[str]]) -> Optional[list]:
    if not values:
        return None
    return [None if v.lower() in ("full", "none", "all") else int(v) for v in values]


def add_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--window-sizes",
        type=str,
        nargs="+",
        default=None,
        help="history window of each agent in turns, 'full' for the whole history "
        "(default: 1 2 3 4 full; single agent: full)",
    )
    parser.add_argument(
        "--encoder-model",
        type=str,
        default=None,
        help="override CONFIG.encoder.model_name (e.g. bert-base-uncased for FARD-BERT)",
    )
    parser.add_argument(
        "--encoder-hidden-size",
        type=int,
        default=None,
        help="hidden size of --encoder-model if it is not 768",
    )


def apply_model_args(args) -> None:
    windows = parse_windows(getattr(args, "window_sizes", None))
    if windows is not None:
        CONFIG.marl.window_sizes = windows
        CONFIG.marl.num_agents = len(windows)
    if getattr(args, "encoder_model", None):
        CONFIG.encoder.model_name = args.encoder_model
    if getattr(args, "encoder_hidden_size", None):
        CONFIG.encoder.hidden_size = args.encoder_hidden_size


def architecture_dict() -> dict:
    return {
        "window_sizes": list(CONFIG.marl.window_sizes),
        "num_agents": CONFIG.marl.num_agents,
        "encoder_model": CONFIG.encoder.model_name,
        "encoder_hidden_size": CONFIG.encoder.hidden_size,
        "response_mode": CONFIG.marl.response_mode,
        "use_facet_documents": CONFIG.marl.use_facet_documents,
    }


def restore_architecture(arch: Optional[dict]) -> None:
    if not arch:
        return
    CONFIG.marl.window_sizes = list(arch["window_sizes"])
    CONFIG.marl.num_agents = arch["num_agents"]
    CONFIG.encoder.model_name = arch["encoder_model"]
    CONFIG.encoder.hidden_size = arch["encoder_hidden_size"]
