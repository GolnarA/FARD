
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.facet_rag import build_dense_embedder

from intent_clarification.config import CONFIG
from intent_clarification.data.clariq_dataset import QuestionBank, load_tsv, write_synthetic_clariq_tsv
from intent_clarification.llm.llm_client import build_llm_client
from intent_clarification.rag.question_docs import LENGTH_PRESETS, QuestionDocumentBuilder
from intent_clarification.utils.manifest import build_manifest, write_manifest


def add_llm_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--llm-provider",
        type=str,
        default=CONFIG.llm.provider,
        choices=["llama_local", "anthropic", "openai_compatible", "mock"],
    )
    parser.add_argument("--llm-model", type=str, default=None, help="override CONFIG.llm.model")
    parser.add_argument(
        "--llm-local-dir",
        type=str,
        default=None,
        help="local folder with a downloaded Llama 2-Chat model",
    )
    parser.add_argument(
        "--llm-8bit", action="store_true", help="load the local LLM in 8-bit (bitsandbytes)"
    )
    parser.add_argument("--llm-base-url", type=str, default=None)
    parser.add_argument("--no-llm-cache", action="store_true")


def apply_llm_args(args) -> None:
    cfg = CONFIG.llm
    cfg.provider = args.llm_provider
    if args.llm_model:
        cfg.model = args.llm_model
    if args.llm_local_dir:
        cfg.local_dir = args.llm_local_dir
    if args.llm_8bit:
        cfg.load_in_8bit = True
    if args.llm_base_url:
        cfg.base_url = args.llm_base_url
    if args.no_llm_cache:
        cfg.cache_path = None


def make_builder(llm_client, retriever: str, top_k: int, length_preset: int, dense_embedder=None):
    qd = CONFIG.question_docs
    return QuestionDocumentBuilder(
        llm_client=llm_client,
        retriever_mode=retriever,
        dense_embedder=dense_embedder,
        top_k=top_k,
        length_preset=length_preset,
        grounding_threshold=qd.grounding_threshold,
    )


def write_stats(records, out_path: str, builder: QuestionDocumentBuilder) -> dict:
    statuses = Counter(r["status"] for r in records)
    n = len(records)
    fallback = sum(c for s, c in statuses.items() if s.startswith("fallback"))
    stats = {
        "num_questions": n,
        "status_counts": dict(statuses),
        "fallback_rate": round(fallback / n, 4) if n else None,
        "retriever_mode_used": builder.retriever_mode,
        "top_k": builder.top_k,
        "length_preset": builder.length_preset,
    }
    with open(out_path + ".stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    return stats


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
        "--retriever", type=str, default=CONFIG.question_docs.retriever, choices=["dense", "tfidf"]
    )
    parser.add_argument("--dense-model", type=str, default=CONFIG.question_docs.dense_model_name)
    parser.add_argument("--dense-local-dir", type=str, default=CONFIG.question_docs.dense_local_dir)
    parser.add_argument("--top-k", type=int, default=CONFIG.question_docs.top_k)
    parser.add_argument(
        "--length-preset",
        type=int,
        default=CONFIG.question_docs.length_preset,
        choices=sorted(LENGTH_PRESETS),
    )
    parser.add_argument("--out", type=str, default=CONFIG.question_docs_path)
    add_llm_args(parser)
    args = parser.parse_args()
    apply_llm_args(args)

    data_paths = args.data
    if not data_paths:
        data_paths = [os.path.join(CONFIG.data_dir, "synthetic_train.tsv")]
        write_synthetic_clariq_tsv(data_paths[0], seed=CONFIG.seed.seed)
        print(f"No --data given; wrote a synthetic ClariQ-shaped sample to {data_paths[0]}")
    rows = [r for p in data_paths for r in load_tsv(p)]
    bank = QuestionBank.from_rows(rows)

    if CONFIG.llm.provider == "mock":
        print("WARNING: mock LLM -- outputs are placeholders, never report results built on them.")
    llm_client = build_llm_client(CONFIG.llm)

    dense_embedder = (
        build_dense_embedder(args.dense_model, local_dir=args.dense_local_dir)
        if args.retriever == "dense"
        else None
    )
    CONFIG.question_docs.retriever = args.retriever
    CONFIG.question_docs.top_k = args.top_k
    CONFIG.question_docs.length_preset = args.length_preset
    builder = make_builder(
        llm_client, args.retriever, args.top_k, args.length_preset, dense_embedder
    )

    records = builder.build_and_cache(bank, args.out)
    stats = write_stats(records, args.out, builder)
    write_manifest(
        args.out + ".manifest.json",
        build_manifest(
            CONFIG,
            input_files={f"clariq_tsv_{i}": p for i, p in enumerate(data_paths)},
            extra={
                "question_doc_stats": stats,
                "llm_backend_calls": getattr(llm_client, "num_calls", None),
            },
        ),
    )
    print(f"Built {len(records)} question documents -> {args.out}")
    print(
        f"Retriever actually used: {builder.retriever_mode}; top-k={builder.top_k}; "
        f"length preset={builder.length_preset}"
    )
    print(f"Status counts: {stats['status_counts']}  (fallback rate {stats['fallback_rate']})")


if __name__ == "__main__":
    main()
