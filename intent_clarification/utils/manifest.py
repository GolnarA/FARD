# Run manifest: a JSON record of what is needed to reproduce a run (models, decoding
# parameters, seeds, retriever settings, prompt hashes, library versions, input-file hashes).



from __future__ import annotations

import dataclasses
import hashlib
import json
import platform
import time
from pathlib import Path
from typing import Optional


def file_sha256(path: Optional[str]) -> Optional[str]:
    if not path or not Path(path).exists():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _version(pkg: str) -> Optional[str]:
    try:
        import importlib.metadata as md

        return md.version(pkg)
    except Exception:  # noqa: BLE001
        return None


def build_manifest(
    config, extra: Optional[dict] = None, input_files: Optional[dict] = None
) -> dict:
    from intent_clarification.llm.prompts import prompt_registry

    llm_cfg = dataclasses.asdict(config.llm)
    manifest = {
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": platform.python_version(),
        "packages": {
            p: _version(p)
            for p in [
                "torch",
                "transformers",
                "accelerate",
                "bitsandbytes",
                "sentence-transformers",
                "scikit-learn",
                "openai",
                "anthropic",
                "numpy",
            ]
        },
        "seed": dataclasses.asdict(config.seed),
        "llm": llm_cfg,
        "llm_is_mock": config.llm.provider == "mock",
        "prompts": prompt_registry(),
        "rag": dataclasses.asdict(config.rag),
        "question_docs": dataclasses.asdict(config.question_docs),
        "encoder": dataclasses.asdict(config.encoder),
        "marl": dataclasses.asdict(config.marl),
        "input_file_sha256": {k: file_sha256(v) for k, v in (input_files or {}).items()},
    }
    if extra:
        manifest.update(extra)
    return manifest


def write_manifest(path: str, manifest: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
