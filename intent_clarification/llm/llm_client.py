# Clients for the frozen LLM.
#
# The LLM is never trained; it only enriches the inputs (Section 3-4-b).


from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

from intent_clarification.llm.prompts import (
    FACET_DOC_PROMPT_VERSION,
    FACET_DOC_USER_TEMPLATE,
    QUESTION_DOC_PROMPT_VERSION,
    QUESTION_DOC_USER_TEMPLATE,
    REWRITE_PROMPT_VERSION,
    REWRITE_USER_TEMPLATE,
    prompt_registry,
    sha256,
)
from intent_clarification.llm.prompts import (
    FACET_DOC_SYSTEM_PROMPT as _FACET_DOC_SYSTEM_PROMPT,
)
from intent_clarification.llm.prompts import (
    QUESTION_DOC_SYSTEM_PROMPT as _QUESTION_DOC_SYSTEM_PROMPT,
)
from intent_clarification.llm.prompts import (
    REWRITE_SYSTEM_PROMPT as _REWRITE_SYSTEM_PROMPT,
)

logger = logging.getLogger(__name__)


@dataclass
class RewriteResult:
    rewritten_question: str
    hypothetical_answer: str


_RQ = re.compile(r"rewritten\s+question\s*:\s*(.+)", re.IGNORECASE)
_UR = re.compile(r"(?:user\s+response|hypothetical\s+answer)\s*:\s*(.+)", re.IGNORECASE)


def parse_rewrite_output(raw: str, fallback_question: str):
    """Parse the P1 output. Returns (RewriteResult, parsed_ok).

    Expects the two-line format of prompt rewrite-v4,

        Rewritten question: ...
        User response: ...

    and tolerates a chatty preamble. The JSON format of the earlier prompt is still accepted.
    If nothing can be parsed, the original question is kept and the response is empty.
    """
    text = (raw or "").strip()
    rq, ur = _RQ.search(text), _UR.search(text)
    if rq or ur:
        return RewriteResult(
            rewritten_question=rq.group(1).strip() if rq else fallback_question,
            hypothetical_answer=ur.group(1).strip() if ur else "",
        ), bool(rq and ur)
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            data = json.loads(text[start : end + 1])
            return RewriteResult(
                rewritten_question=data.get("rewritten_question", fallback_question),
                hypothetical_answer=data.get("hypothetical_answer", ""),
            ), True
        except json.JSONDecodeError:
            pass
    return RewriteResult(rewritten_question=fallback_question, hypothetical_answer=""), False


def _passages_block(retrieved_passages: List[str]) -> str:
    return "\n---\n".join(retrieved_passages) if retrieved_passages else "(no passages retrieved)"


class BaseLLMClient:
    """Prompt logic shared by all backends. Subclasses implement `_call`."""

    model_id: str = "unknown"

    def __init__(self):
        self.parse_failures = 0  # P1 outputs that could not be parsed
        self.num_calls = 0  # backend calls actually made (cache hits not counted)

    def _call(self, system: str, user: str) -> str:
        raise NotImplementedError

    def _counted_call(self, system: str, user: str) -> str:
        self.num_calls = getattr(self, "num_calls", 0) + 1
        return self._call(system, user)

    def rewrite_question_and_hypothesize_answer(
        self, dialogue_history_text: str, question_to_rewrite: str
    ) -> RewriteResult:
        user = REWRITE_USER_TEMPLATE.format(
            dialogue_history=dialogue_history_text, question_to_rewrite=question_to_rewrite
        )
        raw = self._counted_call(_REWRITE_SYSTEM_PROMPT, user)
        result, ok = parse_rewrite_output(raw, question_to_rewrite)
        if not ok:
            self.parse_failures = getattr(self, "parse_failures", 0) + 1
            logger.warning("Could not fully parse P1 output; kept the original question.")
        return result

    def generate_question_document(
        self,
        topic_desc: str,
        question: str,
        retrieved_passages: List[str],
        min_words: int = 40,
        max_words: int = 70,
    ) -> str:
        system = _QUESTION_DOC_SYSTEM_PROMPT.format(min_words=min_words, max_words=max_words)
        user = QUESTION_DOC_USER_TEMPLATE.format(
            topic=topic_desc, question=question, passages=_passages_block(retrieved_passages)
        )
        return self._counted_call(system, user).strip()

    def generate_facet_document(
        self, topic_desc: str, facet_desc: str, retrieved_passages: List[str]
    ) -> str:
        user = FACET_DOC_USER_TEMPLATE.format(
            topic=topic_desc, facet=facet_desc, passages=_passages_block(retrieved_passages)
        )
        return self._counted_call(_FACET_DOC_SYSTEM_PROMPT, user)


class LlamaLocalLLMClient(BaseLLMClient):
    """Llama 2-Chat run locally with transformers.

    Decoding is greedy (do_sample=False), so outputs are deterministic. The prompt is built
    with the tokenizer's chat template. Llama 2 is gated on the Hub: accept the licence and
    log in, or pass `local_dir` with a downloaded copy. The 13B model needs about 26 GB of GPU
    memory in fp16, or about 14 GB with load_in_8bit (bitsandbytes).
    """

    def __init__(
        self,
        model: str = "meta-llama/Llama-2-13b-chat-hf",
        local_dir: Optional[str] = None,
        torch_dtype: str = "float16",
        load_in_8bit: bool = False,
        max_new_tokens: int = 128,
        device_map: str = "auto",
    ):
        super().__init__()
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        load_from = local_dir or model
        load_kwargs = {"local_files_only": True} if local_dir else {}
        self.model_id = model
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(load_from, **load_kwargs)
        model_kwargs = {"device_map": device_map}
        if load_in_8bit:
            from transformers import BitsAndBytesConfig

            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        else:
            model_kwargs["torch_dtype"] = getattr(torch, torch_dtype)
        self.model = AutoModelForCausalLM.from_pretrained(load_from, **load_kwargs, **model_kwargs)
        self.model.eval()
        self._torch = torch

    def _call(self, system: str, user: str) -> str:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        enc = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        )
        enc = {k: v.to(self.model.device) for k, v in enc.items()}
        with self._torch.no_grad():
            out = self.model.generate(
                **enc,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=None,  # unset the sampling defaults of Llama 2's generation_config
                top_p=None,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = out[0, enc["input_ids"].shape[-1] :]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


class AnthropicLLMClient(BaseLLMClient):
    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        temperature: float = 0.0,
        max_tokens: int = 256,
        max_retries: int = 3,
        request_timeout: float = 30.0,
    ):
        super().__init__()
        import anthropic  # imported here so the package works without the SDK installed

        self._client = anthropic.Anthropic()
        self.model = model
        self.model_id = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.request_timeout = request_timeout

    def _call(self, system: str, user: str) -> str:
        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                resp = self._client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    system=system,
                    messages=[{"role": "user", "content": user}],
                    timeout=self.request_timeout,
                )
                return "".join(
                    block.text for block in resp.content if getattr(block, "type", "") == "text"
                )
            except Exception as e:  # noqa: BLE001  retry on any API error
                last_err = e
                logger.warning(
                    "LLM call failed (attempt %d/%d): %s", attempt + 1, self.max_retries, e
                )
                time.sleep(min(2**attempt, 8))
        raise RuntimeError(f"LLM call failed after {self.max_retries} attempts") from last_err


class OpenAICompatibleLLMClient(BaseLLMClient):
    """Any OpenAI-compatible gateway (GapGPT, LocalAI, a vLLM server, ...). Set `base_url`
    and export the key under the name given in `api_key_env`. For GapGPT the default
    base_url is https://api.gapgpt.app/v1 and the key variable is GAPGPT_API_KEY.
    """

    def __init__(
        self,
        model: str,
        base_url: str = "https://api.gapgpt.app/v1",
        api_key_env: str = "GAPGPT_API_KEY",
        temperature: float = 0.0,
        max_tokens: int = 256,
        max_retries: int = 3,
        request_timeout: float = 30.0,
        seed: Optional[int] = None,
    ):
        super().__init__()
        from openai import OpenAI  # imported here so the package works without `openai`

        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(
                f"Environment variable {api_key_env} is not set. Export your GapGPT "
                f"(or other OpenAI-compatible provider) API key first, e.g.:\n"
                f"    export {api_key_env}=sk-..."
            )
        self._client = OpenAI(api_key=api_key, base_url=base_url, timeout=request_timeout)
        self.model = model
        self.model_id = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.seed = seed  # best effort: not every gateway or model honours it

    def _call(self, system: str, user: str) -> str:
        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                extra = {"seed": self.seed} if self.seed is not None else {}
                resp = self._client.chat.completions.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    **extra,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                )
                return resp.choices[0].message.content or ""
            except Exception as e:  # noqa: BLE001  retry on any API error
                last_err = e
                logger.warning(
                    "LLM call failed (attempt %d/%d): %s", attempt + 1, self.max_retries, e
                )
                time.sleep(min(2**attempt, 8))
        raise RuntimeError(f"LLM call failed after {self.max_retries} attempts") from last_err


class MockLLMClient(BaseLLMClient):
    """Deterministic offline stand-in for tests and smoke runs. Never use it for results."""

    model_id = "mock"

    def _call(self, system: str, user: str) -> str:
        return "Rewritten question: (mock)\nUser response: (mock)"

    def rewrite_question_and_hypothesize_answer(
        self, dialogue_history_text: str, question_to_rewrite: str
    ) -> RewriteResult:
        return RewriteResult(
            rewritten_question=question_to_rewrite,
            hypothetical_answer="Yes, that matches what I'm looking for.",
        )

    def generate_question_document(
        self,
        topic_desc: str,
        question: str,
        retrieved_passages: List[str],
        min_words: int = 40,
        max_words: int = 70,
    ) -> str:
        return f"{question} {topic_desc}: {' '.join(retrieved_passages)}".strip()

    def generate_facet_document(
        self, topic_desc: str, facet_desc: str, retrieved_passages: List[str]
    ) -> str:
        base = " ".join(retrieved_passages)[:400]
        return f"{facet_desc.capitalize()} in the context of {topic_desc}: {base}".strip()


class CachedLLMClient(BaseLLMClient):
    """Append-only JSONL cache around any backend.

    The key is the SHA-256 of (method, model id, prompt hash, inputs). Decoding is greedy, so
    the same key always gives the same output and the cache does not change results. Changing
    the model or a prompt changes the key, so stale entries are never reused.
    """

    def __init__(self, inner: BaseLLMClient, cache_path: str):
        super().__init__()
        self.inner = inner
        self.model_id = getattr(inner, "model_id", "unknown")
        self.cache_path = cache_path
        self.hits = 0
        self._cache: Dict[str, object] = {}
        self._prompt_sha = {k: v["sha256"] for k, v in prompt_registry().items()}
        if os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        rec = json.loads(line)
                        self._cache[rec["key"]] = rec["value"]
        else:
            os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)

    @property
    def parse_failures(self):  # taken from the wrapped backend
        return getattr(self.inner, "parse_failures", 0)

    @parse_failures.setter
    def parse_failures(self, value):
        pass

    @property
    def num_calls(self):
        return getattr(self.inner, "num_calls", 0)

    @num_calls.setter
    def num_calls(self, value):
        pass

    def _key(self, method: str, prompt_version: str, *args) -> str:
        payload = json.dumps(
            [method, self.model_id, self._prompt_sha.get(prompt_version, prompt_version), args],
            ensure_ascii=False,
        )
        return sha256(payload)

    def _store(self, key: str, value) -> None:
        self._cache[key] = value
        with open(self.cache_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"key": key, "value": value}, ensure_ascii=False) + "\n")

    def rewrite_question_and_hypothesize_answer(
        self, dialogue_history_text: str, question_to_rewrite: str
    ):
        key = self._key(
            "rewrite", REWRITE_PROMPT_VERSION, dialogue_history_text, question_to_rewrite
        )
        if key in self._cache:
            self.hits += 1
            v = self._cache[key]
            return RewriteResult(v["rewritten_question"], v["hypothetical_answer"])
        r = self.inner.rewrite_question_and_hypothesize_answer(
            dialogue_history_text, question_to_rewrite
        )
        self._store(
            key,
            {
                "rewritten_question": r.rewritten_question,
                "hypothetical_answer": r.hypothetical_answer,
            },
        )
        return r

    def generate_question_document(
        self,
        topic_desc: str,
        question: str,
        retrieved_passages: List[str],
        min_words: int = 40,
        max_words: int = 70,
    ) -> str:
        key = self._key(
            "question_doc",
            QUESTION_DOC_PROMPT_VERSION,
            topic_desc,
            question,
            list(retrieved_passages),
            min_words,
            max_words,
        )
        if key in self._cache:
            self.hits += 1
            return self._cache[key]
        out = self.inner.generate_question_document(
            topic_desc, question, retrieved_passages, min_words, max_words
        )
        self._store(key, out)
        return out

    def generate_facet_document(
        self, topic_desc: str, facet_desc: str, retrieved_passages: List[str]
    ) -> str:
        key = self._key(
            "facet_doc", FACET_DOC_PROMPT_VERSION, topic_desc, facet_desc, list(retrieved_passages)
        )
        if key in self._cache:
            self.hits += 1
            return self._cache[key]
        out = self.inner.generate_facet_document(topic_desc, facet_desc, retrieved_passages)
        self._store(key, out)
        return out


def build_llm_client(cfg) -> BaseLLMClient:
    if cfg.provider == "llama_local":
        client: BaseLLMClient = LlamaLocalLLMClient(
            model=cfg.model,
            local_dir=getattr(cfg, "local_dir", None),
            torch_dtype=getattr(cfg, "torch_dtype", "float16"),
            load_in_8bit=getattr(cfg, "load_in_8bit", False),
            max_new_tokens=cfg.max_tokens,
            device_map=getattr(cfg, "device_map", "auto"),
        )
    elif cfg.provider == "anthropic":
        client = AnthropicLLMClient(
            model=cfg.model,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            max_retries=cfg.max_retries,
            request_timeout=cfg.request_timeout,
        )
    elif cfg.provider == "openai_compatible":
        client = OpenAICompatibleLLMClient(
            model=cfg.model,
            base_url=cfg.base_url,
            api_key_env=cfg.api_key_env,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            max_retries=cfg.max_retries,
            request_timeout=cfg.request_timeout,
            seed=cfg.seed,
        )
    elif cfg.provider == "mock":
        return MockLLMClient()
    else:
        raise ValueError(f"Unknown LLM provider: {cfg.provider}")

    cache_path = getattr(cfg, "cache_path", None)
    return CachedLLMClient(client, cache_path) if cache_path else client
