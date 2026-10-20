

from dataclasses import dataclass, field
from typing import List, Optional

try:
    import torch
except ImportError:  # data, statistics and baseline tools run without torch
    torch = None


@dataclass
class Seeding:
    seed: int = 42
    deterministic_cudnn: bool = True


@dataclass
class EncoderConfig:
    # Backbone shared by the ambiguity classifier and (as a separate copy) the FARD module.
    # The paper's FARD encoder is LaBSE: MARL runs pass
    # --encoder-model sentence-transformers/LaBSE (see scripts/run_cv.sh).
    model_name: str = "bert-base-uncased"
    # Load tokenizer and model from this local folder instead of the Hugging Face Hub.
    # It needs config.json, vocab.txt, tokenizer_config.json and the model weights.
    local_dir: Optional[str] = None
    max_length: int = 256
    hidden_size: int = 768  # must match model_name
    projection_dim: int = 256
    dropout: float = 0.1


@dataclass
class AmbiguityConfig:
    num_levels: int = 4  # ambiguity levels 1..4
    triplet_margin: float = 0.5  # alpha in Eq. (1)
    lr: float = 2e-5
    mlp_lr: float = 1e-3
    batch_size: int = 16
    epochs: int = 10
    triplet_epochs: int = 5  # phase 1: encoder with triplet loss
    classifier_epochs: int = 5  # phase 2: frozen encoder, train the MLP head


@dataclass
class LLMConfig:
    # The LLM is frozen. It only enriches the inputs: it rewrites the last asked question,
    # writes a hypothetical user response, and writes the response documents (offline RAG).
    #
    # provider:
    #   "llama_local"       Llama 2-Chat through transformers (default, used in the paper)
    #   "anthropic"         Anthropic API (needs ANTHROPIC_API_KEY)
    #   "openai_compatible" any OpenAI-compatible gateway, e.g. GapGPT (base_url + api_key_env)
    #   "mock"              offline placeholder outputs, for tests only
    provider: str = "llama_local"
    model: str = "meta-llama/Llama-2-13b-chat-hf"
    # llama_local decodes greedily, so temperature is ignored there.
    temperature: float = 0.0
    max_tokens: int = 128  # max new tokens per call
    request_timeout: float = 30.0
    max_retries: int = 3

    # llama_local only. Llama 2 is gated: accept the licence on Hugging Face and log in,
    # or point local_dir at a pre-downloaded copy.
    local_dir: Optional[str] = None
    torch_dtype: str = "float16"  # "float16" | "bfloat16" | "float32"
    load_in_8bit: bool = False  # needs bitsandbytes
    device_map: str = "auto"

    # On-disk cache of LLM outputs. Decoding is greedy, so the cache never changes results;
    # it only avoids repeated calls. Set to None to disable.
    cache_path: Optional[str] = "./data_files/llm_cache.jsonl"

    # openai_compatible only. Export the key before running, e.g. export GAPGPT_API_KEY=sk-...
    base_url: str = "https://api.gapgpt.app/v1"
    api_key_env: str = "GAPGPT_API_KEY"
    # Seed sent to the API when supported. The Anthropic API has no seed parameter.
    seed: Optional[int] = 42


@dataclass
class RAGConfig:
    """Per-facet document settings (rag/facet_rag.py). Superseded by QuestionDocConfig,
    kept so older runs can be reproduced. All lengths are in whitespace-split words."""

    # "tfidf" (sparse), "dense" (embeddings) or "hybrid" (both, fused with RRF)
    retriever: str = "hybrid"
    tfidf_stop_words: str = "english"
    query_template: str = "{facet}"

    dense_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    dense_local_dir: Optional[str] = None

    fetch_k_per_retriever: int = 10  # candidates taken from each retriever before fusion
    rrf_k: int = 60  # RRF constant: 1 / (rrf_k + rank)
    top_k_chunks: int = 5

    chunk_size_words: int = 140
    chunk_overlap_words: int = 25
    grounding_threshold: float = 0.6
    min_doc_words: int = 15
    max_doc_words: int = 220


@dataclass
class QuestionDocConfig:
    """One response document per clarification question (Section 3-4-b, Appendix A).

    Built offline by train/build_question_docs.py using rag/question_docs.py.
    Corpus: the facet descriptions of the question's own topic, one description per passage.
    Query:  the question text.
    retriever, top_k and length_preset are chosen on the validation fold with
    train/tune_question_docs.py, never on the test fold.
    """

    retriever: str = "dense"  # "tfidf" | "dense"
    dense_model_name: str = "sentence-transformers/LaBSE"
    dense_local_dir: Optional[str] = None
    top_k: int = 2  # passages retrieved per question
    # 60 asks for 40-70 words (accepts 40-80); 120 asks for 90-130 (accepts 80-150).
    # See rag/question_docs.LENGTH_PRESETS.
    length_preset: int = 60
    grounding_threshold: float = 0.5  # min. share of content words found in the evidence


@dataclass
class MARLConfig:
    num_agents: int = 5
    # History window of each agent, in (question, answer) turns. The first agent sees only
    # the most recent turn, the last one the full history (None = unbounded).
    window_sizes: List[int] = field(default_factory=lambda: [1, 2, 3, 4, None])

    temperature: float = 1.0  # softmax temperature tau
    gamma: float = 0.95  # discount factor
    entropy_coef: float = 0.01  # beta
    critic_loss_coef: float = 0.5  # lambda_V
    policy_lr: float = 1e-4
    critic_lr: float = 3e-4
    max_grad_norm: float = 1.0

    # Not used: training is online, with one update per turn (see run_episode in
    # train/train_marl.py). Do not report it as the batch size.
    batch_size_episodes: int = 8
    num_training_episodes: int = 20000
    max_turns_per_episode: int = 5
    optimizer: str = "AdamW"

    # What the agents are told about the answer to a candidate question:
    #   "hypothetical"  main model: the frozen LLM rewrites the last asked question and writes
    #                   a plausible user response. It never sees a candidate, the gold facet
    #                   or any label.
    #   "none"          no LLM enrichment; the observation is the raw windowed history.
    #   "oracle_actual" diagnostic upper bound, not deployable: each candidate is paired with
    #                   the true simulated answer (env.oracle_peek).
    response_mode: str = "hypothetical"
    # False: score candidates on the question text alone, without a response document.
    use_facet_documents: bool = True

    # Stop when the mean reward over the last `eval_window` episodes reaches
    # `reward_stop_threshold`, or after `num_training_episodes`, whichever comes first.
    eval_window: int = 500
    reward_stop_threshold: float = 0.8


@dataclass
class PipelineConfig:
    seed: Seeding = field(default_factory=Seeding)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    ambiguity: AmbiguityConfig = field(default_factory=AmbiguityConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    question_docs: QuestionDocConfig = field(default_factory=QuestionDocConfig)
    marl: MARLConfig = field(default_factory=MARLConfig)

    device: str = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
    data_dir: str = "./data_files/clariq"
    facet_docs_path: str = "./data_files/facet_documents.jsonl"  # per-facet documents (old)
    question_docs_path: str = "./data_files/question_documents.jsonl"  # per-question documents
    checkpoint_dir: str = "./checkpoints"


CONFIG = PipelineConfig()
