"""The one place that picks models for every lab: generation, embeddings, and the LLM judge.

Three env switches, each a ``provider:model`` string:

| Env var            | Default                           | Used for                          |
|--------------------|-----------------------------------|-----------------------------------|
| `LABS_MODEL`       | `openai:gpt-5.4-mini`             | the chart assistant (generation)  |
| `LABS_EMBEDDINGS`  | `openai:text-embedding-3-small`   | the vector store                  |
| `LABS_JUDGE_MODEL` | `openai:gpt-5.4-mini`             | LLM-as-judge evaluators           |

Local models work the same way through Ollama: `LABS_MODEL="ollama:qwen2.5:7b"` and
`LABS_EMBEDDINGS="ollama:nomic-embed-text"`. Judges default to a hosted model on purpose, because a
small local judge is too noisy to compare experiments.

**OpenAI through the LangSmith gateway (default).** `openai:` models are called through
`LABS_OPENAI_BASE_URL` (default: the LangSmith gateway), authenticated with your `LANGSMITH_API_KEY`, so
attendees need one key only. To call OpenAI directly instead, set `LABS_OPENAI_BASE_URL=""` and provide
`OPENAI_API_KEY`.

API keys are read from the environment only; nothing is hard-coded.
"""

from __future__ import annotations

import os

from langchain.chat_models import init_chat_model

GATEWAY_URL = "https://gateway.smith.langchain.com/openai/v1"

DEFAULT_MODEL = "openai:gpt-5.4-mini"
DEFAULT_EMBEDDINGS = "openai:text-embedding-3-small"
DEFAULT_JUDGE_MODEL = "openai:gpt-5.4-mini"


def model_id() -> str:
    return os.getenv("LABS_MODEL") or DEFAULT_MODEL


def embeddings_id() -> str:
    return os.getenv("LABS_EMBEDDINGS") or DEFAULT_EMBEDDINGS


def judge_model_id() -> str:
    return os.getenv("LABS_JUDGE_MODEL") or DEFAULT_JUDGE_MODEL


def _max_retries() -> int:
    """Client-level retry budget so a transient connection drop (a laptop napping mid-run) self-heals."""
    raw = os.getenv("LABS_MAX_RETRIES", "")
    return int(raw) if raw.isdigit() else 6


def openai_base_url() -> str | None:
    """The OpenAI-compatible endpoint to use. Default: LangSmith gateway. Empty string = direct OpenAI."""
    raw = os.getenv("LABS_OPENAI_BASE_URL")
    if raw is None:
        return GATEWAY_URL
    return raw.strip() or None


def _provider_kwargs(spec: str) -> dict:
    provider = spec.split(":", 1)[0]
    if provider == "openai":
        base_url = openai_base_url()
        if base_url and "gateway.smith.langchain.com" in base_url:
            return {"base_url": base_url, "api_key": os.getenv("LANGSMITH_API_KEY_GATEWAY")}
        if base_url:
            return {"base_url": base_url}
        return {}
    if provider == "ollama":
        return {"base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")}
    return {}


def get_model(spec: str | None = None, **kwargs):
    """Return the chat model named by `spec` (default: `LABS_MODEL`)."""
    spec = spec or model_id()
    params = {"max_retries": _max_retries(), **_provider_kwargs(spec), **kwargs}
    if spec.startswith("ollama:"):
        params.pop("max_retries", None)  # ChatOllama has no client retry option
        params.setdefault("temperature", 0)
    return init_chat_model(spec, **params)


def get_judge_model(**kwargs):
    """The LLM-as-judge model (`LABS_JUDGE_MODEL`, hosted by default), configured separately from generation."""
    return get_model(judge_model_id(), **kwargs)


def get_embeddings(spec: str | None = None):
    """Return the embeddings model named by `spec` (default: `LABS_EMBEDDINGS`)."""
    from langchain.embeddings import init_embeddings

    spec = spec or embeddings_id()
    kwargs = _provider_kwargs(spec)
    if spec.startswith("openai:"):
        kwargs["max_retries"] = _max_retries()
    return init_embeddings(spec, **kwargs)
