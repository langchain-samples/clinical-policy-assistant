"""Single place to pick the chat model for every lab.

Swap providers here (or via the LABS_MODEL env var) and every notebook follows.
Default is a cheap model for teaching; the research capstone may override to a
stronger one (see docs/PROD-READINESS.md P8).

API keys are read from the environment (see content/.env.example) — never
hardcode them here.
"""

import os

from langchain.chat_models import init_chat_model

# Two tiers (OpenAI — no Anthropic keys, see PRD D15). Override the default with LABS_MODEL.
# Model IDs pending team confirmation (team brief) — swap here in one place.
DEFAULT_MODEL = os.getenv("LABS_MODEL", "openai:gpt-5.4-mini")  # light — most labs (no gpt-5.5-mini exists)
HEAVY_MODEL = "openai:gpt-5.5"                                   # strong — research capstone


def _max_retries() -> int:
    """Client-level retry budget (exponential backoff) so a transient connection drop — e.g. ECONNRESET
    when a laptop naps mid-run — self-heals instead of killing a long agent/eval run. Trusted int env
    override only; anything non-numeric or negative falls back to the default."""
    raw = os.getenv("LABS_MAX_RETRIES", "")
    return int(raw) if raw.isdigit() else 6


def get_model(model: str | None = None):
    """Return a chat model. Pass a `provider:model` string (e.g. HEAVY_MODEL) to override."""
    return init_chat_model(model or DEFAULT_MODEL, max_retries=_max_retries())
