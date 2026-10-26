"""First cell of every notebook: load `.env`, fail loudly on a missing `WORKSHOP_USER`, print the config.

    from utils.setup import init_lab
    LAB = init_lab()
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]


# Per-run switches an exported shell variable may override (e.g. `LABS_MODEL=ollama:qwen2.5:7b uv run …`).
# Everything else, notably credentials, comes from .env, which wins over globally exported keys.
SHELL_OVERRIDABLE = ("WORKSHOP_USER", "LABS_MODEL", "LABS_EMBEDDINGS", "LABS_JUDGE_MODEL", "LABS_VECTORSTORE")


def load_env() -> None:
    exported = {k: os.environ[k] for k in SHELL_OVERRIDABLE if os.environ.get(k)}
    load_dotenv(ROOT / ".env", override=True)
    os.environ.update(exported)
    if "WORKSHOP_USER" in exported:
        os.environ["LANGSMITH_PROJECT"] = f"chart-qa-{exported['WORKSHOP_USER']}"


def quiet_known_warnings() -> None:
    """Hide a harmless Pydantic serialization warning emitted when run context is traced; it floods cell output."""
    import warnings

    warnings.filterwarnings("ignore", message=r".*PydanticSerializationUnexpectedValue.*")
    warnings.filterwarnings("ignore", message=r".*Pydantic serializer warnings.*")
    warnings.filterwarnings("ignore", category=UserWarning, module=r"pydantic.*")


def init_lab(*, require_langsmith: bool = True, quiet: bool = False) -> dict:
    quiet_known_warnings()
    load_env()
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    from utils.models import embeddings_id, judge_model_id, model_id, openai_base_url
    from utils.workshop import scoped, workshop_user

    user = workshop_user()  # raises with a clear message if unset; deliberately first
    if require_langsmith and not os.getenv("LANGSMITH_API_KEY"):
        raise RuntimeError("LANGSMITH_API_KEY is not set. Add it to .env (see .env.example) and restart the kernel.")

    project = os.getenv("LANGSMITH_PROJECT") or scoped("chart-qa")
    if user not in project:
        project = scoped("chart-qa")  # never trace into a shared project
    os.environ["LANGSMITH_PROJECT"] = project
    os.environ.setdefault("LANGSMITH_TRACING", "true")

    info = {
        "workshop_user": user,
        "langsmith_project": project,
        "tracing": os.getenv("LANGSMITH_TRACING"),
        "model": model_id(),
        "embeddings": embeddings_id(),
        "judge_model": judge_model_id(),
        "openai_route": "LangSmith gateway" if (openai_base_url() or "").find("gateway.smith") >= 0 else (openai_base_url() or "OpenAI direct"),
    }
    if not quiet:
        for k, v in info.items():
            print(f"{k:18s}: {v}")
    return info
