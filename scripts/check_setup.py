#!/usr/bin/env python
"""One-command workshop setup check. Prints PASS / WARN / FAIL per item and exits non-zero on any FAIL.

    uv run python scripts/check_setup.py

Checks: Python version, .env, WORKSHOP_USER, LangSmith key + auth, chat model call, embeddings call,
judge model call, local vector store, and that a trace actually lands in your LangSmith project.
"""

from __future__ import annotations

import os
import sys
import time
import traceback
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RESULTS: list[tuple[str, str, str]] = []
ICON = {"PASS": "✅", "WARN": "⚠️ ", "FAIL": "❌"}


def record(name: str, status: str, detail: str = "") -> None:
    RESULTS.append((name, status, detail))
    print(f"{ICON[status]} {status:4s}  {name:28s} {detail}", flush=True)


def check(name: str, fn, *, warn_only: bool = False) -> None:
    try:
        detail = fn() or ""
        record(name, "PASS", str(detail))
    except Exception as exc:  # noqa: BLE001: report every failure as a line item, never crash
        msg = f"{type(exc).__name__}: {str(exc).splitlines()[0][:160] if str(exc) else ''}"
        record(name, "WARN" if warn_only else "FAIL", msg)
        if os.getenv("CHECK_SETUP_DEBUG"):
            traceback.print_exc()


def main() -> int:
    print("Northfield Health chart Q&A workshop: setup check\n")

    def py():
        v = sys.version_info
        if (v.major, v.minor) != (3, 11):
            raise RuntimeError(f"Python {v.major}.{v.minor} found; this lab pins 3.11 (run via `uv run`)")
        return f"{v.major}.{v.minor}.{v.micro}"

    check("Python 3.11", py)

    def env():
        from utils.setup import load_env

        path = ROOT / ".env"
        if not path.exists():
            raise FileNotFoundError("no .env: run `cp .env.example .env` and fill it in")
        load_env()
        return str(path.relative_to(ROOT))

    check(".env loaded", env)

    def user():
        from utils.workshop import scoped, workshop_user

        return f"{workshop_user()} → e.g. dataset {scoped('chart-qa-golden')}"

    check("WORKSHOP_USER", user)

    def ls_auth():
        if not os.getenv("LANGSMITH_API_KEY"):
            raise RuntimeError("LANGSMITH_API_KEY is not set")
        from langsmith import Client

        next(iter(Client().list_projects(limit=1)), None)
        return f"project: {os.getenv('LANGSMITH_PROJECT')}"

    check("LangSmith key + auth", ls_auth)

    from utils.models import embeddings_id, get_embeddings, get_judge_model, get_model, judge_model_id, model_id

    def chat():
        reply = get_model(max_retries=1).invoke("Reply with the single word: ready").content
        return f"{model_id()} → {str(reply).strip()[:30]!r}"

    check("Chat model call", chat)

    def emb():
        vec = get_embeddings().embed_query("creatinine trend")
        return f"{embeddings_id()} → dim {len(vec)}"

    check("Embeddings call", emb)

    def judge():
        if judge_model_id() == model_id():
            return f"{judge_model_id()} (same as chat model)"
        get_judge_model(max_retries=1).invoke("Reply with: ok")
        return judge_model_id()

    check("Judge model call", judge)

    def store():
        import chromadb  # noqa: F401
        from langchain_chroma import Chroma  # noqa: F401

        return "Chroma available (persists to .chroma/)"

    check("Local vector store", store, warn_only=True)

    def trace():
        from langsmith import Client, traceable

        client = Client()
        run_id = uuid.uuid4()
        project = os.getenv("LANGSMITH_PROJECT")

        @traceable(name="setup_check", client=client, project_name=project)
        def ping(x: str) -> str:
            return f"pong {x}"

        ping("chart-qa", langsmith_extra={"run_id": run_id})
        client.flush()
        for _ in range(20):
            try:
                run = client.read_run(run_id)
                return f"landed in {project!r} (run {str(run.id)[:8]}…)"
            except Exception:  # noqa: BLE001
                time.sleep(1.5)
        raise TimeoutError(f"trace not visible in {project!r} after 30s")

    check("Trace lands in LangSmith", trace)

    failed = [r for r in RESULTS if r[1] == "FAIL"]
    print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed" + (": fix the ❌ items above." if failed else ". You're ready. 🎉"))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
