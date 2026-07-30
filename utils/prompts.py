"""Manage lab system prompts in LangSmith (Prompt Hub), with an offline-safe fallback.

Edit a prompt in the LangSmith UI and it's picked up on the next run — no code change. If
LangSmith is unreachable (offline, no key, not seeded yet), labs fall back to a committed local
string so they always run standalone.

Security: `pull_prompt` runs with `secrets_from_env=False` (the default), so no environment
secrets are materialized into the prompt; we only read the template text (no deserialization).
"""

import os

# UI base for building "edit this prompt" links (override for self-hosted LangSmith).
SMITH_UI_BASE = os.getenv("LANGSMITH_UI_BASE", "https://smith.langchain.com")


def _extract_text(prompt_obj) -> str | None:
    """Read the system/template text out of a pulled LangChain prompt (safe attribute reads only)."""
    messages = getattr(prompt_obj, "messages", None)
    if messages:
        for m in messages:
            tmpl = getattr(getattr(m, "prompt", None), "template", None)
            if isinstance(tmpl, str) and tmpl.strip():
                return tmpl
    tmpl = getattr(prompt_obj, "template", None)
    return tmpl if isinstance(tmpl, str) and tmpl.strip() else None


def get_prompt(name: str, fallback: str) -> str:
    """Return prompt `name` from LangSmith, or `fallback` if it can't be pulled.

    Offline-safe: any failure (no key, offline, not seeded) returns `fallback`.
    """
    if not os.getenv("LANGSMITH_API_KEY"):
        return fallback
    try:
        from langsmith import Client

        obj = Client().pull_prompt(name)  # secrets_from_env=False by default
        return _extract_text(obj) or fallback
    except Exception:
        return fallback


def seed_prompt(name: str, text: str) -> str:
    """Publish `text` as prompt `name` in LangSmith (one-time setup). Returns the prompt URL."""
    from langchain_core.prompts import ChatPromptTemplate
    from langsmith import Client

    prompt = ChatPromptTemplate.from_messages([("system", text)])
    return Client().push_prompt(name, object=prompt)


def prompt_url(name: str) -> str:
    """LangSmith UI link to view/edit the prompt, **scoped to the configured workspace**.

    An unscoped `/prompts/<name>` link opens in whatever workspace the browser was last logged into —
    possibly the wrong one. Scoping with `?organizationId=<LANGSMITH_WORKSPACE_ID>` (the same param
    the LangSmith SDK uses) pins the link to this lab's workspace. This only fixes the *link*; the
    prompt data always lives wherever the API key + `LANGSMITH_WORKSPACE_ID` resolve.
    """
    base = f"{SMITH_UI_BASE}/prompts/{name}"
    workspace = os.getenv("LANGSMITH_WORKSPACE_ID")
    return f"{base}?organizationId={workspace}" if workspace else base


def dataset_url(name: str) -> str:
    """LangSmith UI link to the eval dataset `name`, **scoped to the configured workspace**.

    So a presenter can open and review the dataset (what the agent is graded on) before running an
    experiment — not just the experiment link afterward. When a key is available, use the SDK's own
    dataset URL (already `/o/<workspace>/datasets/<id>`); offline or on failure, fall back to the
    workspace-scoped datasets index so the link still opens the right place.
    """
    workspace = os.getenv("LANGSMITH_WORKSPACE_ID")
    index = f"{SMITH_UI_BASE}/o/{workspace}/datasets" if workspace else f"{SMITH_UI_BASE}/datasets"
    if not os.getenv("LANGSMITH_API_KEY"):
        return index
    try:
        from langsmith import Client

        return Client().read_dataset(dataset_name=name).url
    except Exception:
        return index
