"""LangSmith-managed pieces a deep agent plugs into (Lab 2, Part B).

- **Context Hub**: the agent's instructions (`AGENTS.md`) and skills live in a versioned agent repo in LangSmith,
  so they can be reviewed, rolled back and shared without a code deploy. Source of truth: `agent_context/`.
- **Sandboxes**: an isolated, managed VM that backs the deep agent's `execute` tool, so model-written code never
  runs on a laptop or an app server.

Every helper degrades gracefully: if a service isn't enabled for your workspace, it returns None and the lab
continues without that capability. Names go through `scoped()` so 30 attendees never share a repo or sandbox.
"""

from __future__ import annotations

import contextlib

from utils.corpus import ROOT
from utils.workshop import scoped

AGENT_CONTEXT_DIR = ROOT / "agent_context"
SANDBOX_SNAPSHOT = "system/default"


def hub_repo() -> str:
    """Your scoped Context Hub agent repo, e.g. `-/chart-qa-deep-agent-jane-doe`."""
    return f"-/{scoped('chart-qa-deep-agent')}"


def local_agent_context() -> dict[str, str]:
    """`agent_context/` as {"AGENTS.md": ..., "skills/<name>/SKILL.md": ...}."""
    return {p.relative_to(AGENT_CONTEXT_DIR).as_posix(): p.read_text()
            for p in sorted(AGENT_CONTEXT_DIR.rglob("*")) if p.is_file()}


def publish_agent_context(files: dict[str, str] | None = None, *, description: str | None = None) -> str | None:
    """Push files to your Context Hub repo as ONE new commit. Returns the commit URL (None if unavailable)."""
    from langsmith import Client
    from langsmith.schemas import FileEntry

    files = files if files is not None else local_agent_context()
    try:
        return Client().push_agent(
            hub_repo(),
            files={path: FileEntry(type="file", content=content) for path, content in files.items()},
            description=description or "SYNTHETIC Northfield Health chart deep agent: team conventions + skills",
        )
    except Exception as exc:  # noqa: BLE001: the lab continues with local files
        print(f"Context Hub unavailable ({type(exc).__name__}: {str(exc)[:120]}); using local agent_context/ instead.")
        return None


def pull_agent_context(version: str | None = None) -> tuple[str | None, dict[str, str]]:
    """(commit hash, files) of your agent repo, at `version` (a commit hash) or latest. (None, {}) if unavailable."""
    from langsmith import Client
    from langsmith.schemas import FileEntry

    try:
        ctx = Client().pull_agent(hub_repo(), version=version)
    except Exception:  # noqa: BLE001
        return None, {}
    return ctx.commit_hash, {p: e.content for p, e in ctx.files.items() if isinstance(e, FileEntry)}


def context_backend(published: bool):
    """Where the agent reads its conventions and skills: Context Hub if published, else the local folder."""
    if published:
        from deepagents.backends import ContextHubBackend

        return ContextHubBackend(hub_repo())
    from deepagents.backends import FilesystemBackend

    return FilesystemBackend(root_dir=str(AGENT_CONTEXT_DIR), virtual_mode=True)


def start_sandbox(idle_ttl_seconds: int = 1800):
    """Create a scoped sandbox that spans several notebook cells. Returns a `LangSmithSandbox` backend, or None
    if sandboxes aren't available. It deletes itself after `idle_ttl_seconds` idle; call `stop_sandbox` when done."""
    try:
        import uuid

        from deepagents.backends import LangSmithSandbox
        from langsmith.sandbox import SandboxClient

        sb = SandboxClient().create_sandbox(
            snapshot_name=SANDBOX_SNAPSHOT, name=f"{scoped('chart-qa')}-{uuid.uuid4().hex[:6]}",
            timeout=180, idle_ttl_seconds=idle_ttl_seconds,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"LangSmith Sandboxes unavailable ({type(exc).__name__}: {str(exc)[:120]}); continuing without `execute`.")
        return None
    backend = LangSmithSandbox(sb)
    backend.execute("mkdir -p /work")
    return backend


def stop_sandbox(backend) -> None:
    """Delete a sandbox created by `start_sandbox` (no-op for None)."""
    if backend is None:
        return
    from langsmith.sandbox import SandboxClient

    try:
        SandboxClient().delete_sandbox(backend.id)
        print(f"deleted sandbox {backend.id}")
    except Exception as exc:  # noqa: BLE001
        print(f"(could not delete {backend.id}: {type(exc).__name__}; it expires when idle)")


@contextlib.contextmanager
def managed_sandbox(timeout: int = 180):
    """Yield a `LangSmithSandbox` backend (deleted on exit), or None if sandboxes aren't available."""
    try:
        from deepagents.backends import LangSmithSandbox
        from langsmith.sandbox import SandboxClient

        cm = SandboxClient().sandbox(snapshot_name=SANDBOX_SNAPSHOT, timeout=timeout)
        sb = cm.__enter__()
    except Exception as exc:  # noqa: BLE001
        print(f"LangSmith Sandboxes unavailable ({type(exc).__name__}: {str(exc)[:120]}); continuing without `execute`.")
        yield None
        return
    try:
        backend = LangSmithSandbox(sb)
        backend.execute("mkdir -p /work")
        yield backend
    finally:
        cm.__exit__(None, None, None)


__all__ = ["AGENT_CONTEXT_DIR", "context_backend", "hub_repo", "local_agent_context", "managed_sandbox",
           "publish_agent_context", "pull_agent_context"]
