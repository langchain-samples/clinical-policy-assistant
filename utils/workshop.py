"""Per-attendee resource scoping for concurrent workshop runs.

About 30 people run these notebooks against the **same** LangSmith workspace at the same time. Any
resource addressed by a shared *name* would be overwritten by whoever runs last unless each person's name
is unique. That covers tracing projects, Prompt Hub prompts, eval datasets, experiments, annotation
queues, and run rules.

`WORKSHOP_USER` (set in `.env`, e.g. "jane-doe") is the per-user token spliced into every one of those
names. Read it through `workshop_user()` so a missing or placeholder value fails loudly instead of
silently colliding on a shared default.

Ported from the `medical-order-processing-agent-workshop` pattern.
"""

from __future__ import annotations

import os
import re
import warnings

# The value shipped in .env.example. If it's still this, the user didn't set it.
_PLACEHOLDER = "<first-last>"


def workshop_user() -> str:
    """Return the sanitized WORKSHOP_USER, or raise if it isn't set.

    Raises a clear error when the variable is missing or still the `<first-last>` placeholder, so no one
    accidentally scopes their resources to a shared default name and clobbers a neighbor.
    """
    raw = (os.environ.get("WORKSHOP_USER") or "").strip()
    if not raw or raw == _PLACEHOLDER:
        raise RuntimeError(
            "WORKSHOP_USER is not set. Set it in your .env to a unique value "
            '(e.g. WORKSHOP_USER="jane-doe") so your workshop resources don\'t '
            "collide with other attendees, then restart the kernel / reload the environment."
        )
    # Keep resource names tidy and portable: lowercase, only [a-z0-9-].
    slug = re.sub(r"[^a-z0-9]+", "-", raw.lower()).strip("-")
    if not slug:
        raise RuntimeError(
            f"WORKSHOP_USER={raw!r} has no usable characters. Use letters/digits "
            '(e.g. WORKSHOP_USER="jane-doe").'
        )
    if slug != raw:
        # `.env` interpolates the *raw* value into LANGSMITH_PROJECT, while resources created through
        # this helper use the slug. Anything needing normalization therefore splits the two apart.
        warnings.warn(
            f"WORKSHOP_USER={raw!r} was normalized to {slug!r}. Your LANGSMITH_PROJECT still carries the "
            f"raw value, so your tracing project and your other resources will be named inconsistently. "
            f'Set WORKSHOP_USER="{slug}" in .env to keep them aligned.',
            stacklevel=2,
        )
    return slug


def scoped(name: str) -> str:
    """Suffix a shared resource `name` with the current WORKSHOP_USER.

    Example (with WORKSHOP_USER="jane-doe")::

        scoped("chart-qa-golden")  # -> "chart-qa-golden-jane-doe"
    """
    return f"{name}-{workshop_user()}"
