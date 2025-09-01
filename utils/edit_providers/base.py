from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class EditOperation:
    """A single edit operation.

    For builtin provider, supported op types:
    - replace: {filepath, find, replace, count(optional)}
    - insert: {filepath, anchor, content, position: before|after}
    - write:  {filepath, content}  (overwrite)
    """

    type: str
    filepath: str
    find: str | None = None
    replace: str | None = None
    count: int | None = None
    anchor: str | None = None
    content: str | None = None
    position: str | None = None  # before|after


@dataclass
class EditPlan:
    """High-level edit plan.

    Providers return a plan that can be previewed (diff) or applied.
    """

    id: str
    instructions: str | None
    operations: list[EditOperation]
    provider: str
    metadata: dict[str, Any]


class EditProvider:
    """Interface for edit providers."""

    name = "base"

    def plan_edit(self, *, instructions: str | None = None, operations: list[dict] | None = None,
                  context: dict[str, Any] | None = None) -> EditPlan:
        raise NotImplementedError

    def dry_run(self, plan: EditPlan) -> dict:
        """Return a preview (diffs per file) without touching disk."""
        raise NotImplementedError

    def apply(self, plan: EditPlan) -> dict:
        """Apply the plan to disk and return results (files changed, counts)."""
        raise NotImplementedError

