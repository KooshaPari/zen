"""
Edit provider adapters for code/file modifications.

Current providers:
- builtin_fastpatch: simple, safe, instruction-driven edits (search/replace ops)
- morph_fastapply: integration stub for Morph Fast Apply (env-gated)

Select provider via env var `ZEN_EDIT_PROVIDER` (builtin|morph).
"""
from __future__ import annotations

import os
from typing import Optional

from .base import EditPlan, EditProvider
from .builtin_fastpatch import BuiltinFastPatchProvider
from .morph_fastapply import MorphFastApplyProvider


def get_edit_provider(preferred: str | None = None) -> EditProvider:
    """Return an edit provider based on env or preference.

    Order:
    - preferred value if provided
    - ZEN_EDIT_PROVIDER env (builtin|morph)
    - default to builtin
    """
    choice = (preferred or os.getenv("ZEN_EDIT_PROVIDER", "builtin")).strip().lower()
    if choice == "morph":
        return MorphFastApplyProvider()
    return BuiltinFastPatchProvider()


__all__ = [
    "EditPlan",
    "EditProvider",
    "get_edit_provider",
    "BuiltinFastPatchProvider",
    "MorphFastApplyProvider",
]

