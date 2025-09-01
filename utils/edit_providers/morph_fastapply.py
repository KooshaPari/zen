from __future__ import annotations

import json
import os
import uuid
from typing import Any

try:
    import httpx
except Exception:  # pragma: no cover
    httpx = None  # type: ignore

from .base import EditOperation, EditPlan, EditProvider


class MorphFastApplyProvider(EditProvider):
    """Adapter for Morph Fast Apply.

    Configuration (env):
    - MORPH_API_KEY: required for cloud usage
    - MORPH_BASE_URL: base URL (e.g., https://api.morphllm.com). Defaults to value if provided by client.

    Methods:
    - plan_edit: send instructions to Morph to get a concrete edit plan
    - dry_run: request diff preview from Morph (if available) or locally compose from plan
    - apply: request Morph to apply edits (cloud) or return plan for self-hosted runner

    Notes: The exact Morph API surface may differ. This adapter is intentionally
    conservative: it will raise a helpful error if not configured or if httpx is missing.
    """

    name = "morph_fastapply"

    def _client(self):
        if httpx is None:
            raise RuntimeError("httpx is required for Morph provider. Install httpx to enable.")
        key = os.getenv("MORPH_API_KEY")
        if not key:
            raise RuntimeError("MORPH_API_KEY not set. Configure environment to use Morph.")
        base = os.getenv("MORPH_BASE_URL", "https://api.morphllm.com")
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }
        return base, headers

    def plan_edit(
        self,
        *,
        instructions: str | None = None,
        operations: list[dict] | None = None,
        context: dict[str, Any] | None = None,
    ) -> EditPlan:
        base, headers = self._client()
        payload: dict[str, Any] = {
            "instructions": instructions or "",
            "operations": operations or [],
            "context": context or {},
        }
        # Endpoint path is provisional; allow override via env if needed later
        ep = os.getenv("MORPH_PLAN_ENDPOINT", f"{base}/v1/edits/plan")
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(ep, content=json.dumps(payload), headers=headers)
            if resp.status_code != 200:
                raise RuntimeError(f"Morph plan failed: HTTP {resp.status_code} {resp.text}")
            data = resp.json()

        ops: list[EditOperation] = []
        for op in data.get("operations", []):
            ops.append(EditOperation(**op))
        plan_id = data.get("plan_id") or str(uuid.uuid4())
        return EditPlan(id=plan_id, instructions=instructions, operations=ops, provider=self.name, metadata=data.get("metadata", {}))

    def dry_run(self, plan: EditPlan) -> dict:
        base, headers = self._client()
        ep = os.getenv("MORPH_DRY_RUN_ENDPOINT", f"{base}/v1/edits/dry-run")
        payload = {
            "plan_id": plan.id,
            "operations": [op.__dict__ for op in plan.operations],
        }
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(ep, content=json.dumps(payload), headers=headers)
            if resp.status_code != 200:
                raise RuntimeError(f"Morph dry-run failed: HTTP {resp.status_code} {resp.text}")
            return resp.json()

    def apply(self, plan: EditPlan) -> dict:
        base, headers = self._client()
        ep = os.getenv("MORPH_APPLY_ENDPOINT", f"{base}/v1/edits/apply")
        payload = {
            "plan_id": plan.id,
            "operations": [op.__dict__ for op in plan.operations],
        }
        with httpx.Client(timeout=60.0) as client:
            resp = client.post(ep, content=json.dumps(payload), headers=headers)
            if resp.status_code != 200:
                raise RuntimeError(f"Morph apply failed: HTTP {resp.status_code} {resp.text}")
            return resp.json()

