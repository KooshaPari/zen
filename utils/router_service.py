"""
RouterService: LLM-assisted routing using OpenRouter (Gemini 2.5 Flash) with Redis TTL cache.

- Loads config/model_routing.yaml via utils.model_router
- Computes basic ComplexitySignals
- Optionally refines YAML-heuristic decision using OpenRouter provider with gemini-2.5-flash
- Caches decisions in Redis (CACHE DB) with short TTL
- Graceful fallback to YAML-only decision when OpenRouter unavailable or no API key
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Any

from providers.base import ProviderType
from providers.registry import ModelProviderRegistry
from utils.model_router import ComplexitySignals, get_model_router
from utils.redis_manager import RedisDB, get_redis_manager


@dataclass
class RouterInput:
    task_type: str = "quick_qa"
    prompt: str = ""
    files_count: int = 0
    images_count: int = 0
    prior_failures: int = 0
    privileged_tools: bool = False
    secrets_present: bool = False
    predicted_files_touched: int = 0
    dependency_depth: int = 0
    coordination_needed: bool = False
    allow_long_context: bool = True


class RouterService:
    def __init__(self, cache_ttl_seconds: int = 300) -> None:
        self.cache_ttl = int(os.getenv("ZEN_ROUTER_CACHE_TTL", str(cache_ttl_seconds)))
        self.redis = get_redis_manager()
        self.router = get_model_router()
        # In-memory fallback cache when Redis is unavailable
        self._mem_cache: dict[str, dict[str, Any]] = {}
        self._mem_cache_exp: dict[str, float] = {}

    def _cache_key(self, rin: RouterInput) -> str:
        # Normalize inputs into a stable key
        payload = {
            "task_type": rin.task_type,
            "prompt": rin.prompt[:4000],  # cap for key stability
            "files": rin.files_count,
            "images": rin.images_count,
            "pf": rin.prior_failures,
            "pt": rin.predicted_files_touched,
            "dd": rin.dependency_depth,
            "priv": rin.privileged_tools,
            "sec": rin.secrets_present,
            "coord": rin.coordination_needed,
            "alc": rin.allow_long_context,
            "policy_v": os.getenv("ZEN_ROUTER_POLICY_VERSION", "v1"),
        }
        h = hashlib.md5(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()  # nosec
        return f"router:decision:{h}"

    def _signals(self, rin: RouterInput) -> ComplexitySignals:
        return ComplexitySignals(
            prompt_chars=len(rin.prompt or ""),
            files_count=rin.files_count,
            images_count=rin.images_count,
            prior_failures=rin.prior_failures,
            privileged_tools=rin.privileged_tools,
            secrets_present=rin.secrets_present,
            predicted_files_touched=rin.predicted_files_touched,
            dependency_depth=rin.dependency_depth,
            coordination_needed=rin.coordination_needed,
        )

    def _get_cache(self, key: str) -> dict[str, Any] | None:
        try:
            conn = self.redis.get_connection(RedisDB.CACHE)
            raw = conn.get(key)
            if raw:
                return json.loads(raw)
        except Exception:
            # Fall back to in-memory cache
            try:
                import time
                exp = self._mem_cache_exp.get(key, 0)
                if exp and time.time() < exp:
                    return self._mem_cache.get(key)
            except Exception:
                pass
        return None

    def _set_cache(self, key: str, value: dict[str, Any]) -> None:
        try:
            conn = self.redis.get_connection(RedisDB.CACHE)
            conn.setex(key, self.cache_ttl, json.dumps(value))
        except Exception:
            # In-memory fallback
            try:
                import time
                self._mem_cache[key] = value
                self._mem_cache_exp[key] = time.time() + max(1, self.cache_ttl)
            except Exception:
                pass

    def decide(self, rin: RouterInput) -> dict[str, Any]:
        """Return routing decision with fields: chosen_model, tier, candidates, rationale, budgets, plan."""
        # 1) Cache first
        key = self._cache_key(rin)
        cached = self._get_cache(key)
        if cached:
            cached["cache_hit"] = True
            return cached

        # 2) Base decision via YAML-only router
        sig = self._signals(rin)
        base = self.router.decide(rin.task_type, signals=sig, est_tokens=int(len(rin.prompt)/4) if rin.prompt else None, allow_long_context=rin.allow_long_context)
        decision: dict[str, Any] = {
            **base,
            "cache_hit": False,
            "rationale": "yaml_heuristics",
            "plan": {
                "tool_plan": [],
                "risk_controls": [],
            },
            "budgets": {
                "max_tokens": 4096,
                "temperature": 0.3,
            },
        }

        # 3) Optional refinement via OpenRouter Gemini 2.5 Flash
        try:
            provider = ModelProviderRegistry.get_provider(ProviderType.OPENROUTER)
            if provider:
                # Ask gemini-2.5-flash to choose among candidates and set budgets
                prompt = self._build_router_prompt(rin, base)
                resp = provider.generate_content(
                    prompt=prompt,
                    model_name="gemini-2.5-flash",
                    system_prompt="You are a fast routing assistant. Output strict JSON only.",
                    temperature=0.0,
                    max_output_tokens=512,
                )
                refined = self._parse_router_json(resp.content)
                if refined and isinstance(refined, dict):
                    # Merge refined fields conservatively
                    decision["rationale"] = "gemini_2_5_flash_refinement"
                    decision["chosen_model"] = refined.get("chosen_model") or decision["chosen_model"]
                    if isinstance(refined.get("candidates"), list) and refined["candidates"]:
                        decision["candidates"] = refined["candidates"]
                    if isinstance(refined.get("budgets"), dict):
                        decision["budgets"].update(refined["budgets"])
                    if isinstance(refined.get("plan"), dict):
                        decision["plan"].update(refined["plan"])
        except Exception:
            # Fail open to YAML-only decision
            pass

        # 4) Cache and return
        self._set_cache(key, decision)
        return decision

    def _build_router_prompt(self, rin: RouterInput, base: dict[str, Any]) -> str:
        return (
            "Decide model and budgets for an AI task. Return JSON with keys: "
            "chosen_model, candidates, budgets{max_tokens,temperature}, plan{tool_plan[],risk_controls[]}.\n"
            f"Task type: {rin.task_type}\n"
            f"Prompt size: {len(rin.prompt or '')} chars\n"
            f"Base candidates: {json.dumps(base.get('candidates', []))}\n"
            f"Prefer minimal cost unless complexity demands reasoning.\n"
            f"Return only JSON.\n"
            f"User prompt (truncated):\n{(rin.prompt or '')[:1200]}"
        )

    def _parse_router_json(self, content: str) -> dict[str, Any] | None:
        try:
            # Strip code fences if present
            s = content.strip()
            if s.startswith("```") and s.endswith("```"):
                s = s.strip("`\n")
            # Find first JSON object
            start = s.find("{")
            end = s.rfind("}")
            if start != -1 and end != -1 and end > start:
                return json.loads(s[start:end+1])
        except Exception:
            return None
        return None
