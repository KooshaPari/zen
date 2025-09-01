"""
Model Router and Task Complexity Engine (MVP)

- Loads config/model_routing.yaml (path override via ZEN_MODEL_ROUTING_CONFIG)
- Provides a minimal ComplexityEngine (heuristic scoring → tier)
- Provides ModelRouter.select_model(...) with task-type defaults and policy filters
- Returns provider-native model_name (e.g., "gemini-2.5-flash")

Opt-in usage: Only applies when callers explicitly opt-in (e.g., model == "auto")
and ZEN_ROUTER_ENABLE=1. Non-invasive to existing flows.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

import yaml

logger = logging.getLogger(__name__)

_CONFIG_CACHE: dict[str, Any] | None = None
_ROUTER_CACHE: ModelRouter | None = None


DEFAULT_CONFIG_PATHS = [
    os.getenv("ZEN_MODEL_ROUTING_CONFIG"),
    os.path.join("config", "model_routing.yaml"),
    os.path.join("config", "model_routing.yml"),
]


@dataclass
class ComplexitySignals:
    prompt_chars: int = 0
    files_count: int = 0
    images_count: int = 0
    prior_failures: int = 0
    privileged_tools: bool = False
    secrets_present: bool = False
    predicted_files_touched: int = 0
    dependency_depth: int = 0
    coordination_needed: bool = False  # cross-agent, threads, delegation


class ComplexityEngine:
    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg.get("complexity_engine", {})
        self.weights = self.cfg.get("weights", {})
        self.thresholds = self.cfg.get("thresholds", {})

    def score(self, s: ComplexitySignals) -> tuple[str, int]:
        # Simple heuristic score
        def w(k):
            return int(self.weights.get(k, 1))
        score = 0
        score += w("size") * (1 if s.prompt_chars > 2000 else 0)
        score += w("size") * (1 if s.prompt_chars > 8000 else 0)
        score += w("code_impact") * (1 if s.predicted_files_touched >= 2 else 0)
        score += w("code_impact") * (2 if s.predicted_files_touched >= 5 else 0)
        score += w("ambiguity") * (1 if s.files_count == 0 and s.prompt_chars > 1200 else 0)
        score += w("safety") * (1 if s.privileged_tools or s.secrets_present else 0)
        score += w("history") * (1 if s.prior_failures > 0 else 0)
        score += w("coordination") * (1 if s.coordination_needed else 0)

        # Map to tier
        t = self.thresholds or {
            "trivial": "0-4",
            "simple": "5-8",
            "moderate": "9-13",
            "complex": "14-18",
            "extreme": "19-100",
        }
        tier = "simple"
        for name, band in t.items():
            low, high = [int(x) for x in str(band).split("-")]
            if low <= score <= high:
                tier = name
                break
        return tier, score


class ModelRouter:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.models = {m["key"]: m for m in config.get("models", [])}
        self.disabled = set((config.get("policies", {}) or {}).get("disabled_models", []) or [])
        self.routing = config.get("routing", {})
        self.caching = config.get("caching", {})
        self.policies = config.get("policies", {})
        self.engine = ComplexityEngine(config)

    def _key_to_model_name(self, key: str) -> str | None:
        # config entry provides model_id (provider-native name)
        model = self.models.get(key)
        if not model:
            return None
        return model.get("model_id") or key.split(":")[-1]

    def _filter_disabled(self, keys: list[str]) -> list[str]:
        out = []
        for k in keys:
            if k in self.disabled:
                continue
            # also filter by model_id match if listed in disabled
            model_name = self._key_to_model_name(k) or ""
            if any(model_name == (self.models.get(dk, {}) or {}).get("model_id") for dk in self.disabled):
                continue
            out.append(k)
        return out

    def _resolve_candidates(self, task_type: str) -> list[str]:
        defaults = (self.routing.get("defaults_by_task_type") or {}).get(task_type, [])
        return self._filter_disabled(defaults)

    def _ensure_context_fit(self, key: str, est_tokens: int | None) -> bool:
        if est_tokens is None:
            return True
        m = self.models.get(key)
        if not m:
            return True
        ctx = int(m.get("context_window", 0) or 0)
        return (ctx == 0) or (est_tokens <= ctx)

    def decide(
        self,
        task_type: str,
        signals: ComplexitySignals | None = None,
        est_tokens: int | None = None,
        allow_long_context: bool = True,
    ) -> dict[str, Any]:
        """Return a structured routing decision including chosen model and rationale."""
        signals = signals or ComplexitySignals()
        tier, score = self.engine.score(signals)

        candidates = self._resolve_candidates(task_type)
        if not candidates:
            logger.debug(f"No candidates for task_type={task_type}, falling back to quick_qa")
            candidates = self._resolve_candidates("quick_qa")

        ordered = []
        for k in candidates:
            if not allow_long_context and "long_context" in (self.models.get(k, {}).get("tags") or []):
                continue
            if not self._ensure_context_fit(k, est_tokens):
                continue
            ordered.append(k)

        if not ordered:
            long_ctx = self._resolve_candidates("long_context")
            ordered = [k for k in long_ctx if self._ensure_context_fit(k, est_tokens)] or candidates

        ordered_models = [self._key_to_model_name(k) for k in ordered if self._key_to_model_name(k)]
        chosen = ordered_models[0] if ordered_models else "gemini-2.5-flash"
        return {
            "chosen_model": chosen,
            "tier": tier,
            "score": score,
            "candidates": ordered_models,
            "ordered_keys": ordered,
        }

    def select_model(
        self,
        task_type: str,
        signals: ComplexitySignals | None = None,
        est_tokens: int | None = None,
        allow_long_context: bool = True,
    ) -> str:
        """Return provider-native model_name (e.g., "gemini-2.5-flash")."""
        decision = self.decide(task_type, signals=signals, est_tokens=est_tokens, allow_long_context=allow_long_context)
        return decision.get("chosen_model", "gemini-2.5-flash")


def _load_config() -> dict[str, Any]:
    global _CONFIG_CACHE
    if _CONFIG_CACHE is not None:
        return _CONFIG_CACHE
    for p in DEFAULT_CONFIG_PATHS:
        if not p:
            continue
        if os.path.exists(p):
            try:
                with open(p) as f:
                    _CONFIG_CACHE = yaml.safe_load(f) or {}
                    logger.info(f"Model router config loaded from {p}")
                    return _CONFIG_CACHE
            except Exception as e:
                logger.warning(f"Failed to load model routing config from {p}: {e}")
    _CONFIG_CACHE = {}
    return _CONFIG_CACHE


def get_model_router(force_reload: bool = False) -> ModelRouter:
    global _ROUTER_CACHE, _CONFIG_CACHE
    if force_reload or _ROUTER_CACHE is None:
        if force_reload:
            _CONFIG_CACHE = None
        cfg = _load_config()
        _ROUTER_CACHE = ModelRouter(cfg)
    return _ROUTER_CACHE

