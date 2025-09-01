# Task Complexity Engine (MVP) — Spec v1

Goal
- Decide minimal vs reasoning LLM mode, candidate model set, token budget, and risk controls
- Drive decomposition workflows (research → plan → draft → review → apply → validate)
- Stay cheap/fast by default; escalate only when needed; be explainable

Scope (MVP)
- Heuristic scorer over structured signals; maps to tier: trivial, simple, moderate, complex, extreme
- Router hints: model class preference (minimal/reasoning), candidate task_type, est token budget
- Risk controls: require tests/review for higher tiers; sandbox/approval for privileged tools
- Extensible to ML scorer later (out-of-band training)

Signals (inputs)
- size
  - prompt_chars: int
  - artifact_volume: int (derived from attached files/chunks)
- code_impact
  - predicted_files_touched: int (from static heuristics or last run deltas)
  - dependency_depth: int (shallow≈0, medium≈1, deep≥2)
- ambiguity
  - missing_constraints: int (count of TODOs/unknowns detected by template rules)
  - files_count == 0 and prompt_chars > threshold (implicit ambiguity)
- safety
  - privileged_tools: bool (file write, git push, deployment)
  - secrets_present: bool (detected via secret scanner or inputs)
  - prod_target: bool
- history
  - prior_failures: int (related tasks/batches)
  - flakiness_score: int (0–3, optional)
- coordination
  - delegation_required: bool (needs parent→child or peer↔peer)
  - cross_agent_threads: bool (referenced channels/DMs)

Weights (config)
- Default weights in config/model_routing.yaml → complexity_engine.weights
  - size: 2, code_impact: 3, ambiguity: 3, safety: 4, history: 2, coordination: 2 (example)

Scoring (MVP)
- For each dimension add weighted points when thresholds crossed, e.g.:
  - size: +1 if prompt>2k chars; +1 if >8k
  - code_impact: +1 if files_touched≥2; +2 if ≥5
  - ambiguity: +1 if files_count==0 and prompt>1200; +1 per missing_constraint (cap 2)
  - safety: +1 if privileged_tools or secrets_present; +1 if prod_target
  - history: +1 if prior_failures>0; +flakiness_score (cap 2)
  - coordination: +1 if delegation_required or cross_agent_threads

Tier thresholds (config)
- From config/model_routing.yaml → complexity_engine.thresholds
  - trivial: 0–4, simple: 5–8, moderate: 9–13, complex: 14–18, extreme: 19–100 (defaults)

Outputs
- tier: trivial|simple|moderate|complex|extreme
- router_hints
  - class_preference: minimal | minimal_or_reasoning_when_ambiguity_high | reasoning | reasoning_with_guardrails
  - task_type: quick_qa | complex_reasoning | code_edit | long_context | planning | research_reasoning | code_generation
  - est_tokens: int (prompt_chars/4 + context padding)
- risk_controls
  - require_tests: bool (moderate+)
  - require_review: bool (complex+)
  - sandbox_only: bool when privileged_tools or prod_target
  - approvals_required: bool (extreme or privileged)

Routing policy interaction
- The ModelRouter selects candidates based on task_type defaults (config), filters by context fit and disabled models, returns provider-native model_name
- Complexity tier influences task_type and class_preference but does not strictly override task_type defaults in MVP (advisory bias)

Governance
- Intent allowlists by tier (e.g., no deploy on moderate without approval)
- Budget caps per tier (config.policy.budget) applied at selection time
- Observability: explainability blob with signals, weights, score, tier, router decision rationale

Explainability (emitted with task)
- signals: {...}
- score_breakdown: [ {dimension, weight, contribution, reason}, ... ]
- tier and thresholds
- selected_model and why (context fit, cost band, availability)
- risk_controls applied

Integration points
- utils/model_router.py: ComplexityEngine + ModelRouter already provide MVP hooks
- BaseTool: when model auto/unavailable and ZEN_ROUTER_ENABLE=1 → build signals (prompt length, files/images count) and let router pick; later expand with more signals
- Legacy: `server_http.handle_llm_task_create` described direct LLM tasks. Use MCP `tools/call` for equivalent flows.

Future (v2+)
- ML Scorer: learn weights from outcomes; per-domain profiles
- Better ambiguity detector: pattern rules + small classifier
- Code impact predictor: static analysis + dependency graph
- Safety: secrets/prod detection wired to scanners/policies
- Coordination: integrate project graph (threads/channels/DMs) and delegation plan
- Dynamic token budgeting with retrieval packer and reranker feedback

Acceptance criteria
- Tiering is stable and explainable on synthetic scenarios
- Router decisions align with tier expectations and context limits
- Risk controls enforced by tier and intent policies
- No regressions when ZEN_ROUTER_ENABLE=0 (feature-flag safety)
