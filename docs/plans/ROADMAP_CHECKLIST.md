# Roadmap Checklist (Owners, Files, Status)

Legend: Status → Deferred | Partially Available | Implemented

LLM Architecture
- LLM tool-calling loop (LLM as MCP client)
  - Owner: LLM/Router
  - Files: `tools/universal_executor.py`, `server_mcp_http.py`, `utils/streaming_protocol.py`
  - Status: Deferred
- ReAct agent pattern
  - Owner: Orchestration
  - Files: `workflows/multi_agent_workflow.py`, `tools/planner.py`, `tools/tracer.py`
  - Status: Deferred
- Stream-interrupt tool calls during LLM streaming
  - Owner: LLM/Router
  - Files: `utils/streaming_protocol.py`, `server_mcp_http.py`
  - Status: Deferred
- Framework adapters (LangChain, LangGraph)
  - Owner: Orchestration
  - Files: `workflows/*`, `utils/router_service.py`
  - Status: Deferred
- Advanced multimodal (audio/video)
  - Owner: Providers
  - Files: Providers layer, `utils/model_context.py`
  - Status: Deferred
- Distributed orchestrator (cross-service)
  - Owner: Orchestration/Infra
  - Files: `utils/a2a_protocol.py`, `utils/nats_*`, `utils/kafka_events.py`
  - Status: Deferred
- LLM batch/async endpoints
  - Owner: LLM/Router
  - Files: `tools/universal_executor.py` (present), dedicated LLM HTTP handlers
  - Status: Partially Available

Communication Protocol
- Full tag taxonomy + transforms
  - Owner: Protocol
  - Files: `utils/agent_prompts.py`, `/ENHANCED_COMMUNICATION_PROTOCOL.md`
  - Status: Partially Available
- Resource-aware interrupts (token/time)
  - Owner: Protocol/Runtime
  - Files: `utils/streaming_monitor.py`, `utils/token_budget_manager.py`
  - Status: Deferred
- Provenance/compliance signatures
  - Owner: Protocol/Compliance
  - Files: `utils/audit_trail.py`, `utils/kafka_events.py`
  - Status: Deferred

Integration Architecture
- NATS edge/leaf nodes + NEX
  - Owner: Infra
  - Files: `utils/nats_config.py`, `utils/nats_streaming.py`, `utils/nats_communicator.py`
  - Status: Deferred
- Service-mesh discovery via NATS req/reply
  - Owner: Infra/Orchestration
  - Files: `utils/nats_*`
  - Status: Partially Available
- A2A cross-org registry + delegation
  - Owner: Orchestration
  - Files: `utils/a2a_protocol.py`
  - Status: Partially Available
- Temporal workflows
  - Owner: Orchestration
  - Files: `utils/temporal_client.py`, `workflows/*`
  - Status: Deferred
- PostgreSQL-first persistence
  - Owner: Storage
  - Files: New DB layer, migrations
  - Status: Deferred
- Redis time-series metrics (TS)
  - Owner: Observability
  - Files: Redis TS integration, dashboard
  - Status: Deferred
- Unified Kafka/NATS/RabbitMQ bridge
  - Owner: Infra
  - Files: `utils/kafka_events.py`, `utils/nats_*`, `utils/rabbitmq_queue.py`
  - Status: Partially Available

