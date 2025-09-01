# Messaging, Project Graph, and Prompt Injection — Spec v1

Goals
- Enable peer + hierarchical collaboration between agents across devices and projects
- Provide Slack/Discord-like messaging: channels, threads, DMs; blocking/non-blocking behavior
- Productize via a Project Graph (Projects, Agents, Tasks, Batches, Artifacts, Threads)
- Safely inject message summaries and relevant artifacts into LLM prompts within tight token budgets

Scope (MVP)
- Redis-backed graph and messaging stores; NATS subjects for events
- Minimal tools/APIs to create channels/threads/DMs and send/receive messages
- Blocking/non-blocking semantics with inbox handling and prompt injection rules
- A2A routing via NATS (delegate/request_tool/status_update) with message linkage

Entities & Keys (Redis)
- Project
  - H: project:<id> {name, owner, description, created_at, schema_version}
  - S: project:<id>:agents → agent_ids
  - S: project:<id>:channels → channel_ids
  - S: project:<id>:artifacts → artifact_ids
- Agent
  - H: agent:<id> {name, endpoint_url, capabilities, owner, project_id}
  - S: agent:<id>:subscriptions → channel_ids
- Task/Batch
  - H: task:<id>, batch:<id> (already exist)
  - S: batch:<id>:tasks → task_ids (exists)
- Channel/Thread/DM
  - H: channel:<id> {project_id, name, topic, visibility, created_by, created_at}
  - H: thread:<id> {channel_id, root_message_id, created_by, created_at}
  - H: dm:<id> {participants: json, created_at}
  - L: channel:<id>:messages (list of message_ids, newest at head, capped N)
  - L: thread:<id>:messages (list)
  - L: dm:<id>:messages (list)
- Message
  - H: message:<id> {from, to, type, body, mentions, artifacts, created_at, importance, blocking}
  - Optional: message:<id>:meta {correlation_id, task_id, batch_id, project_id}
- Indexes
  - Z: inbox:agent:<id>:unread → score=created_at, members=message_ids
  - Z: inbox:channel:<id>:recent → for list views

Retention
- TTLs on message lists (e.g., keep last 1–5k per channel) and on message hashes based on project policy
- Long-lived artifacts kept separately with their own TTLs/persistence

NATS Subjects (Events)
- messaging.created, messaging.posted, messaging.read, messaging.thread_created
- channels.created, channels.updated, channels.archived
- a2a.* (separate spec) with message linkage via correlation_id

Messaging Tools/APIs (MVP)
- create_channel(project_id, name, visibility="project|private") → channel_id
- post_message(project_id, channel_id, body, mentions=[], artifacts=[], blocking=false, importance="normal|high|critical") → message_id
- reply_thread(project_id, thread_id, body, ...)
- dm_send(participants=[agent_ids], body, blocking=false)
- channel_subscribe(agent_id, channel_id)
- channel_history(channel_id, limit, since)
- dm_list(agent_id, limit)
- message_mark_read(agent_id, message_id)

Blocking vs Non-blocking
- Blocking message:
  - Injection: “You received a message from X …” + summary, pause execution, resume with context
  - Recording: add to task timeline and message’s meta; mark as handled when resumed
- Non-blocking message:
  - Enqueue to inbox ZSET; show unread count in UX; end-of-run nudge suggests review

Prompt Injection Policy
- Budget: configurable max tokens for messaging injection (e.g., 512–1024)
- Selection:
  - Prioritize critical and high-importance messages; include top unread thread summaries
  - Respect mentions; include latest in subscribed channels that reference current task/project
- Summarization:
  - Use fast model (e.g., gemini-2.5-flash) to compress threads to fit
  - Include "handles" (tool calls) for fetching more when needed
- Safety:
  - Redact secrets/PII via rules; strip long attachments; cap injected links

Project Graph Context Pack
- Build per invocation:
  - Parents/peers (project graph), active threads involving agent/task, relevant artifacts
  - Rerank code/doc snippets (Morph reranker where available; OSS fallback) and dedupe history
  - Strict token budgeting (router provides est_tokens)

A2A Integration
- Link messages to A2A envelope via correlation_id and context (task_id, batch_id, project_id)
- Delegation flows add auto-updates into a channel/thread dedicated to the task
- request_tool intent can route through messaging when human-in-the-loop required

APIs (HTTP) — Suggested
- POST /projects, GET /projects/{id}
- POST /channels, GET /channels/{id}, GET /projects/{id}/channels
- POST /threads, GET /threads/{id}
- POST /messages, GET /channels/{id}/messages, GET /threads/{id}/messages
- POST /dms, GET /dms/{id}/messages, GET /agents/{id}/inbox
- POST /messages/{id}/read

Minimal Data Contracts (JSON)
- Message { id, from, to?, type: channel|thread|dm, body, mentions[], artifacts[], blocking, importance, created_at, context: {project_id, task_id?, batch_id?}, correlation_id? }
- Channel { id, project_id, name, topic?, visibility, created_by, created_at }
- Thread { id, channel_id, root_message_id, created_by, created_at }
- DM { id, participants[], created_at }

Security & Governance
- Auth: Bearer/mTLS; per-agent JWT scopes for messaging
- ACLs: channel visibility and membership; DM participants only
- Rate limits per agent for spam control; max message size
- Audit trail to Redis/NATS; optional signatures later

Observability
- Log structured events: who, where, importance, blocking, size
- Metrics: messages posted/read, unread counts, time-to-read, injection tokens used
- Tracing: correlation_id spans from A2A → messaging → task execution

MVP Acceptance Criteria
- Create channels/threads/DMs; send/receive; unread counts; mark-read
- Blocking messages pause and resume execution with context injection
- Non-blocking messages show up as end-of-run prompts; tool to fetch details
- NATS events are published for messaging lifecycle; Redis indexes support list views

Roadmap
- v1: Redis-only storage + NATS events; tools in Python; UI hooks via SSE endpoints
- v2: Web UI for channels/threads; richer summarization pipelines; per-project policies
- v3: Graph DB option (Neo4j) for large projects; ML prioritization for inbox; cross-tenant channels

