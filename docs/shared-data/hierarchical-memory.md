# Hierarchical Memory System

## Goals
- Provide shared and private memory layers across agents, teams, and orgs
- Integrate with existing conversation_memory and agent_memory modules
- Support vector similarity, TTLs, prioritization, and scope-aware retrieval

## Memory Layers
- Short-term (per-thread; minutes)
- Working (per-task; hours)
- Long-term (per-agent; days)
- Shared (per-team/project/org)
- Procedural/Episodic/Semantic types (already defined in utils/agent_memory.py)

## Data Model
- Leverage MemoryEntry with vector and tags; add scope_context
- Keying: `{org}:{proj}:{team}:{agent}:{type}` + memory_id
- Metadata: work_dir, repo path(s), related_memories, priority, expiry

## Operations
- store_memory(agent_id, type, content, tags, priority) -> returns MemoryEntry
- retrieve_memory(scope_context, query, filters, top_k) -> semantic + tag filters
- link_memory(memory_id, related_id) to build associations
- evict_memory to enforce quotas by type and tenant

## Retrieval Strategy
- Combine:
  - Recent window for short-term/working
  - Vector similarity across long-term/shared
  - Scope filter by work_dir + tenant prefixes
  - KG-linked expansion (optional)

## Integration with Conversations
- conversation_memory: store turn summaries in short-term; thread context includes scope
- agent_memory: persist distilled insights to long-term/shared
- Provide APIs to promote/demote memories between layers based on usage and priority

## Security & Auditing
- Enforce identity and scope at API boundary
- Record access_count and last_accessed; emit memory access events

## Implementation Notes
- utils/agent_memory.py already defines vector similarity and TTLs — extend to accept scope_context and namespaces
- Use Redis (fast keys + TTL) + Postgres (durable) + vector DB for memory index
- Background jobs to summarize and compress long threads

## Roadmap
- Phase 1: Add scope_context to memory APIs; per-tenant prefixes
- Phase 2: Shared memories by team/project with moderation
- Phase 3: Automatic summarization and promotion/demotion policies

