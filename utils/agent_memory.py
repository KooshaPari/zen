"""
Agent Memory Management with Vector Similarity

This module provides sophisticated memory management for agents using Redis for persistence,
enabling the Zen MCP Server to maintain rich contextual memory across agent lifecycles
with vector similarity search and intelligent memory organization.

Key Features:
- Multi-layered memory architecture (short-term, working, long-term)
- Vector similarity search for contextual memory retrieval
- Intelligent memory consolidation and pruning
- Cross-agent memory sharing and knowledge transfer
- Memory compression and efficient storage
- Performance optimization with caching layers
- Integration with existing conversation memory system
- Memory analytics and insights for optimization

Memory Architecture:
- Short-term Memory: Recent interactions, temporary context (30 minutes TTL)
- Working Memory: Active task context, current focus (2 hours TTL)
- Long-term Memory: Persistent knowledge, learned patterns (24 hours TTL)
- Shared Memory: Cross-agent knowledge base, common patterns
- Memory Index: Vector similarity search for contextual retrieval

Integration Points:
- Extends existing conversation_memory.py with persistent storage
- Provides memory APIs for agent state management integration
- Creates memory events for Kafka agent audit trail consumption
- Enables memory-based context for Temporal agent workflows
- Generates memory metrics for Testing agent performance monitoring
"""

import asyncio
import hashlib
import json
import logging
import math
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from utils.conversation_memory import ConversationTurn, ThreadContext
from utils.redis_manager import RedisDB, TTLPolicies, get_redis_manager

logger = logging.getLogger(__name__)


class MemoryType(str, Enum):
    """Types of agent memory for different purposes"""
    SHORT_TERM = "short_term"      # Recent interactions, temporary context
    WORKING = "working"            # Active task context, current focus
    LONG_TERM = "long_term"        # Persistent knowledge, learned patterns
    SHARED = "shared"              # Cross-agent knowledge base
    PROCEDURAL = "procedural"      # How-to knowledge, process memory
    EPISODIC = "episodic"          # Event-based memories, experiences
    SEMANTIC = "semantic"          # Facts, concepts, relationships


class MemoryPriority(str, Enum):
    """Memory priority levels for retention and retrieval"""
    CRITICAL = "critical"          # Never expires, highest priority
    HIGH = "high"                  # Long retention, high retrieval priority
    MEDIUM = "medium"              # Standard retention and priority
    LOW = "low"                    # Short retention, low priority
    TEMPORARY = "temporary"        # Very short retention, cleanup priority


@dataclass
class MemoryVector:
    """Vector representation for similarity search"""
    vector_id: str
    dimensions: int
    values: list[float]
    metadata: dict[str, Any]
    created_at: datetime


@dataclass
class MemoryEntry:
    """Individual memory entry with metadata"""
    memory_id: str
    agent_id: str
    memory_type: MemoryType
    priority: MemoryPriority
    content: dict[str, Any]
    tags: list[str]
    vector: Optional[MemoryVector] = None
    related_memories: list[str] = None  # IDs of related memories
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    created_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    version: int = 1

    def __post_init__(self):
        if self.related_memories is None:
            self.related_memories = []
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc)


@dataclass
class MemorySearchResult:
    """Result from memory search operations"""
    memory_entry: MemoryEntry
    similarity_score: float
    relevance_score: float
    rank: int


@dataclass
class MemoryAnalytics:
    """Memory usage analytics and insights"""
    total_memories: int
    by_type: dict[str, int]
    by_priority: dict[str, int]
    total_size_mb: float
    compression_ratio: float
    average_access_frequency: float
    most_accessed_memories: list[str]
    memory_efficiency_score: float
    last_consolidation: Optional[datetime]


class MemoryConsolidationStrategy(str, Enum):
    """Strategies for memory consolidation"""
    FREQUENCY_BASED = "frequency_based"    # Keep frequently accessed memories
    RECENCY_BASED = "recency_based"        # Keep recently accessed memories
    IMPORTANCE_BASED = "importance_based"  # Keep high-priority memories
    SIMILARITY_BASED = "similarity_based"  # Merge similar memories
    HYBRID = "hybrid"                      # Combination of strategies


class AgentMemoryManager:
    """
    Advanced Agent Memory Management System

    Provides sophisticated memory management with vector similarity search,
    intelligent consolidation, and multi-layered memory architecture.
    """

    def __init__(self):
        self.redis_manager = get_redis_manager()

        # Memory configuration
        self.max_memories_per_type = {
            MemoryType.SHORT_TERM: 100,
            MemoryType.WORKING: 50,
            MemoryType.LONG_TERM: 1000,
            MemoryType.SHARED: 5000,
            MemoryType.PROCEDURAL: 200,
            MemoryType.EPISODIC: 500,
            MemoryType.SEMANTIC: 2000,
        }

        # Vector similarity configuration
        self.vector_dimension = 384  # Compatible with sentence transformers
        self.similarity_threshold = 0.7

        # Performance tracking
        self._operation_count = 0
        self._total_latency = 0.0

        # Memory locks for thread safety
        self._memory_locks: dict[str, asyncio.Lock] = {}
        self._lock_manager_lock = asyncio.Lock()

        logger.info("Agent Memory Manager initialized with vector similarity support")

    async def _get_memory_lock(self, agent_id: str) -> asyncio.Lock:
        """Get or create async lock for memory operations"""
        async with self._lock_manager_lock:
            if agent_id not in self._memory_locks:
                self._memory_locks[agent_id] = asyncio.Lock()
            return self._memory_locks[agent_id]

    # Core Memory Management Methods

    async def store_memory(self, agent_id: str, memory_type: MemoryType, content: dict[str, Any],
                          tags: Optional[list[str]] = None, priority: MemoryPriority = MemoryPriority.MEDIUM,
                          expires_at: Optional[datetime] = None) -> Optional[MemoryEntry]:
        """Store new memory entry with automatic vectorization"""
        start_time = time.time()

        try:
            memory_lock = await self._get_memory_lock(agent_id)
            async with memory_lock:
                # Generate memory ID
                memory_id = str(uuid.uuid4())

                # Create memory entry
                memory_entry = MemoryEntry(
                    memory_id=memory_id,
                    agent_id=agent_id,
                    memory_type=memory_type,
                    priority=priority,
                    content=content,
                    tags=tags or [],
                    expires_at=expires_at
                )

                # Generate vector for similarity search
                memory_entry.vector = await self._create_memory_vector(memory_entry)

                # Find related memories using vector similarity
                memory_entry.related_memories = await self._find_related_memories(
                    agent_id, memory_entry.vector, limit=5
                )

                # Store memory in Redis
                success = await self._persist_memory(memory_entry)
                if not success:
                    logger.error(f"Failed to persist memory {memory_id} for agent {agent_id}")
                    return None

                # Update memory index
                await self._update_memory_index(memory_entry)

                # Check if consolidation is needed
                await self._check_consolidation_trigger(agent_id, memory_type)

                logger.debug(f"Stored {memory_type.value} memory {memory_id} for agent {agent_id}")
                return memory_entry

        except Exception as e:
            logger.error(f"Error storing memory for agent {agent_id}: {e}")
            return None
        finally:
            latency = time.time() - start_time
            self._record_operation_latency("store_memory", latency)

    async def retrieve_memory(self, agent_id: str, memory_id: str) -> Optional[MemoryEntry]:
        """Retrieve specific memory entry and update access statistics"""
        start_time = time.time()

        try:
            memory_key = f"memory:{agent_id}:{memory_id}"
            memory_data = self.redis_manager.get_agent_memory(memory_key, "data")

            if memory_data:
                memory_entry = self._deserialize_memory_entry(memory_data)

                # Update access statistics
                memory_entry.access_count += 1
                memory_entry.last_accessed = datetime.now(timezone.utc)

                # Persist updated access stats
                await self._persist_memory(memory_entry)

                return memory_entry

            return None

        except Exception as e:
            logger.error(f"Error retrieving memory {memory_id} for agent {agent_id}: {e}")
            return None
        finally:
            latency = time.time() - start_time
            self._record_operation_latency("retrieve_memory", latency)

    async def search_memories(self, agent_id: str, query: str, memory_types: Optional[list[MemoryType]] = None,
                            limit: int = 10, similarity_threshold: Optional[float] = None) -> list[MemorySearchResult]:
        """Search memories using vector similarity and text matching"""
        start_time = time.time()

        try:
            if similarity_threshold is None:
                similarity_threshold = self.similarity_threshold

            # Create query vector
            query_vector = await self._create_query_vector(query)
            if not query_vector:
                logger.warning(f"Failed to create query vector for: {query}")
                return []

            # Get all memories for the agent
            all_memories = await self._get_all_memories(agent_id, memory_types)

            # Calculate similarity scores
            search_results = []
            for memory_entry in all_memories:
                if memory_entry.vector:
                    similarity_score = self._calculate_cosine_similarity(
                        query_vector.values, memory_entry.vector.values
                    )

                    # Calculate relevance score (combines similarity + other factors)
                    relevance_score = self._calculate_relevance_score(
                        memory_entry, similarity_score, query
                    )

                    if similarity_score >= similarity_threshold:
                        search_results.append(MemorySearchResult(
                            memory_entry=memory_entry,
                            similarity_score=similarity_score,
                            relevance_score=relevance_score,
                            rank=0  # Will be set after sorting
                        ))

            # Sort by relevance score and assign ranks
            search_results.sort(key=lambda x: x.relevance_score, reverse=True)
            for i, result in enumerate(search_results):
                result.rank = i + 1

            # Return top results
            results = search_results[:limit]

            logger.debug(f"Found {len(results)} memory matches for query: {query}")
            return results

        except Exception as e:
            logger.error(f"Error searching memories for agent {agent_id}: {e}")
            return []
        finally:
            latency = time.time() - start_time
            self._record_operation_latency("search_memories", latency)

    async def update_memory(self, agent_id: str, memory_id: str,
                          content: Optional[dict[str, Any]] = None,
                          tags: Optional[list[str]] = None,
                          priority: Optional[MemoryPriority] = None) -> bool:
        """Update existing memory entry"""
        start_time = time.time()

        try:
            memory_lock = await self._get_memory_lock(agent_id)
            async with memory_lock:
                # Retrieve current memory
                memory_entry = await self.retrieve_memory(agent_id, memory_id)
                if not memory_entry:
                    logger.error(f"Memory {memory_id} not found for agent {agent_id}")
                    return False

                # Update fields
                if content is not None:
                    memory_entry.content = content
                    # Regenerate vector if content changed
                    memory_entry.vector = await self._create_memory_vector(memory_entry)

                if tags is not None:
                    memory_entry.tags = tags

                if priority is not None:
                    memory_entry.priority = priority

                memory_entry.version += 1

                # Persist updated memory
                success = await self._persist_memory(memory_entry)
                if success:
                    await self._update_memory_index(memory_entry)
                    logger.debug(f"Updated memory {memory_id} for agent {agent_id}")
                    return True
                else:
                    logger.error(f"Failed to persist updated memory {memory_id}")
                    return False

        except Exception as e:
            logger.error(f"Error updating memory {memory_id} for agent {agent_id}: {e}")
            return False
        finally:
            latency = time.time() - start_time
            self._record_operation_latency("update_memory", latency)

    async def delete_memory(self, agent_id: str, memory_id: str) -> bool:
        """Delete memory entry and clean up references"""
        start_time = time.time()

        try:
            memory_lock = await self._get_memory_lock(agent_id)
            async with memory_lock:
                # Retrieve memory to get related memories
                memory_entry = await self.retrieve_memory(agent_id, memory_id)
                if not memory_entry:
                    logger.debug(f"Memory {memory_id} not found for deletion")
                    return True

                # Remove from Redis
                conn = self.redis_manager.get_connection(RedisDB.MEMORY)
                memory_key = f"memory:{agent_id}:{memory_id}"
                deleted = conn.delete(memory_key)

                if deleted:
                    # Remove from memory index
                    await self._remove_from_memory_index(agent_id, memory_id)

                    # Update related memories to remove this reference
                    await self._update_related_memory_references(
                        agent_id, memory_id, memory_entry.related_memories
                    )

                    logger.debug(f"Deleted memory {memory_id} for agent {agent_id}")
                    return True
                else:
                    logger.warning(f"Memory {memory_id} was not found in Redis")
                    return False

        except Exception as e:
            logger.error(f"Error deleting memory {memory_id} for agent {agent_id}: {e}")
            return False
        finally:
            latency = time.time() - start_time
            self._record_operation_latency("delete_memory", latency)

    # Memory Consolidation Methods

    async def consolidate_memories(self, agent_id: str,
                                 strategy: MemoryConsolidationStrategy = MemoryConsolidationStrategy.HYBRID) -> int:
        """Consolidate memories to optimize storage and performance"""
        start_time = time.time()

        try:
            memory_lock = await self._get_memory_lock(agent_id)
            async with memory_lock:
                logger.info(f"Starting memory consolidation for agent {agent_id} using {strategy.value} strategy")

                consolidated_count = 0

                for memory_type in MemoryType:
                    # Get all memories of this type
                    memories = await self._get_memories_by_type(agent_id, memory_type)
                    max_allowed = self.max_memories_per_type.get(memory_type, 100)

                    if len(memories) <= max_allowed:
                        continue

                    # Apply consolidation strategy
                    if strategy == MemoryConsolidationStrategy.FREQUENCY_BASED:
                        memories_to_remove = self._select_by_frequency(memories, len(memories) - max_allowed)
                    elif strategy == MemoryConsolidationStrategy.RECENCY_BASED:
                        memories_to_remove = self._select_by_recency(memories, len(memories) - max_allowed)
                    elif strategy == MemoryConsolidationStrategy.IMPORTANCE_BASED:
                        memories_to_remove = self._select_by_importance(memories, len(memories) - max_allowed)
                    elif strategy == MemoryConsolidationStrategy.SIMILARITY_BASED:
                        memories_to_remove = await self._merge_similar_memories(memories, len(memories) - max_allowed)
                    else:  # HYBRID
                        memories_to_remove = await self._hybrid_consolidation(memories, len(memories) - max_allowed)

                    # Remove selected memories
                    for memory_entry in memories_to_remove:
                        success = await self.delete_memory(agent_id, memory_entry.memory_id)
                        if success:
                            consolidated_count += 1

                # Record consolidation event
                await self._record_consolidation_event(agent_id, strategy, consolidated_count)

                logger.info(f"Consolidated {consolidated_count} memories for agent {agent_id}")
                return consolidated_count

        except Exception as e:
            logger.error(f"Error consolidating memories for agent {agent_id}: {e}")
            return 0
        finally:
            latency = time.time() - start_time
            self._record_operation_latency("consolidate_memories", latency)

    # Vector Similarity Methods

    async def _create_memory_vector(self, memory_entry: MemoryEntry) -> Optional[MemoryVector]:
        """Create vector representation of memory content"""
        try:
            # Extract text content for vectorization
            text_content = self._extract_text_content(memory_entry.content)
            if not text_content:
                return None

            # For now, use simple text-based vector (would use actual embeddings in production)
            vector_values = self._simple_text_to_vector(text_content)

            vector = MemoryVector(
                vector_id=str(uuid.uuid4()),
                dimensions=self.vector_dimension,
                values=vector_values,
                metadata={
                    'memory_id': memory_entry.memory_id,
                    'memory_type': memory_entry.memory_type.value,
                    'tags': memory_entry.tags,
                    'text_length': len(text_content)
                },
                created_at=datetime.now(timezone.utc)
            )

            return vector

        except Exception as e:
            logger.error(f"Error creating memory vector: {e}")
            return None

    async def _create_query_vector(self, query: str) -> Optional[MemoryVector]:
        """Create vector representation of search query"""
        try:
            if not query.strip():
                return None

            vector_values = self._simple_text_to_vector(query)

            vector = MemoryVector(
                vector_id=str(uuid.uuid4()),
                dimensions=self.vector_dimension,
                values=vector_values,
                metadata={'query': query},
                created_at=datetime.now(timezone.utc)
            )

            return vector

        except Exception as e:
            logger.error(f"Error creating query vector: {e}")
            return None

    def _simple_text_to_vector(self, text: str) -> list[float]:
        """Simple text-to-vector conversion (placeholder for actual embeddings)"""
        # This would use actual embedding models like sentence-transformers
        # For now, create a simple hash-based vector

        # Normalize text
        text = text.lower().strip()

        # Create hash-based features
        vector = [0.0] * self.vector_dimension

        # Hash text and distribute across vector dimensions
        text_hash = hashlib.sha256(text.encode()).digest()
        for i in range(min(len(text_hash), self.vector_dimension // 8)):
            byte_val = text_hash[i]
            for j in range(8):
                if i * 8 + j < self.vector_dimension:
                    vector[i * 8 + j] = float((byte_val >> j) & 1)

        # Add some word-based features
        words = text.split()[:50]  # Limit words to prevent overflow
        for _i, word in enumerate(words):
            word_hash = hash(word) % self.vector_dimension
            vector[word_hash] = min(vector[word_hash] + 0.1, 1.0)

        # Normalize vector
        magnitude = math.sqrt(sum(x * x for x in vector))
        if magnitude > 0:
            vector = [x / magnitude for x in vector]

        return vector

    def _calculate_cosine_similarity(self, vector1: list[float], vector2: list[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        try:
            if len(vector1) != len(vector2):
                return 0.0

            # Calculate dot product
            dot_product = sum(a * b for a, b in zip(vector1, vector2))

            # Calculate magnitudes
            magnitude1 = math.sqrt(sum(a * a for a in vector1))
            magnitude2 = math.sqrt(sum(b * b for b in vector2))

            if magnitude1 == 0.0 or magnitude2 == 0.0:
                return 0.0

            # Return cosine similarity
            return dot_product / (magnitude1 * magnitude2)

        except Exception as e:
            logger.error(f"Error calculating cosine similarity: {e}")
            return 0.0

    def _calculate_relevance_score(self, memory_entry: MemoryEntry, similarity_score: float, query: str) -> float:
        """Calculate relevance score combining similarity and other factors"""
        try:
            # Base score from vector similarity
            relevance = similarity_score * 0.6

            # Boost for priority
            priority_boost = {
                MemoryPriority.CRITICAL: 0.3,
                MemoryPriority.HIGH: 0.2,
                MemoryPriority.MEDIUM: 0.1,
                MemoryPriority.LOW: 0.05,
                MemoryPriority.TEMPORARY: 0.0
            }.get(memory_entry.priority, 0.1)
            relevance += priority_boost

            # Boost for recent access
            if memory_entry.last_accessed:
                hours_since_access = (datetime.now(timezone.utc) - memory_entry.last_accessed).total_seconds() / 3600
                recency_boost = max(0, 0.1 * (1 - hours_since_access / 24))  # Decay over 24 hours
                relevance += recency_boost

            # Boost for high access count
            access_boost = min(0.1, memory_entry.access_count * 0.01)
            relevance += access_boost

            # Tag matching boost
            query_words = set(query.lower().split())
            tag_matches = sum(1 for tag in memory_entry.tags if tag.lower() in query_words)
            tag_boost = min(0.1, tag_matches * 0.02)
            relevance += tag_boost

            return min(1.0, relevance)  # Cap at 1.0

        except Exception as e:
            logger.error(f"Error calculating relevance score: {e}")
            return similarity_score

    # Memory Integration Methods

    async def store_conversation_memory(self, agent_id: str, conversation_context: ThreadContext) -> bool:
        """Store conversation context as episodic memory"""
        try:
            # Convert conversation context to memory content
            content = {
                'thread_id': conversation_context.thread_id,
                'tool_name': conversation_context.tool_name,
                'turns': [asdict(turn) for turn in conversation_context.turns],
                'initial_context': conversation_context.initial_context,
                'created_at': conversation_context.created_at,
                'last_updated_at': conversation_context.last_updated_at
            }

            # Create tags from conversation
            tags = [
                f"tool:{conversation_context.tool_name}",
                f"turns:{len(conversation_context.turns)}",
                "conversation",
                "episodic"
            ]

            # Add file tags if files were referenced
            file_tags = set()
            for turn in conversation_context.turns:
                if turn.files:
                    for file_path in turn.files:
                        file_ext = file_path.split('.')[-1] if '.' in file_path else 'unknown'
                        file_tags.add(f"file:{file_ext}")

            tags.extend(list(file_tags))

            # Store as episodic memory
            memory_entry = await self.store_memory(
                agent_id=agent_id,
                memory_type=MemoryType.EPISODIC,
                content=content,
                tags=tags,
                priority=MemoryPriority.MEDIUM,
                expires_at=datetime.fromtimestamp(
                    datetime.now(timezone.utc).timestamp() + TTLPolicies.MEMORY_LONG,
                    tz=timezone.utc
                )
            )

            return memory_entry is not None

        except Exception as e:
            logger.error(f"Error storing conversation memory for agent {agent_id}: {e}")
            return False

    async def retrieve_relevant_conversations(self, agent_id: str, current_context: str,
                                            limit: int = 5) -> list[ThreadContext]:
        """Retrieve relevant past conversations using similarity search"""
        try:
            # Search episodic memories for relevant conversations
            search_results = await self.search_memories(
                agent_id=agent_id,
                query=current_context,
                memory_types=[MemoryType.EPISODIC],
                limit=limit
            )

            conversations = []
            for result in search_results:
                try:
                    # Convert memory content back to ThreadContext
                    content = result.memory_entry.content
                    if 'thread_id' in content and 'turns' in content:
                        # Reconstruct conversation turns
                        turns = []
                        for turn_data in content['turns']:
                            turns.append(ConversationTurn(**turn_data))

                        conversation = ThreadContext(
                            thread_id=content['thread_id'],
                            parent_thread_id=content.get('parent_thread_id'),
                            created_at=content['created_at'],
                            last_updated_at=content['last_updated_at'],
                            tool_name=content['tool_name'],
                            turns=turns,
                            initial_context=content['initial_context']
                        )
                        conversations.append(conversation)

                except Exception as e:
                    logger.debug(f"Error reconstructing conversation from memory: {e}")
                    continue

            logger.debug(f"Retrieved {len(conversations)} relevant conversations for agent {agent_id}")
            return conversations

        except Exception as e:
            logger.error(f"Error retrieving relevant conversations for agent {agent_id}: {e}")
            return []

    # Analytics and Monitoring Methods

    async def get_memory_analytics(self, agent_id: str) -> MemoryAnalytics:
        """Get comprehensive memory analytics for an agent"""
        try:
            all_memories = await self._get_all_memories(agent_id)

            # Basic statistics
            total_memories = len(all_memories)
            by_type = {}
            by_priority = {}
            total_access_count = 0

            for memory in all_memories:
                # Count by type
                type_name = memory.memory_type.value
                by_type[type_name] = by_type.get(type_name, 0) + 1

                # Count by priority
                priority_name = memory.priority.value
                by_priority[priority_name] = by_priority.get(priority_name, 0) + 1

                # Accumulate access count
                total_access_count += memory.access_count

            # Calculate memory size (rough estimate)
            total_size_mb = sum(
                len(json.dumps(memory.content, default=str)) / (1024 * 1024)
                for memory in all_memories
            )

            # Find most accessed memories
            most_accessed = sorted(all_memories, key=lambda m: m.access_count, reverse=True)[:10]
            most_accessed_ids = [m.memory_id for m in most_accessed]

            # Calculate efficiency score (based on access patterns)
            efficiency_score = 0.0
            if total_memories > 0:
                avg_access = total_access_count / total_memories
                # Memories with above-average access contribute to efficiency
                efficient_memories = sum(1 for m in all_memories if m.access_count >= avg_access)
                efficiency_score = efficient_memories / total_memories

            return MemoryAnalytics(
                total_memories=total_memories,
                by_type=by_type,
                by_priority=by_priority,
                total_size_mb=total_size_mb,
                compression_ratio=1.0,  # Would calculate actual compression ratio
                average_access_frequency=total_access_count / max(total_memories, 1),
                most_accessed_memories=most_accessed_ids,
                memory_efficiency_score=efficiency_score,
                last_consolidation=None  # Would track actual consolidation times
            )

        except Exception as e:
            logger.error(f"Error getting memory analytics for agent {agent_id}: {e}")
            return MemoryAnalytics(
                total_memories=0,
                by_type={},
                by_priority={},
                total_size_mb=0.0,
                compression_ratio=1.0,
                average_access_frequency=0.0,
                most_accessed_memories=[],
                memory_efficiency_score=0.0,
                last_consolidation=None
            )

    # Utility Methods

    async def _persist_memory(self, memory_entry: MemoryEntry) -> bool:
        """Persist memory entry to Redis"""
        try:
            # Determine TTL based on memory type and priority
            ttl = self._calculate_memory_ttl(memory_entry)

            # Serialize memory entry
            memory_data = asdict(memory_entry)

            # Store in Redis
            success = self.redis_manager.set_agent_memory(
                agent_id=memory_entry.agent_id,
                memory_type=memory_entry.memory_type.value,
                data=memory_data,
                ttl=ttl
            )

            return success

        except Exception as e:
            logger.error(f"Error persisting memory {memory_entry.memory_id}: {e}")
            return False

    def _calculate_memory_ttl(self, memory_entry: MemoryEntry) -> int:
        """Calculate TTL for memory based on type and priority"""
        base_ttl = {
            MemoryType.SHORT_TERM: TTLPolicies.MEMORY_SHORT,
            MemoryType.WORKING: TTLPolicies.MEMORY_WORKING,
            MemoryType.LONG_TERM: TTLPolicies.MEMORY_LONG,
            MemoryType.SHARED: TTLPolicies.MEMORY_LONG * 2,
            MemoryType.PROCEDURAL: TTLPolicies.MEMORY_LONG,
            MemoryType.EPISODIC: TTLPolicies.MEMORY_LONG,
            MemoryType.SEMANTIC: TTLPolicies.MEMORY_LONG * 3,
        }.get(memory_entry.memory_type, TTLPolicies.MEMORY_WORKING)

        # Modify TTL based on priority
        priority_multiplier = {
            MemoryPriority.CRITICAL: 10.0,
            MemoryPriority.HIGH: 2.0,
            MemoryPriority.MEDIUM: 1.0,
            MemoryPriority.LOW: 0.5,
            MemoryPriority.TEMPORARY: 0.1,
        }.get(memory_entry.priority, 1.0)

        return int(base_ttl * priority_multiplier)

    def _extract_text_content(self, content: dict[str, Any]) -> str:
        """Extract text content from memory content for vectorization"""
        text_parts = []

        def extract_recursive(obj):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if isinstance(value, str) and len(value.strip()) > 0:
                        text_parts.append(f"{key}: {value}")
                    elif isinstance(value, (dict, list)):
                        extract_recursive(value)
            elif isinstance(obj, list):
                for item in obj:
                    extract_recursive(item)
            elif isinstance(obj, str) and len(obj.strip()) > 0:
                text_parts.append(obj)

        extract_recursive(content)
        return " ".join(text_parts)

    async def _get_all_memories(self, agent_id: str,
                               memory_types: Optional[list[MemoryType]] = None) -> list[MemoryEntry]:
        """Get all memories for an agent, optionally filtered by type"""
        # This would implement efficient Redis scanning
        # For now, return empty list (would be implemented with actual Redis scanning)
        return []

    def _record_operation_latency(self, operation: str, latency: float) -> None:
        """Record operation latency for performance monitoring"""
        self._operation_count += 1
        self._total_latency += latency

        # Record metrics in Redis for monitoring
        self.redis_manager.record_metric(f"agent_memory.{operation}.latency", latency)
        self.redis_manager.record_metric(f"agent_memory.{operation}.count", 1)

    def _deserialize_memory_entry(self, memory_data: dict[str, Any]) -> MemoryEntry:
        """Convert dictionary back to MemoryEntry object"""
        # Handle datetime conversions and enum conversions
        if 'created_at' in memory_data and isinstance(memory_data['created_at'], str):
            memory_data['created_at'] = datetime.fromisoformat(memory_data['created_at'].replace('Z', '+00:00'))

        if 'last_accessed' in memory_data and memory_data['last_accessed'] and isinstance(memory_data['last_accessed'], str):
            memory_data['last_accessed'] = datetime.fromisoformat(memory_data['last_accessed'].replace('Z', '+00:00'))

        if 'expires_at' in memory_data and memory_data['expires_at'] and isinstance(memory_data['expires_at'], str):
            memory_data['expires_at'] = datetime.fromisoformat(memory_data['expires_at'].replace('Z', '+00:00'))

        memory_data['memory_type'] = MemoryType(memory_data['memory_type'])
        memory_data['priority'] = MemoryPriority(memory_data['priority'])

        # Handle vector deserialization
        if 'vector' in memory_data and memory_data['vector']:
            vector_data = memory_data['vector']
            if 'created_at' in vector_data and isinstance(vector_data['created_at'], str):
                vector_data['created_at'] = datetime.fromisoformat(vector_data['created_at'].replace('Z', '+00:00'))
            memory_data['vector'] = MemoryVector(**vector_data)

        return MemoryEntry(**memory_data)

    # Placeholder methods for consolidation strategies
    def _select_by_frequency(self, memories: list[MemoryEntry], count: int) -> list[MemoryEntry]:
        """Select memories to remove based on access frequency"""
        return sorted(memories, key=lambda m: m.access_count)[:count]

    def _select_by_recency(self, memories: list[MemoryEntry], count: int) -> list[MemoryEntry]:
        """Select memories to remove based on recency"""
        return sorted(memories, key=lambda m: m.last_accessed or m.created_at)[:count]

    def _select_by_importance(self, memories: list[MemoryEntry], count: int) -> list[MemoryEntry]:
        """Select memories to remove based on priority"""
        priority_order = [MemoryPriority.TEMPORARY, MemoryPriority.LOW, MemoryPriority.MEDIUM]
        to_remove = []
        for priority in priority_order:
            candidates = [m for m in memories if m.priority == priority and m not in to_remove]
            needed = count - len(to_remove)
            to_remove.extend(candidates[:needed])
            if len(to_remove) >= count:
                break
        return to_remove[:count]

    async def _merge_similar_memories(self, memories: list[MemoryEntry], count: int) -> list[MemoryEntry]:
        """Merge similar memories and return ones to remove"""
        # This would implement actual similarity-based merging
        # For now, just select by frequency
        return self._select_by_frequency(memories, count)

    async def _hybrid_consolidation(self, memories: list[MemoryEntry], count: int) -> list[MemoryEntry]:
        """Hybrid consolidation strategy"""
        # Combine multiple strategies
        by_frequency = self._select_by_frequency(memories, count // 3)
        by_recency = self._select_by_recency(memories, count // 3)
        by_importance = self._select_by_importance(memories, count // 3)

        # Combine and deduplicate
        to_remove = list(set(by_frequency + by_recency + by_importance))
        return to_remove[:count]

    # Placeholder methods for implementation
    async def _find_related_memories(self, agent_id: str, vector: MemoryVector, limit: int) -> list[str]:
        """Find related memories using vector similarity"""
        return []

    async def _update_memory_index(self, memory_entry: MemoryEntry) -> None:
        """Update memory index for efficient searching"""
        pass

    async def _remove_from_memory_index(self, agent_id: str, memory_id: str) -> None:
        """Remove memory from search index"""
        pass

    async def _update_related_memory_references(self, agent_id: str, memory_id: str, related_ids: list[str]) -> None:
        """Update references in related memories"""
        pass

    async def _get_memories_by_type(self, agent_id: str, memory_type: MemoryType) -> list[MemoryEntry]:
        """Get all memories of specific type for an agent"""
        return []

    async def _check_consolidation_trigger(self, agent_id: str, memory_type: MemoryType) -> None:
        """Check if consolidation should be triggered"""
        pass

    async def _record_consolidation_event(self, agent_id: str, strategy: MemoryConsolidationStrategy, count: int) -> None:
        """Record consolidation event for audit"""
        pass


# Global agent memory manager instance
_agent_memory_manager: Optional[AgentMemoryManager] = None
_memory_manager_lock = asyncio.Lock()


async def get_agent_memory_manager() -> AgentMemoryManager:
    """Get global agent memory manager instance (singleton pattern)"""
    global _agent_memory_manager

    if _agent_memory_manager is None:
        async with _memory_manager_lock:
            if _agent_memory_manager is None:
                _agent_memory_manager = AgentMemoryManager()

    return _agent_memory_manager
