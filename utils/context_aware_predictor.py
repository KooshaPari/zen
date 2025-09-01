"""
Context-Aware Predictor for Token Management and Context Window Optimization

This module handles intelligent context window management, token budgeting, and
predictive allocation to maximize efficiency while staying within model limits.

Key Features:
- Dynamic context window allocation
- Token budget management across conversation turns
- Predictive token usage estimation
- Context compression strategies
- Overflow prevention and recovery
- Multi-model context coordination
"""

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

import tiktoken

logger = logging.getLogger(__name__)


class ContextStrategy(Enum):
    """Context management strategies."""
    ROLLING_WINDOW = "rolling_window"  # Keep most recent N tokens
    IMPORTANCE_BASED = "importance_based"  # Keep most important content
    SUMMARY_COMPRESSION = "summary_compression"  # Summarize old content
    HIERARCHICAL = "hierarchical"  # Multi-level context management
    ADAPTIVE = "adaptive"  # Dynamically adjust based on patterns


class TokenType(Enum):
    """Types of tokens in context."""
    SYSTEM_PROMPT = "system_prompt"
    CONVERSATION_HISTORY = "conversation_history"
    USER_INPUT = "user_input"
    TOOL_OUTPUTS = "tool_outputs"
    FUNCTION_CALLS = "function_calls"
    RESERVED_OUTPUT = "reserved_output"


@dataclass
class TokenAllocation:
    """Token allocation for different context components."""
    token_type: TokenType
    allocated_tokens: int
    used_tokens: int
    priority: int  # 1-10, higher is more important
    compressible: bool

    @property
    def remaining_tokens(self) -> int:
        return max(0, self.allocated_tokens - self.used_tokens)

    @property
    def utilization_rate(self) -> float:
        if self.allocated_tokens == 0:
            return 0.0
        return min(1.0, self.used_tokens / self.allocated_tokens)


@dataclass
class ContextWindow:
    """Represents a model's context window state."""
    model_name: str
    total_tokens: int
    max_tokens: int

    # Token allocations by type
    allocations: dict[TokenType, TokenAllocation] = field(default_factory=dict)

    # Historical usage
    token_history: list[int] = field(default_factory=list)
    overflow_count: int = 0
    compression_count: int = 0

    @property
    def available_tokens(self) -> int:
        return max(0, self.max_tokens - self.total_tokens)

    @property
    def utilization_rate(self) -> float:
        return min(1.0, self.total_tokens / self.max_tokens)

    @property
    def is_near_limit(self) -> bool:
        return self.utilization_rate > 0.85

    @property
    def has_overflow_risk(self) -> bool:
        return self.utilization_rate > 0.95


@dataclass
class ConversationContext:
    """Manages conversation-level context across turns."""
    conversation_id: str
    turns: list[dict[str, Any]] = field(default_factory=list)
    total_tokens: int = 0

    # Token usage by turn
    turn_tokens: list[int] = field(default_factory=list)

    # Importance scores for content
    importance_scores: dict[str, float] = field(default_factory=dict)

    # Compression history
    compressions: list[dict[str, Any]] = field(default_factory=list)

    # Metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def add_turn(self, role: str, content: str, tokens: int, metadata: Optional[dict] = None):
        """Add a conversation turn."""
        turn = {
            'role': role,
            'content': content,
            'tokens': tokens,
            'timestamp': datetime.now(timezone.utc),
            'metadata': metadata or {}
        }
        self.turns.append(turn)
        self.turn_tokens.append(tokens)
        self.total_tokens += tokens
        self.last_updated = datetime.now(timezone.utc)

    def get_recent_context(self, max_tokens: int) -> tuple[list[dict], int]:
        """Get most recent context within token limit."""
        selected_turns = []
        current_tokens = 0

        # Work backwards from most recent
        for turn, tokens in zip(reversed(self.turns), reversed(self.turn_tokens)):
            if current_tokens + tokens > max_tokens:
                break
            selected_turns.insert(0, turn)
            current_tokens += tokens

        return selected_turns, current_tokens


class TokenEstimator:
    """Estimates token counts for different content types."""

    def __init__(self):
        """Initialize token estimator with encoding models."""
        self.encoders = {}
        self._init_encoders()

        # Estimation factors for different content types
        self.estimation_factors = {
            'code': 1.2,  # Code tends to use more tokens
            'json': 1.3,  # JSON structure adds overhead
            'markdown': 1.1,  # Markdown formatting
            'plain_text': 1.0,  # Base rate
            'compressed': 0.7  # After compression
        }

    def _init_encoders(self):
        """Initialize tiktoken encoders for different models."""
        try:
            # GPT-4 and GPT-3.5 models
            self.encoders['gpt-4'] = tiktoken.get_encoding("cl100k_base")
            self.encoders['gpt-3.5'] = tiktoken.get_encoding("cl100k_base")

            # Claude models (approximate with GPT-4 encoding)
            self.encoders['claude'] = tiktoken.get_encoding("cl100k_base")

            # Default fallback
            self.encoders['default'] = tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            logger.warning(f"Failed to initialize tiktoken encoders: {e}")
            self.encoders['default'] = None

    def estimate_tokens(self, content: str, content_type: str = 'plain_text', model: str = 'default') -> int:
        """
        Estimate token count for content.

        Args:
            content: Text content to estimate
            content_type: Type of content (code, json, etc.)
            model: Model family for encoding

        Returns:
            Estimated token count
        """
        if not content:
            return 0

        # Get appropriate encoder
        encoder = self.encoders.get(model, self.encoders.get('default'))

        if encoder:
            try:
                # Actual token count
                base_tokens = len(encoder.encode(content))
            except Exception as e:
                logger.warning(f"Encoding failed, using heuristic: {e}")
                # Fallback heuristic: ~4 characters per token
                base_tokens = len(content) // 4
        else:
            # Heuristic estimation
            base_tokens = len(content) // 4

        # Apply content type factor
        factor = self.estimation_factors.get(content_type, 1.0)

        return int(base_tokens * factor)

    def estimate_structured_data(self, data: dict[str, Any]) -> int:
        """Estimate tokens for structured data (JSON-like)."""
        json_str = json.dumps(data, separators=(',', ':'))
        return self.estimate_tokens(json_str, 'json')

    def estimate_function_call(self, function_name: str, arguments: dict[str, Any]) -> int:
        """Estimate tokens for a function call."""
        # Function name and structure overhead
        base_tokens = self.estimate_tokens(function_name) + 10

        # Arguments
        arg_tokens = self.estimate_structured_data(arguments)

        return base_tokens + arg_tokens


class ContextCompressor:
    """Handles context compression strategies."""

    def __init__(self):
        """Initialize context compressor."""
        self.compression_cache = {}
        self.cache_ttl_seconds = 3600  # 1 hour

    def compress_text(self, text: str, target_ratio: float = 0.5) -> tuple[str, float]:
        """
        Compress text to reduce token usage.

        Args:
            text: Text to compress
            target_ratio: Target compression ratio (0-1)

        Returns:
            Tuple of (compressed_text, actual_ratio)
        """
        # Check cache
        cache_key = hashlib.md5(f"{text}:{target_ratio}".encode()).hexdigest()
        if cache_key in self.compression_cache:
            cached = self.compression_cache[cache_key]
            if (datetime.now(timezone.utc) - cached['timestamp']).seconds < self.cache_ttl_seconds:
                return cached['compressed'], cached['ratio']

        # Simple compression strategies
        original_length = len(text)

        if target_ratio >= 0.8:
            # Light compression: remove extra whitespace
            compressed = ' '.join(text.split())
        elif target_ratio >= 0.5:
            # Medium compression: remove redundant content
            compressed = self._remove_redundancy(text)
        else:
            # Heavy compression: extract key points
            compressed = self._extract_key_points(text)

        actual_ratio = len(compressed) / original_length if original_length > 0 else 1.0

        # Cache result
        self.compression_cache[cache_key] = {
            'compressed': compressed,
            'ratio': actual_ratio,
            'timestamp': datetime.now(timezone.utc)
        }

        return compressed, actual_ratio

    def _remove_redundancy(self, text: str) -> str:
        """Remove redundant content from text."""
        lines = text.split('\n')
        unique_lines = []
        seen_content = set()

        for line in lines:
            # Skip empty lines
            if not line.strip():
                continue

            # Skip duplicate or very similar lines
            line_hash = hashlib.md5(line.lower().strip().encode()).hexdigest()[:8]
            if line_hash not in seen_content:
                unique_lines.append(line)
                seen_content.add(line_hash)

        return '\n'.join(unique_lines)

    def _extract_key_points(self, text: str) -> str:
        """Extract key points from text (heavy compression)."""
        # This is a simplified version - in production, you'd use NLP
        sentences = text.split('.')

        # Keep first and last sentences, and every 3rd sentence
        key_sentences = []
        for i, sentence in enumerate(sentences):
            if i == 0 or i == len(sentences) - 1 or i % 3 == 0:
                if sentence.strip():
                    key_sentences.append(sentence.strip())

        return '. '.join(key_sentences) + '.'

    def compress_conversation(self,
                            turns: list[dict[str, Any]],
                            target_tokens: int,
                            estimator: TokenEstimator) -> tuple[list[dict], int]:
        """
        Compress conversation history to fit within token limit.

        Args:
            turns: Conversation turns
            target_tokens: Target token count
            estimator: Token estimator

        Returns:
            Tuple of (compressed_turns, total_tokens)
        """
        if not turns:
            return [], 0

        # Start with most recent turns (most important)
        compressed_turns = []
        current_tokens = 0

        # Keep recent turns as-is
        recent_threshold = min(3, len(turns))
        for turn in turns[-recent_threshold:]:
            turn_tokens = estimator.estimate_tokens(turn['content'])
            compressed_turns.insert(0, turn)
            current_tokens += turn_tokens

        # Compress older turns if needed
        for turn in reversed(turns[:-recent_threshold]):
            turn_tokens = estimator.estimate_tokens(turn['content'])

            if current_tokens + turn_tokens > target_tokens:
                # Try compression
                compressed_content, ratio = self.compress_text(
                    turn['content'],
                    target_ratio=0.5
                )
                compressed_tokens = estimator.estimate_tokens(compressed_content)

                if current_tokens + compressed_tokens <= target_tokens:
                    compressed_turn = turn.copy()
                    compressed_turn['content'] = compressed_content
                    compressed_turn['compressed'] = True
                    compressed_turns.insert(0, compressed_turn)
                    current_tokens += compressed_tokens
                else:
                    # Can't fit even compressed, stop
                    break
            else:
                compressed_turns.insert(0, turn)
                current_tokens += turn_tokens

        return compressed_turns, current_tokens


class ContextAwarePredictor:
    """
    Main predictor for context-aware token management and optimization.
    """

    def __init__(self):
        """Initialize the context-aware predictor."""
        self.estimator = TokenEstimator()
        self.compressor = ContextCompressor()

        # Context windows for different models
        self.model_contexts = {}

        # Conversation contexts
        self.conversations = {}

        # Default context strategies by model tier
        self.default_strategies = {
            'small': ContextStrategy.ROLLING_WINDOW,
            'medium': ContextStrategy.IMPORTANCE_BASED,
            'large': ContextStrategy.HIERARCHICAL,
            'xlarge': ContextStrategy.ADAPTIVE
        }

        # Model context limits (approximate)
        self.model_limits = {
            "gemini-2.5-flash": 32768,
            "gpt-4o-mini": 128000,
            "claude-3-5-haiku-20241022": 200000,
            "gpt-4o": 128000,
            "claude-3-5-sonnet-20241022": 200000,
            "gemini-2.5-pro": 2097152,
            "o1-preview": 128000,
            "deepseek-chat": 32768
        }

    def initialize_context_window(self, model_name: str) -> ContextWindow:
        """
        Initialize context window for a model.

        Args:
            model_name: Name of the model

        Returns:
            Initialized context window
        """
        max_tokens = self.model_limits.get(model_name, 32768)

        window = ContextWindow(
            model_name=model_name,
            total_tokens=0,
            max_tokens=max_tokens
        )

        # Set default allocations based on model size
        if max_tokens <= 32768:
            # Small context: conservative allocation
            allocations = {
                TokenType.SYSTEM_PROMPT: 0.10,
                TokenType.CONVERSATION_HISTORY: 0.40,
                TokenType.USER_INPUT: 0.20,
                TokenType.TOOL_OUTPUTS: 0.15,
                TokenType.RESERVED_OUTPUT: 0.15
            }
        elif max_tokens <= 128000:
            # Medium context: balanced allocation
            allocations = {
                TokenType.SYSTEM_PROMPT: 0.05,
                TokenType.CONVERSATION_HISTORY: 0.50,
                TokenType.USER_INPUT: 0.15,
                TokenType.TOOL_OUTPUTS: 0.20,
                TokenType.RESERVED_OUTPUT: 0.10
            }
        else:
            # Large context: generous allocation
            allocations = {
                TokenType.SYSTEM_PROMPT: 0.02,
                TokenType.CONVERSATION_HISTORY: 0.60,
                TokenType.USER_INPUT: 0.10,
                TokenType.TOOL_OUTPUTS: 0.20,
                TokenType.RESERVED_OUTPUT: 0.08
            }

        # Create token allocations
        for token_type, ratio in allocations.items():
            window.allocations[token_type] = TokenAllocation(
                token_type=token_type,
                allocated_tokens=int(max_tokens * ratio),
                used_tokens=0,
                priority=self._get_default_priority(token_type),
                compressible=token_type != TokenType.SYSTEM_PROMPT
            )

        self.model_contexts[model_name] = window
        return window

    def _get_default_priority(self, token_type: TokenType) -> int:
        """Get default priority for token type."""
        priorities = {
            TokenType.SYSTEM_PROMPT: 10,
            TokenType.USER_INPUT: 9,
            TokenType.RESERVED_OUTPUT: 8,
            TokenType.TOOL_OUTPUTS: 6,
            TokenType.CONVERSATION_HISTORY: 5,
            TokenType.FUNCTION_CALLS: 4
        }
        return priorities.get(token_type, 5)

    def predict_token_usage(self,
                          model_name: str,
                          user_input: str,
                          conversation_id: Optional[str] = None,
                          include_tools: bool = False) -> dict[str, Any]:
        """
        Predict token usage for a request.

        Args:
            model_name: Model to use
            user_input: User input text
            conversation_id: Optional conversation ID
            include_tools: Whether tools might be used

        Returns:
            Token usage prediction
        """
        # Get or create context window
        if model_name not in self.model_contexts:
            window = self.initialize_context_window(model_name)
        else:
            window = self.model_contexts[model_name]

        predictions = {
            'model': model_name,
            'max_context': window.max_tokens,
            'current_usage': window.total_tokens,
            'available': window.available_tokens
        }

        # Estimate input tokens
        input_tokens = self.estimator.estimate_tokens(user_input)
        predictions['input_tokens'] = input_tokens

        # Get conversation context if exists
        conversation_tokens = 0
        if conversation_id and conversation_id in self.conversations:
            conv = self.conversations[conversation_id]
            conversation_tokens = conv.total_tokens
        predictions['conversation_tokens'] = conversation_tokens

        # Estimate system prompt (approximate)
        system_tokens = window.allocations.get(TokenType.SYSTEM_PROMPT,
                                              TokenAllocation(TokenType.SYSTEM_PROMPT, 1000, 0, 10, False)).used_tokens
        predictions['system_tokens'] = system_tokens

        # Estimate tool outputs if applicable
        tool_tokens = 0
        if include_tools:
            # Rough estimate for tool interactions
            tool_tokens = min(2000, window.max_tokens * 0.1)
        predictions['tool_tokens'] = tool_tokens

        # Calculate total input context
        total_input = input_tokens + conversation_tokens + system_tokens + tool_tokens
        predictions['total_input_tokens'] = total_input

        # Estimate output tokens based on input complexity
        if "write" in user_input.lower() or "create" in user_input.lower():
            output_tokens = min(4000, window.max_tokens * 0.2)
        elif "explain" in user_input.lower() or "analyze" in user_input.lower():
            output_tokens = min(2000, window.max_tokens * 0.15)
        else:
            output_tokens = min(1000, window.max_tokens * 0.1)
        predictions['estimated_output_tokens'] = output_tokens

        # Total predicted usage
        total_predicted = total_input + output_tokens
        predictions['total_predicted_tokens'] = total_predicted

        # Check if fits
        predictions['fits_in_context'] = total_predicted <= window.max_tokens
        predictions['utilization_rate'] = min(1.0, total_predicted / window.max_tokens)

        # Recommend compression if needed
        if not predictions['fits_in_context']:
            compression_needed = total_predicted - window.max_tokens
            predictions['compression_needed'] = compression_needed
            predictions['recommended_strategy'] = self._recommend_compression_strategy(
                window, compression_needed
            )

        return predictions

    def _recommend_compression_strategy(self,
                                       window: ContextWindow,
                                       tokens_to_save: int) -> dict[str, Any]:
        """Recommend compression strategy to save tokens."""
        strategy = {
            'tokens_to_save': tokens_to_save,
            'recommendations': []
        }

        # Sort allocations by priority (ascending) and compressibility
        compressible = [
            alloc for alloc in window.allocations.values()
            if alloc.compressible and alloc.used_tokens > 0
        ]
        compressible.sort(key=lambda x: x.priority)

        remaining_to_save = tokens_to_save

        for alloc in compressible:
            if remaining_to_save <= 0:
                break

            # Can compress up to 50% of compressible content
            potential_savings = int(alloc.used_tokens * 0.5)

            if potential_savings > 0:
                strategy['recommendations'].append({
                    'type': alloc.token_type.value,
                    'current_tokens': alloc.used_tokens,
                    'potential_savings': min(potential_savings, remaining_to_save),
                    'method': 'compression' if alloc.token_type == TokenType.CONVERSATION_HISTORY else 'truncation'
                })
                remaining_to_save -= potential_savings

        strategy['total_potential_savings'] = tokens_to_save - max(0, remaining_to_save)
        strategy['sufficient'] = remaining_to_save <= 0

        return strategy

    def allocate_tokens(self,
                       model_name: str,
                       requirements: dict[TokenType, int],
                       strict: bool = False) -> tuple[dict[TokenType, int], bool]:
        """
        Allocate tokens based on requirements and constraints.

        Args:
            model_name: Model to allocate for
            requirements: Required tokens by type
            strict: If True, fail if requirements can't be met

        Returns:
            Tuple of (allocations, success)
        """
        # Get or create context window
        if model_name not in self.model_contexts:
            window = self.initialize_context_window(model_name)
        else:
            window = self.model_contexts[model_name]

        allocations = {}
        total_required = sum(requirements.values())

        if total_required > window.max_tokens:
            if strict:
                return {}, False

            # Scale down proportionally
            scale_factor = window.max_tokens / total_required * 0.95  # Leave 5% buffer

            for token_type, required in requirements.items():
                allocations[token_type] = int(required * scale_factor)
        else:
            allocations = requirements.copy()

        # Update window allocations
        for token_type, allocated in allocations.items():
            if token_type in window.allocations:
                window.allocations[token_type].used_tokens = allocated

        window.total_tokens = sum(allocations.values())

        return allocations, True

    def manage_conversation_context(self,
                                   conversation_id: str,
                                   new_turn: dict[str, Any],
                                   model_name: str,
                                   max_history_tokens: Optional[int] = None) -> tuple[list[dict], int]:
        """
        Manage conversation context for a turn.

        Args:
            conversation_id: Conversation ID
            new_turn: New conversation turn
            model_name: Model being used
            max_history_tokens: Maximum tokens for history

        Returns:
            Tuple of (context_turns, total_tokens)
        """
        # Get or create conversation
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = ConversationContext(conversation_id)

        conv = self.conversations[conversation_id]

        # Add new turn
        turn_tokens = self.estimator.estimate_tokens(new_turn.get('content', ''))
        conv.add_turn(
            role=new_turn.get('role', 'user'),
            content=new_turn.get('content', ''),
            tokens=turn_tokens,
            metadata=new_turn.get('metadata')
        )

        # Determine max tokens for history
        if max_history_tokens is None:
            window = self.model_contexts.get(model_name)
            if window:
                history_alloc = window.allocations.get(TokenType.CONVERSATION_HISTORY)
                max_history_tokens = history_alloc.allocated_tokens if history_alloc else 10000
            else:
                max_history_tokens = 10000

        # Get context within limits
        if conv.total_tokens <= max_history_tokens:
            return conv.turns, conv.total_tokens

        # Need compression or truncation
        strategy = self._get_context_strategy(model_name)

        if strategy == ContextStrategy.ROLLING_WINDOW:
            return conv.get_recent_context(max_history_tokens)

        elif strategy == ContextStrategy.SUMMARY_COMPRESSION:
            return self.compressor.compress_conversation(
                conv.turns, max_history_tokens, self.estimator
            )

        else:
            # Default to rolling window
            return conv.get_recent_context(max_history_tokens)

    def _get_context_strategy(self, model_name: str) -> ContextStrategy:
        """Get context management strategy for model."""
        # Determine model tier
        max_tokens = self.model_limits.get(model_name, 32768)

        if max_tokens <= 32768:
            tier = 'small'
        elif max_tokens <= 128000:
            tier = 'medium'
        elif max_tokens <= 500000:
            tier = 'large'
        else:
            tier = 'xlarge'

        return self.default_strategies.get(tier, ContextStrategy.ROLLING_WINDOW)

    def get_context_recommendations(self, model_name: str) -> dict[str, Any]:
        """
        Get recommendations for context management.

        Args:
            model_name: Model to get recommendations for

        Returns:
            Context management recommendations
        """
        window = self.model_contexts.get(model_name)

        if not window:
            return {'status': 'no_context_initialized'}

        recommendations = {
            'model': model_name,
            'utilization': window.utilization_rate,
            'status': 'healthy'
        }

        if window.has_overflow_risk:
            recommendations['status'] = 'critical'
            recommendations['action'] = 'immediate_compression_required'
            recommendations['strategy'] = self._recommend_compression_strategy(
                window, int(window.total_tokens * 0.3)
            )

        elif window.is_near_limit:
            recommendations['status'] = 'warning'
            recommendations['action'] = 'consider_compression'
            recommendations['strategy'] = self._recommend_compression_strategy(
                window, int(window.total_tokens * 0.2)
            )

        # Token allocation efficiency
        allocation_efficiency = {}
        for token_type, alloc in window.allocations.items():
            allocation_efficiency[token_type.value] = {
                'allocated': alloc.allocated_tokens,
                'used': alloc.used_tokens,
                'utilization': alloc.utilization_rate,
                'priority': alloc.priority
            }

        recommendations['allocation_efficiency'] = allocation_efficiency

        # Historical patterns
        if window.token_history:
            recommendations['average_usage'] = sum(window.token_history) / len(window.token_history)
            recommendations['peak_usage'] = max(window.token_history)
            recommendations['overflow_rate'] = window.overflow_count / len(window.token_history) if window.token_history else 0

        return recommendations


# Global instance
_context_predictor = None


def get_context_predictor() -> ContextAwarePredictor:
    """Get or create the global context-aware predictor."""
    global _context_predictor
    if _context_predictor is None:
        _context_predictor = ContextAwarePredictor()
    return _context_predictor
