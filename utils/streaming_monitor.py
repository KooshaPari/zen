"""
Streaming Monitor for Real-Time Metrics Tracking

This module provides real-time monitoring of model execution, tracking tokens
as they stream, measuring latency, and updating predictions with actual values.

Key Features:
- Real-time token counting during streaming
- Time to first token (TTFT) measurement
- Tokens per second (TPS) calculation
- Progressive cost calculation
- Quality signal detection
- Anomaly detection for performance issues
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class StreamingPhase(Enum):
    """Phases of streaming execution."""
    INITIALIZING = "initializing"
    WAITING_FIRST_TOKEN = "waiting_first_token"
    STREAMING = "streaming"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class StreamingMetrics:
    """Real-time metrics during streaming."""
    request_id: str
    model_name: str

    # Timing
    start_time: float = field(default_factory=time.time)
    first_token_time: Optional[float] = None
    last_token_time: Optional[float] = None
    end_time: Optional[float] = None

    # Token counts
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    # Streaming metrics
    tokens_per_second: List[float] = field(default_factory=list)
    current_tps: float = 0.0
    average_tps: float = 0.0

    # Cost tracking
    running_cost: float = 0.0
    input_cost_per_token: float = 0.0
    output_cost_per_token: float = 0.0

    # Quality signals
    coherence_score: float = 1.0
    repetition_detected: bool = False
    truncation_detected: bool = False

    # Status
    phase: StreamingPhase = StreamingPhase.INITIALIZING
    error: Optional[str] = None

    @property
    def time_to_first_token_ms(self) -> Optional[int]:
        """Calculate TTFT in milliseconds."""
        if self.first_token_time and self.start_time:
            return int((self.first_token_time - self.start_time) * 1000)
        return None

    @property
    def total_latency_ms(self) -> Optional[int]:
        """Calculate total latency in milliseconds."""
        if self.end_time and self.start_time:
            return int((self.end_time - self.start_time) * 1000)
        return None

    @property
    def is_streaming(self) -> bool:
        """Check if currently streaming."""
        return self.phase == StreamingPhase.STREAMING


@dataclass
class TokenChunk:
    """Represents a chunk of tokens received."""
    content: str
    token_count: int
    timestamp: float
    chunk_index: int

    # Quality indicators
    is_repetitive: bool = False
    is_truncated: bool = False
    confidence: float = 1.0


class StreamingMonitor:
    """
    Monitors real-time streaming metrics for model execution.
    """

    def __init__(self):
        """Initialize the streaming monitor."""
        self.active_streams: Dict[str, StreamingMetrics] = {}
        self.completed_streams: Dict[str, StreamingMetrics] = {}

        # Callbacks for events
        self.on_first_token_callbacks: List[Callable] = []
        self.on_stream_complete_callbacks: List[Callable] = []
        self.on_anomaly_callbacks: List[Callable] = []

        # Performance thresholds
        self.ttft_warning_ms = 2000  # Warn if TTFT > 2s
        self.ttft_critical_ms = 5000  # Critical if TTFT > 5s
        self.tps_warning = 10  # Warn if TPS < 10
        self.tps_critical = 5  # Critical if TPS < 5

        # Quality detection
        self.repetition_threshold = 0.3  # 30% repeated content
        self.min_coherence_score = 0.7

        logger.info("Streaming monitor initialized")

    def start_stream(self,
                     request_id: str,
                     model_name: str,
                     input_tokens: int,
                     input_cost_per_token: float,
                     output_cost_per_token: float) -> StreamingMetrics:
        """
        Start monitoring a new stream.
        
        Args:
            request_id: Unique request identifier
            model_name: Model being used
            input_tokens: Number of input tokens
            input_cost_per_token: Cost per input token
            output_cost_per_token: Cost per output token
            
        Returns:
            StreamingMetrics object
        """
        metrics = StreamingMetrics(
            request_id=request_id,
            model_name=model_name,
            input_tokens=input_tokens,
            input_cost_per_token=input_cost_per_token,
            output_cost_per_token=output_cost_per_token,
            phase=StreamingPhase.WAITING_FIRST_TOKEN
        )

        # Calculate initial input cost
        metrics.running_cost = input_tokens * input_cost_per_token
        metrics.total_tokens = input_tokens

        self.active_streams[request_id] = metrics

        logger.debug(f"Started monitoring stream {request_id} for model {model_name}")

        return metrics

    def record_token_chunk(self,
                          request_id: str,
                          content: str,
                          token_count: Optional[int] = None) -> Optional[TokenChunk]:
        """
        Record a chunk of tokens received.
        
        Args:
            request_id: Request identifier
            content: Text content of chunk
            token_count: Number of tokens (estimated if not provided)
            
        Returns:
            TokenChunk object or None if stream not found
        """
        if request_id not in self.active_streams:
            logger.warning(f"Stream {request_id} not found in active streams")
            return None

        metrics = self.active_streams[request_id]
        current_time = time.time()

        # Estimate tokens if not provided (rough heuristic)
        if token_count is None:
            token_count = len(content.split()) // 2 or 1

        # Create chunk
        chunk = TokenChunk(
            content=content,
            token_count=token_count,
            timestamp=current_time,
            chunk_index=len(metrics.tokens_per_second)
        )

        # Check for quality issues
        chunk.is_repetitive = self._detect_repetition(content)
        chunk.is_truncated = self._detect_truncation(content)

        # Update metrics
        if metrics.phase == StreamingPhase.WAITING_FIRST_TOKEN:
            metrics.first_token_time = current_time
            metrics.phase = StreamingPhase.STREAMING

            # Trigger first token callbacks
            for callback in self.on_first_token_callbacks:
                try:
                    callback(metrics)
                except Exception as e:
                    logger.error(f"Error in first token callback: {e}")

            # Check TTFT performance
            ttft = metrics.time_to_first_token_ms
            if ttft and ttft > self.ttft_critical_ms:
                self._trigger_anomaly("critical_ttft", metrics, f"TTFT {ttft}ms exceeds critical threshold")
            elif ttft and ttft > self.ttft_warning_ms:
                self._trigger_anomaly("warning_ttft", metrics, f"TTFT {ttft}ms exceeds warning threshold")

        # Calculate TPS
        if metrics.last_token_time:
            time_delta = current_time - metrics.last_token_time
            if time_delta > 0:
                current_tps = token_count / time_delta
                metrics.tokens_per_second.append(current_tps)
                metrics.current_tps = current_tps

                # Update average TPS
                metrics.average_tps = sum(metrics.tokens_per_second) / len(metrics.tokens_per_second)

                # Check TPS performance
                if current_tps < self.tps_critical:
                    self._trigger_anomaly("critical_tps", metrics, f"TPS {current_tps:.1f} below critical threshold")
                elif current_tps < self.tps_warning:
                    self._trigger_anomaly("warning_tps", metrics, f"TPS {current_tps:.1f} below warning threshold")

        # Update token counts and cost
        metrics.output_tokens += token_count
        metrics.total_tokens += token_count
        metrics.running_cost += token_count * metrics.output_cost_per_token
        metrics.last_token_time = current_time

        # Update quality metrics
        if chunk.is_repetitive:
            metrics.repetition_detected = True
        if chunk.is_truncated:
            metrics.truncation_detected = True

        return chunk

    def end_stream(self,
                  request_id: str,
                  success: bool = True,
                  error: Optional[str] = None) -> Optional[StreamingMetrics]:
        """
        Mark a stream as completed.
        
        Args:
            request_id: Request identifier
            success: Whether stream completed successfully
            error: Error message if failed
            
        Returns:
            Final StreamingMetrics or None
        """
        if request_id not in self.active_streams:
            logger.warning(f"Stream {request_id} not found in active streams")
            return None

        metrics = self.active_streams.pop(request_id)
        metrics.end_time = time.time()
        metrics.phase = StreamingPhase.COMPLETED if success else StreamingPhase.FAILED
        metrics.error = error

        # Store in completed streams (with size limit)
        self.completed_streams[request_id] = metrics
        if len(self.completed_streams) > 1000:
            # Remove oldest completed streams
            oldest_key = next(iter(self.completed_streams))
            del self.completed_streams[oldest_key]

        # Trigger completion callbacks
        for callback in self.on_stream_complete_callbacks:
            try:
                callback(metrics)
            except Exception as e:
                logger.error(f"Error in stream complete callback: {e}")

        logger.debug(f"Stream {request_id} ended - tokens: {metrics.output_tokens}, "
                    f"cost: ${metrics.running_cost:.4f}, "
                    f"avg TPS: {metrics.average_tps:.1f}")

        return metrics

    def get_stream_metrics(self, request_id: str) -> Optional[StreamingMetrics]:
        """Get metrics for a stream (active or completed)."""
        return self.active_streams.get(request_id) or self.completed_streams.get(request_id)

    def get_active_streams(self) -> List[StreamingMetrics]:
        """Get all active stream metrics."""
        return list(self.active_streams.values())

    def _detect_repetition(self, content: str) -> bool:
        """Detect repetitive content."""
        if len(content) < 20:
            return False

        # Simple repetition detection - check for repeated phrases
        words = content.lower().split()
        if len(words) < 10:
            return False

        # Check for repeated sequences
        for window_size in [3, 4, 5]:
            if len(words) < window_size * 2:
                continue

            sequences = []
            for i in range(len(words) - window_size + 1):
                seq = ' '.join(words[i:i+window_size])
                sequences.append(seq)

            # Check for high repetition
            unique_sequences = set(sequences)
            if len(unique_sequences) < len(sequences) * 0.7:
                return True

        return False

    def _detect_truncation(self, content: str) -> bool:
        """Detect if content appears truncated."""
        # Check for common truncation indicators
        truncation_indicators = [
            "...",
            "[truncated]",
            "[cut off]",
            "reached the limit",
            "maximum length"
        ]

        content_lower = content.lower()
        return any(indicator in content_lower for indicator in truncation_indicators)

    def _trigger_anomaly(self, anomaly_type: str, metrics: StreamingMetrics, message: str):
        """Trigger anomaly callbacks."""
        anomaly_data = {
            "type": anomaly_type,
            "request_id": metrics.request_id,
            "model_name": metrics.model_name,
            "message": message,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

        logger.warning(f"Anomaly detected: {message}")

        for callback in self.on_anomaly_callbacks:
            try:
                callback(anomaly_data)
            except Exception as e:
                logger.error(f"Error in anomaly callback: {e}")

    def register_callback(self,
                         event_type: str,
                         callback: Callable):
        """
        Register a callback for events.
        
        Args:
            event_type: Type of event ('first_token', 'complete', 'anomaly')
            callback: Function to call
        """
        if event_type == "first_token":
            self.on_first_token_callbacks.append(callback)
        elif event_type == "complete":
            self.on_stream_complete_callbacks.append(callback)
        elif event_type == "anomaly":
            self.on_anomaly_callbacks.append(callback)
        else:
            logger.warning(f"Unknown event type: {event_type}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get overall streaming statistics."""
        all_streams = list(self.active_streams.values()) + list(self.completed_streams.values())

        if not all_streams:
            return {"status": "no_data"}

        completed = [m for m in all_streams if m.phase == StreamingPhase.COMPLETED]
        failed = [m for m in all_streams if m.phase == StreamingPhase.FAILED]

        ttft_values = [m.time_to_first_token_ms for m in completed if m.time_to_first_token_ms]
        tps_values = [m.average_tps for m in completed if m.average_tps > 0]

        return {
            "total_streams": len(all_streams),
            "active_streams": len(self.active_streams),
            "completed_streams": len(completed),
            "failed_streams": len(failed),
            "success_rate": len(completed) / len(all_streams) if all_streams else 0,

            "ttft_stats": {
                "min_ms": min(ttft_values) if ttft_values else None,
                "max_ms": max(ttft_values) if ttft_values else None,
                "avg_ms": sum(ttft_values) / len(ttft_values) if ttft_values else None,
                "samples": len(ttft_values)
            },

            "tps_stats": {
                "min": min(tps_values) if tps_values else None,
                "max": max(tps_values) if tps_values else None,
                "avg": sum(tps_values) / len(tps_values) if tps_values else None,
                "samples": len(tps_values)
            },

            "quality_issues": {
                "repetition_rate": sum(1 for m in completed if m.repetition_detected) / len(completed) if completed else 0,
                "truncation_rate": sum(1 for m in completed if m.truncation_detected) / len(completed) if completed else 0
            },

            "total_tokens_streamed": sum(m.output_tokens for m in all_streams),
            "total_cost": sum(m.running_cost for m in all_streams)
        }

    async def monitor_stream_async(self,
                                  request_id: str,
                                  stream_generator,
                                  model_name: str,
                                  input_tokens: int,
                                  input_cost_per_token: float,
                                  output_cost_per_token: float):
        """
        Async wrapper to monitor a streaming generator.
        
        Args:
            request_id: Request identifier
            stream_generator: Async generator yielding chunks
            model_name: Model being used
            input_tokens: Number of input tokens
            input_cost_per_token: Cost per input token
            output_cost_per_token: Cost per output token
            
        Yields:
            Content chunks from the generator
        """
        # Start monitoring
        metrics = self.start_stream(
            request_id,
            model_name,
            input_tokens,
            input_cost_per_token,
            output_cost_per_token
        )

        try:
            async for chunk in stream_generator:
                # Record the chunk
                if isinstance(chunk, str):
                    self.record_token_chunk(request_id, chunk)
                elif isinstance(chunk, dict) and 'content' in chunk:
                    self.record_token_chunk(
                        request_id,
                        chunk['content'],
                        chunk.get('token_count')
                    )

                # Yield the chunk onward
                yield chunk

            # Mark as completed
            self.end_stream(request_id, success=True)

        except Exception as e:
            # Mark as failed
            self.end_stream(request_id, success=False, error=str(e))
            raise


# Global instance
_streaming_monitor = None


def get_streaming_monitor() -> StreamingMonitor:
    """Get or create the global streaming monitor."""
    global _streaming_monitor
    if _streaming_monitor is None:
        _streaming_monitor = StreamingMonitor()
    return _streaming_monitor
