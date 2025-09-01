#!/usr/bin/env python3
"""
NATS Performance Benchmarks Runner

This script runs comprehensive performance benchmarks for the NATS messaging
infrastructure to validate performance characteristics and identify bottlenecks.

Usage:
    python scripts/run_nats_benchmarks.py [--component COMPONENT] [--output FILE]

Components:
    - communicator: Core NATS communication benchmarks
    - streaming: JetStream persistent messaging benchmarks
    - discovery: Agent discovery service benchmarks
    - edge: Edge orchestration benchmarks
    - monitoring: Monitoring system benchmarks
    - integration: End-to-end integration benchmarks
    - all: Run all benchmarks (default)
"""

import argparse
import asyncio
import json
import logging
import statistics
import sys
import time
from datetime import datetime, timezone
from typing import Any

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BenchmarkResult:
    """Container for benchmark results."""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.start_time = None
        self.end_time = None
        self.duration = None
        self.metrics = {}
        self.success = False
        self.error = None

    def start(self):
        """Start timing the benchmark."""
        self.start_time = time.perf_counter()

    def end(self):
        """End timing the benchmark."""
        self.end_time = time.perf_counter()
        self.duration = self.end_time - self.start_time
        self.success = True

    def fail(self, error: str):
        """Mark benchmark as failed."""
        if self.start_time and not self.end_time:
            self.end_time = time.perf_counter()
            self.duration = self.end_time - self.start_time
        self.error = error
        self.success = False

    def add_metric(self, name: str, value: Any, unit: str = ""):
        """Add a performance metric."""
        self.metrics[name] = {"value": value, "unit": unit}

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "description": self.description,
            "duration": self.duration,
            "success": self.success,
            "error": self.error,
            "metrics": self.metrics,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }


class NATSBenchmarkRunner:
    """Runs comprehensive NATS performance benchmarks."""

    def __init__(self):
        self.results: list[BenchmarkResult] = []

    async def run_communicator_benchmarks(self) -> list[BenchmarkResult]:
        """Run core NATS communicator benchmarks."""
        benchmarks = []

        # Message throughput benchmark
        result = BenchmarkResult("message_throughput", "NATS message publishing throughput")
        result.start()

        try:
            from unittest.mock import AsyncMock, patch

            from utils.nats_communicator import NATSCommunicator, NATSConfig

            # Mock NATS for benchmarking
            with patch('utils.nats_communicator.nats') as mock_nats:
                mock_client = AsyncMock()
                mock_client._client_id = "benchmark_client"
                mock_client.jetstream.return_value = AsyncMock()
                mock_nats.connect.return_value = mock_client

                config = NATSConfig(servers=["nats://localhost:4222"])
                communicator = NATSCommunicator(config)
                await communicator.connect()

                # Throughput test
                message_count = 10000
                message_size = 1024  # 1KB
                test_message = {"data": "x" * message_size}

                throughput_start = time.perf_counter()

                tasks = []
                for i in range(message_count):
                    task = communicator.publish(f"benchmark.msg.{i}", test_message)
                    tasks.append(task)

                results = await asyncio.gather(*tasks, return_exceptions=True)
                throughput_end = time.perf_counter()

                successful = sum(1 for r in results if r is True)
                duration = throughput_end - throughput_start
                throughput = successful / duration

                result.add_metric("messages_sent", message_count, "messages")
                result.add_metric("successful_sends", successful, "messages")
                result.add_metric("throughput", throughput, "messages/sec")
                result.add_metric("data_rate", (throughput * message_size / 1024 / 1024), "MB/sec")
                result.add_metric("message_size", message_size, "bytes")

                await communicator.disconnect()

        except Exception as e:
            result.fail(str(e))

        result.end()
        benchmarks.append(result)

        # Latency benchmark
        result = BenchmarkResult("message_latency", "NATS message publishing latency distribution")
        result.start()

        try:
            with patch('utils.nats_communicator.nats') as mock_nats:
                mock_client = AsyncMock()
                mock_client._client_id = "latency_client"
                mock_client.jetstream.return_value = AsyncMock()
                mock_nats.connect.return_value = mock_client

                config = NATSConfig(servers=["nats://localhost:4222"])
                communicator = NATSCommunicator(config)
                await communicator.connect()

                # Latency test
                sample_count = 1000
                latencies = []

                for i in range(sample_count):
                    start = time.perf_counter()
                    await communicator.publish(f"benchmark.latency.{i}", {"data": i})
                    end = time.perf_counter()
                    latencies.append((end - start) * 1000)  # Convert to ms

                if latencies:
                    result.add_metric("samples", len(latencies), "samples")
                    result.add_metric("avg_latency", statistics.mean(latencies), "ms")
                    result.add_metric("p50_latency", statistics.median(latencies), "ms")
                    result.add_metric("p95_latency", sorted(latencies)[int(len(latencies) * 0.95)], "ms")
                    result.add_metric("p99_latency", sorted(latencies)[int(len(latencies) * 0.99)], "ms")
                    result.add_metric("min_latency", min(latencies), "ms")
                    result.add_metric("max_latency", max(latencies), "ms")

                await communicator.disconnect()

        except Exception as e:
            result.fail(str(e))

        result.end()
        benchmarks.append(result)

        return benchmarks

    async def run_streaming_benchmarks(self) -> list[BenchmarkResult]:
        """Run JetStream streaming benchmarks."""
        benchmarks = []

        result = BenchmarkResult("jetstream_throughput", "JetStream persistent messaging throughput")
        result.start()

        try:
            from unittest.mock import AsyncMock, MagicMock, patch

            from utils.nats_communicator import NATSCommunicator, NATSConfig
            from utils.nats_streaming import NATSStreamingManager

            with patch('utils.nats_streaming.nats'):
                config = NATSConfig(servers=["nats://localhost:4222"])
                communicator = NATSCommunicator(config)

                manager = NATSStreamingManager(communicator)

                # Mock JetStream
                mock_js = AsyncMock()
                mock_js.publish.return_value = MagicMock(stream="TEST_STREAM", seq=1, duplicate=False)
                mock_js.account_info.return_value = MagicMock(
                    memory=1024, storage=1024*10, streams=1, consumers=1,
                    limits=MagicMock(max_memory=1024*1024, max_storage=1024*1024*10,
                                   max_streams=100, max_consumers=1000)
                )
                manager.js = mock_js

                # Throughput test
                message_count = 5000
                messages_published = 0

                stream_start = time.perf_counter()

                for i in range(message_count):
                    success = await manager.publish_persistent(
                        f"test.stream.{i}",
                        {"data": f"message_{i}", "timestamp": time.time()}
                    )
                    if success:
                        messages_published += 1

                stream_end = time.perf_counter()

                duration = stream_end - stream_start
                throughput = messages_published / duration

                result.add_metric("messages_attempted", message_count, "messages")
                result.add_metric("messages_published", messages_published, "messages")
                result.add_metric("throughput", throughput, "messages/sec")
                result.add_metric("success_rate", messages_published / message_count, "ratio")

        except Exception as e:
            result.fail(str(e))

        result.end()
        benchmarks.append(result)

        return benchmarks

    async def run_discovery_benchmarks(self) -> list[BenchmarkResult]:
        """Run service discovery benchmarks."""
        benchmarks = []

        result = BenchmarkResult("service_discovery", "Agent service discovery performance")
        result.start()

        try:
            from unittest.mock import AsyncMock, MagicMock, patch

            from tools.shared.agent_models import AgentType, TaskStatus
            from utils.agent_discovery import AgentDiscoveryService, AgentService, ServiceQuery
            from utils.nats_communicator import NATSCommunicator, NATSConfig

            with patch('utils.nats_communicator.nats') as mock_nats:
                mock_client = AsyncMock()
                mock_client._client_id = "discovery_client"
                mock_nats.connect.return_value = mock_client

                config = NATSConfig(servers=["nats://localhost:4222"])
                communicator = NATSCommunicator(config)
                await communicator.connect()

                mock_redis = MagicMock()
                mock_redis.ping.return_value = True

                discovery = AgentDiscoveryService(communicator, mock_redis)
                await discovery.start()

                # Register services
                service_count = 1000
                registration_start = time.perf_counter()

                for i in range(service_count):
                    service = AgentService(
                        service_id=f"service_{i}",
                        agent_id=f"agent_{i}",
                        agent_type=AgentType.CLAUDE if i % 2 == 0 else AgentType.AIDER,
                        capabilities={"cap1", "cap2"} if i % 3 == 0 else {"cap2"},
                        status=TaskStatus.RUNNING,
                        load_factor=0.1 * (i % 10)
                    )

                    await discovery.register_service(service)

                registration_end = time.perf_counter()
                registration_time = registration_end - registration_start

                # Query services
                query_count = 500
                query_start = time.perf_counter()

                for i in range(query_count):
                    query = ServiceQuery(
                        agent_type=AgentType.CLAUDE if i % 2 == 0 else None,
                        capabilities={"cap2"},
                        max_load=0.5
                    )

                    services = await discovery.discover_services(query)
                    assert len(services) > 0

                query_end = time.perf_counter()
                query_time = query_end - query_start

                result.add_metric("services_registered", service_count, "services")
                result.add_metric("registration_time", registration_time, "seconds")
                result.add_metric("registration_rate", service_count / registration_time, "services/sec")
                result.add_metric("queries_executed", query_count, "queries")
                result.add_metric("query_time", query_time, "seconds")
                result.add_metric("query_rate", query_count / query_time, "queries/sec")
                result.add_metric("avg_query_latency", (query_time / query_count) * 1000, "ms")

                await discovery.stop()
                await communicator.disconnect()

        except Exception as e:
            result.fail(str(e))

        result.end()
        benchmarks.append(result)

        return benchmarks

    async def run_monitoring_benchmarks(self) -> list[BenchmarkResult]:
        """Run monitoring system benchmarks."""
        benchmarks = []

        result = BenchmarkResult("metrics_recording", "Metrics recording and aggregation performance")
        result.start()

        try:
            from unittest.mock import AsyncMock, MagicMock, patch

            from utils.nats_communicator import NATSCommunicator, NATSConfig
            from utils.nats_monitoring import NATSMonitoringSystem

            with patch('utils.nats_communicator.nats') as mock_nats:
                mock_client = AsyncMock()
                mock_client._client_id = "monitoring_client"
                mock_nats.connect.return_value = mock_client

                config = NATSConfig(servers=["nats://localhost:4222"])
                communicator = NATSCommunicator(config)
                await communicator.connect()

                mock_redis = MagicMock()
                mock_redis.ping.return_value = True

                monitor = NATSMonitoringSystem(communicator, redis_client=mock_redis)
                await monitor.start()

                # Record metrics
                metric_count = 10000
                recording_start = time.perf_counter()

                for i in range(metric_count):
                    await monitor.record_metric(
                        f"test_metric_{i % 10}",  # 10 different metrics
                        float(i % 100),
                        {"source": "benchmark", "batch": str(i // 1000)}
                    )

                recording_end = time.perf_counter()
                recording_time = recording_end - recording_start

                # Test metrics retrieval
                retrieval_start = time.perf_counter()

                summary = await monitor.get_metrics_summary()

                retrieval_end = time.perf_counter()
                retrieval_time = retrieval_end - retrieval_start

                result.add_metric("metrics_recorded", metric_count, "metrics")
                result.add_metric("recording_time", recording_time, "seconds")
                result.add_metric("recording_rate", metric_count / recording_time, "metrics/sec")
                result.add_metric("retrieval_time", retrieval_time, "seconds")
                result.add_metric("unique_metrics", len(summary), "metrics")

                await monitor.stop()
                await communicator.disconnect()

        except Exception as e:
            result.fail(str(e))

        result.end()
        benchmarks.append(result)

        return benchmarks

    async def run_integration_benchmarks(self) -> list[BenchmarkResult]:
        """Run end-to-end integration benchmarks."""
        benchmarks = []

        result = BenchmarkResult("end_to_end_integration", "Complete system integration performance")
        result.start()

        try:
            from unittest.mock import AsyncMock, MagicMock, patch

            from tools.shared.agent_models import AgentType
            from utils.agent_discovery import AgentDiscoveryService, AgentService
            from utils.nats_communicator import NATSCommunicator, NATSConfig
            from utils.nats_monitoring import NATSMonitoringSystem
            from utils.nats_streaming import NATSStreamingManager

            # Setup all components
            with patch('utils.nats_communicator.nats') as mock_nats:
                mock_client = AsyncMock()
                mock_client._client_id = "integration_client"
                mock_client.jetstream.return_value = AsyncMock()
                mock_nats.connect.return_value = mock_client

                config = NATSConfig(servers=["nats://localhost:4222"])
                communicator = NATSCommunicator(config)
                await communicator.connect()

                mock_redis = MagicMock()
                mock_redis.ping.return_value = True

                # Initialize all components
                streaming = NATSStreamingManager(communicator, mock_redis)
                mock_js = AsyncMock()
                mock_js.stream_info.return_value = MagicMock()
                mock_js.account_info.return_value = MagicMock(
                    memory=1024, storage=1024*10, streams=1, consumers=1,
                    limits=MagicMock(max_memory=1024*1024, max_storage=1024*1024*10,
                                   max_streams=100, max_consumers=1000)
                )
                streaming.js = mock_js
                await streaming.initialize()

                discovery = AgentDiscoveryService(communicator, mock_redis)
                await discovery.start()

                monitor = NATSMonitoringSystem(communicator, streaming, discovery, redis_client=mock_redis)
                await monitor.start()

                # Run integration scenario
                scenario_start = time.perf_counter()

                # 1. Register services
                services_registered = 0
                for i in range(100):
                    service = AgentService(
                        service_id=f"integration_service_{i}",
                        agent_id=f"integration_agent_{i}",
                        agent_type=AgentType.CLAUDE,
                        capabilities={"integration", "testing"}
                    )

                    success = await discovery.register_service(service)
                    if success:
                        services_registered += 1

                # 2. Publish messages
                messages_published = 0
                for i in range(500):
                    success = await communicator.publish(
                        f"integration.test.{i}",
                        {"data": f"integration_message_{i}", "timestamp": time.time()}
                    )
                    if success:
                        messages_published += 1

                # 3. Record metrics
                metrics_recorded = 0
                for i in range(200):
                    await monitor.record_metric(f"integration_metric_{i % 20}", float(i), {"test": "integration"})
                    metrics_recorded += 1

                # 4. Query services
                queries_successful = 0
                for i in range(50):
                    from utils.agent_discovery import ServiceQuery
                    query = ServiceQuery(capabilities={"integration"})
                    services = await discovery.discover_services(query)
                    if len(services) > 0:
                        queries_successful += 1

                scenario_end = time.perf_counter()
                scenario_time = scenario_end - scenario_start

                result.add_metric("total_time", scenario_time, "seconds")
                result.add_metric("services_registered", services_registered, "services")
                result.add_metric("messages_published", messages_published, "messages")
                result.add_metric("metrics_recorded", metrics_recorded, "metrics")
                result.add_metric("queries_successful", queries_successful, "queries")
                result.add_metric("overall_throughput", (services_registered + messages_published + metrics_recorded) / scenario_time, "operations/sec")

                # Cleanup
                await monitor.stop()
                await discovery.stop()
                await streaming.shutdown()
                await communicator.disconnect()

        except Exception as e:
            result.fail(str(e))

        result.end()
        benchmarks.append(result)

        return benchmarks

    async def run_benchmarks(self, component: str = "all") -> list[BenchmarkResult]:
        """Run benchmarks for specified component(s)."""
        all_results = []

        logger.info(f"Starting NATS benchmarks for component: {component}")

        if component in ["all", "communicator"]:
            logger.info("Running NATS communicator benchmarks...")
            results = await self.run_communicator_benchmarks()
            all_results.extend(results)

        if component in ["all", "streaming"]:
            logger.info("Running NATS streaming benchmarks...")
            results = await self.run_streaming_benchmarks()
            all_results.extend(results)

        if component in ["all", "discovery"]:
            logger.info("Running service discovery benchmarks...")
            results = await self.run_discovery_benchmarks()
            all_results.extend(results)

        if component in ["all", "monitoring"]:
            logger.info("Running monitoring system benchmarks...")
            results = await self.run_monitoring_benchmarks()
            all_results.extend(results)

        if component in ["all", "integration"]:
            logger.info("Running integration benchmarks...")
            results = await self.run_integration_benchmarks()
            all_results.extend(results)

        self.results.extend(all_results)
        return all_results

    def print_results(self, results: list[BenchmarkResult]) -> None:
        """Print benchmark results to console."""
        print("\n" + "="*80)
        print("NATS PERFORMANCE BENCHMARK RESULTS")
        print("="*80)

        for result in results:
            print(f"\n{result.name.upper()}: {result.description}")
            print("-" * 60)

            if result.success:
                print(f"Duration: {result.duration:.3f}s")
                print("Metrics:")
                for name, metric in result.metrics.items():
                    value = metric["value"]
                    unit = metric["unit"]
                    if isinstance(value, float):
                        print(f"  {name}: {value:.3f} {unit}")
                    else:
                        print(f"  {name}: {value} {unit}")
            else:
                print(f"FAILED: {result.error}")

        # Summary
        successful = sum(1 for r in results if r.success)
        total = len(results)

        print("\n" + "="*80)
        print("BENCHMARK SUMMARY")
        print("="*80)
        print(f"Total benchmarks: {total}")
        print(f"Successful: {successful}")
        print(f"Failed: {total - successful}")
        print(f"Success rate: {(successful/total)*100:.1f}%")

        if successful > 0:
            total_duration = sum(r.duration for r in results if r.duration)
            print(f"Total execution time: {total_duration:.3f}s")

    def save_results(self, results: list[BenchmarkResult], filename: str) -> None:
        """Save benchmark results to JSON file."""
        data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_benchmarks": len(results),
            "successful": sum(1 for r in results if r.success),
            "failed": sum(1 for r in results if not r.success),
            "results": [result.to_dict() for result in results]
        }

        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Benchmark results saved to: {filename}")


async def main():
    """Main function to run benchmarks."""
    parser = argparse.ArgumentParser(description="Run NATS performance benchmarks")
    parser.add_argument("--component", choices=["communicator", "streaming", "discovery", "monitoring", "integration", "all"],
                       default="all", help="Component to benchmark")
    parser.add_argument("--output", help="Output file for results (JSON format)")
    parser.add_argument("--quiet", action="store_true", help="Suppress console output")

    args = parser.parse_args()

    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)

    runner = NATSBenchmarkRunner()

    try:
        results = await runner.run_benchmarks(args.component)

        if not args.quiet:
            runner.print_results(results)

        if args.output:
            runner.save_results(results, args.output)

        # Exit with error code if any benchmarks failed
        failed_count = sum(1 for r in results if not r.success)
        if failed_count > 0:
            logger.error(f"{failed_count} benchmarks failed")
            sys.exit(1)
        else:
            logger.info("All benchmarks completed successfully")
            sys.exit(0)

    except KeyboardInterrupt:
        logger.info("Benchmarks interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Benchmark execution failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
