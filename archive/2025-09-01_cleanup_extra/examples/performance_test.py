#!/usr/bin/env python3
"""
MCP Streamable HTTP Performance Testing

This script tests the performance characteristics of the MCP Streamable HTTP
implementation, including throughput, latency, concurrent connections, and
resource usage patterns.
"""

import asyncio
import os
import statistics
import sys
import time
from dataclasses import dataclass

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clients.mcp_http_client import MCPStreamableHTTPClient


@dataclass
class PerformanceResult:
    """Performance test result."""
    test_name: str
    success_count: int
    error_count: int
    total_time: float
    min_time: float
    max_time: float
    avg_time: float
    median_time: float
    requests_per_second: float
    errors: list[str]


class MCPPerformanceTester:
    """Performance testing framework for MCP Streamable HTTP."""

    def __init__(self, server_url: str = "http://localhost:8080/mcp"):
        self.server_url = server_url
        self.results: list[PerformanceResult] = []

    async def test_basic_latency(self, num_requests: int = 100) -> PerformanceResult:
        """Test basic request/response latency."""
        print(f"🚀 Testing basic latency with {num_requests} requests...")

        times = []
        errors = []
        success_count = 0

        async with MCPStreamableHTTPClient(self.server_url) as client:
            start_total = time.time()

            for i in range(num_requests):
                try:
                    start = time.time()
                    await client.call_tool("echo", {"text": f"test-{i}"})
                    end = time.time()

                    times.append(end - start)
                    success_count += 1

                    if (i + 1) % 20 == 0:
                        print(f"  📊 Completed {i + 1}/{num_requests} requests")

                except Exception as e:
                    errors.append(f"Request {i}: {str(e)}")

            total_time = time.time() - start_total

        return PerformanceResult(
            test_name="Basic Latency",
            success_count=success_count,
            error_count=len(errors),
            total_time=total_time,
            min_time=min(times) if times else 0,
            max_time=max(times) if times else 0,
            avg_time=statistics.mean(times) if times else 0,
            median_time=statistics.median(times) if times else 0,
            requests_per_second=success_count / total_time if total_time > 0 else 0,
            errors=errors[:10]  # Keep first 10 errors
        )

    async def test_concurrent_requests(self, num_concurrent: int = 50, requests_each: int = 5) -> PerformanceResult:
        """Test concurrent request handling."""
        print(f"⚡ Testing {num_concurrent} concurrent connections with {requests_each} requests each...")

        times = []
        errors = []
        success_count = 0

        async def worker(worker_id: int):
            """Worker function for concurrent testing."""
            nonlocal times, errors, success_count

            try:
                async with MCPStreamableHTTPClient(self.server_url, timeout=30) as client:
                    for i in range(requests_each):
                        try:
                            start = time.time()
                            await client.call_tool("multiply", {"a": worker_id, "b": i + 1})
                            end = time.time()

                            times.append(end - start)
                            success_count += 1

                        except Exception as e:
                            errors.append(f"Worker {worker_id}, Request {i}: {str(e)}")

            except Exception as e:
                errors.append(f"Worker {worker_id} connection failed: {str(e)}")

        # Run concurrent workers
        start_total = time.time()
        tasks = [worker(i) for i in range(num_concurrent)]
        await asyncio.gather(*tasks, return_exceptions=True)
        total_time = time.time() - start_total

        return PerformanceResult(
            test_name="Concurrent Requests",
            success_count=success_count,
            error_count=len(errors),
            total_time=total_time,
            min_time=min(times) if times else 0,
            max_time=max(times) if times else 0,
            avg_time=statistics.mean(times) if times else 0,
            median_time=statistics.median(times) if times else 0,
            requests_per_second=success_count / total_time if total_time > 0 else 0,
            errors=errors[:10]
        )

    async def test_large_payload(self, num_requests: int = 20) -> PerformanceResult:
        """Test handling of large payloads."""
        print(f"📦 Testing large payload handling with {num_requests} requests...")

        # Create a large text payload (approximately 1MB)
        large_text = "A" * (1024 * 1024)  # 1MB of A's

        times = []
        errors = []
        success_count = 0

        async with MCPStreamableHTTPClient(self.server_url, timeout=60) as client:
            start_total = time.time()

            for i in range(num_requests):
                try:
                    start = time.time()
                    result = await client.call_tool("echo", {"text": large_text})
                    end = time.time()

                    # Verify response
                    if len(result) >= len(large_text):
                        times.append(end - start)
                        success_count += 1
                    else:
                        errors.append(f"Request {i}: Response truncated")

                    print(f"  📊 Completed {i + 1}/{num_requests} large requests")

                except Exception as e:
                    errors.append(f"Request {i}: {str(e)}")

            total_time = time.time() - start_total

        return PerformanceResult(
            test_name="Large Payload",
            success_count=success_count,
            error_count=len(errors),
            total_time=total_time,
            min_time=min(times) if times else 0,
            max_time=max(times) if times else 0,
            avg_time=statistics.mean(times) if times else 0,
            median_time=statistics.median(times) if times else 0,
            requests_per_second=success_count / total_time if total_time > 0 else 0,
            errors=errors[:10]
        )

    async def test_connection_lifecycle(self, num_connections: int = 20) -> PerformanceResult:
        """Test connection establishment and teardown."""
        print(f"🔄 Testing connection lifecycle with {num_connections} connections...")

        times = []
        errors = []
        success_count = 0

        start_total = time.time()

        for i in range(num_connections):
            try:
                start = time.time()

                # Full connection lifecycle: connect, request, disconnect
                client = MCPStreamableHTTPClient(self.server_url)
                await client.connect()

                # Make a simple request
                await client.call_tool("get_time")

                await client.disconnect()
                end = time.time()

                times.append(end - start)
                success_count += 1

                if (i + 1) % 5 == 0:
                    print(f"  📊 Completed {i + 1}/{num_connections} connection cycles")

            except Exception as e:
                errors.append(f"Connection {i}: {str(e)}")

        total_time = time.time() - start_total

        return PerformanceResult(
            test_name="Connection Lifecycle",
            success_count=success_count,
            error_count=len(errors),
            total_time=total_time,
            min_time=min(times) if times else 0,
            max_time=max(times) if times else 0,
            avg_time=statistics.mean(times) if times else 0,
            median_time=statistics.median(times) if times else 0,
            requests_per_second=success_count / total_time if total_time > 0 else 0,
            errors=errors[:10]
        )

    async def test_tool_variety(self, requests_per_tool: int = 20) -> PerformanceResult:
        """Test performance across different tool types."""
        print(f"🛠️ Testing tool variety with {requests_per_tool} requests per tool...")

        tools_to_test = [
            ("echo", {"text": "performance test"}),
            ("multiply", {"a": 42, "b": 13}),
            ("get_time", {}),
        ]

        times = []
        errors = []
        success_count = 0

        async with MCPStreamableHTTPClient(self.server_url) as client:
            start_total = time.time()

            for tool_name, tool_args in tools_to_test:
                print(f"  🧪 Testing {tool_name} tool...")

                for i in range(requests_per_tool):
                    try:
                        start = time.time()
                        await client.call_tool(tool_name, tool_args)
                        end = time.time()

                        times.append(end - start)
                        success_count += 1

                    except Exception as e:
                        errors.append(f"{tool_name} request {i}: {str(e)}")

            total_time = time.time() - start_total

        return PerformanceResult(
            test_name="Tool Variety",
            success_count=success_count,
            error_count=len(errors),
            total_time=total_time,
            min_time=min(times) if times else 0,
            max_time=max(times) if times else 0,
            avg_time=statistics.mean(times) if times else 0,
            median_time=statistics.median(times) if times else 0,
            requests_per_second=success_count / total_time if total_time > 0 else 0,
            errors=errors[:10]
        )

    async def test_session_reuse(self, num_requests: int = 100) -> PerformanceResult:
        """Test session reuse performance."""
        print(f"🔄 Testing session reuse with {num_requests} requests...")

        times = []
        errors = []
        success_count = 0

        # Single persistent connection
        async with MCPStreamableHTTPClient(self.server_url) as client:
            start_total = time.time()

            for i in range(num_requests):
                try:
                    start = time.time()
                    await client.call_tool("echo", {"text": f"session-test-{i}"})
                    end = time.time()

                    times.append(end - start)
                    success_count += 1

                    if (i + 1) % 25 == 0:
                        print(f"  📊 Completed {i + 1}/{num_requests} session requests")

                except Exception as e:
                    errors.append(f"Request {i}: {str(e)}")

            total_time = time.time() - start_total

        return PerformanceResult(
            test_name="Session Reuse",
            success_count=success_count,
            error_count=len(errors),
            total_time=total_time,
            min_time=min(times) if times else 0,
            max_time=max(times) if times else 0,
            avg_time=statistics.mean(times) if times else 0,
            median_time=statistics.median(times) if times else 0,
            requests_per_second=success_count / total_time if total_time > 0 else 0,
            errors=errors[:10]
        )

    def print_result(self, result: PerformanceResult):
        """Print formatted test result."""
        print(f"\n📊 {result.test_name} Results:")
        print(f"  {'='*50}")
        print(f"  ✅ Successful requests: {result.success_count}")
        print(f"  ❌ Failed requests: {result.error_count}")
        print(f"  🕒 Total time: {result.total_time:.2f}s")
        print(f"  ⚡ Requests per second: {result.requests_per_second:.2f}")
        print("  📈 Response times:")
        print(f"    • Min: {result.min_time*1000:.1f}ms")
        print(f"    • Max: {result.max_time*1000:.1f}ms")
        print(f"    • Avg: {result.avg_time*1000:.1f}ms")
        print(f"    • Median: {result.median_time*1000:.1f}ms")

        if result.errors:
            print("  ⚠️ Sample errors:")
            for error in result.errors[:3]:  # Show first 3 errors
                print(f"    • {error}")
            if len(result.errors) > 3:
                print(f"    • ... and {len(result.errors) - 3} more")

    def print_summary(self):
        """Print performance test summary."""
        if not self.results:
            print("No results to summarize")
            return

        print("\n🎯 Performance Test Summary")
        print(f"{'='*60}")

        total_requests = sum(r.success_count for r in self.results)
        total_errors = sum(r.error_count for r in self.results)
        total_time = sum(r.total_time for r in self.results)

        print("📊 Overall Statistics:")
        print(f"  • Total successful requests: {total_requests}")
        print(f"  • Total failed requests: {total_errors}")
        print(f"  • Total test time: {total_time:.2f}s")
        print(f"  • Overall success rate: {(total_requests/(total_requests+total_errors)*100):.1f}%")

        # Best/worst performing tests
        if len(self.results) > 1:
            best_rps = max(self.results, key=lambda r: r.requests_per_second)
            worst_rps = min(self.results, key=lambda r: r.requests_per_second)

            print("\n🏆 Performance Rankings:")
            print(f"  • Best throughput: {best_rps.test_name} ({best_rps.requests_per_second:.2f} req/s)")
            print(f"  • Worst throughput: {worst_rps.test_name} ({worst_rps.requests_per_second:.2f} req/s)")

            best_latency = min(self.results, key=lambda r: r.avg_time)
            worst_latency = max(self.results, key=lambda r: r.avg_time)

            print(f"  • Best latency: {best_latency.test_name} ({best_latency.avg_time*1000:.1f}ms avg)")
            print(f"  • Worst latency: {worst_latency.test_name} ({worst_latency.avg_time*1000:.1f}ms avg)")


async def main():
    """Run comprehensive performance tests."""

    print("🧘 Zen MCP Streamable HTTP Performance Testing")
    print("=" * 70)
    print("This suite tests various performance characteristics of the MCP server")
    print()

    server_url = "http://localhost:8080/mcp"
    tester = MCPPerformanceTester(server_url)

    # Test server connectivity first
    print("🔌 Testing server connectivity...")
    try:
        async with MCPStreamableHTTPClient(server_url, timeout=5) as client:
            await client.call_tool("echo", {"text": "connectivity test"})
        print("✅ Server is responsive")
    except Exception as e:
        print(f"❌ Cannot connect to server: {e}")
        print("💡 Make sure the server is running: python server_mcp_http.py")
        return

    print("\n🚀 Starting performance tests...")

    try:
        # Run all performance tests
        tests = [
            ("Basic Latency", tester.test_basic_latency(50)),
            ("Session Reuse", tester.test_session_reuse(75)),
            ("Tool Variety", tester.test_tool_variety(15)),
            ("Concurrent Requests", tester.test_concurrent_requests(10, 3)),
            ("Connection Lifecycle", tester.test_connection_lifecycle(10)),
            ("Large Payload", tester.test_large_payload(5)),
        ]

        for _test_name, test_coro in tests:
            print(f"\n{'='*60}")
            result = await test_coro
            tester.results.append(result)
            tester.print_result(result)

        # Print final summary
        tester.print_summary()

        print("\n🎉 Performance testing completed!")
        print("\n💡 Optimization tips:")
        print("  • Keep connections alive for better performance")
        print("  • Use appropriate timeouts for your use case")
        print("  • Monitor server resources under load")
        print("  • Consider connection pooling for high throughput")

    except KeyboardInterrupt:
        print("\n⚠️ Performance tests interrupted by user")
    except Exception as e:
        print(f"\n❌ Performance testing failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())
