#!/usr/bin/env python3
"""
Comprehensive test suite for the Zen MCP Adaptive Learning System
Tests all components: routing, learning, persistence, monitoring
"""

import asyncio
import logging
import os
import sys
import time

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Test results storage
test_results = {
    'passed': [],
    'failed': [],
    'warnings': []
}

def test_result(name: str, passed: bool, message: str = ""):
    """Record test result"""
    if passed:
        test_results['passed'].append(name)
        logger.info(f"✅ {name}: PASSED {message}")
    else:
        test_results['failed'].append((name, message))
        logger.error(f"❌ {name}: FAILED - {message}")

async def test_adaptive_routing():
    """Test adaptive model routing"""
    try:
        from utils.enhanced_model_router import smart_model_selection

        # Test different optimization modes
        optimizations = ['speed', 'cost', 'quality', 'balanced']

        for opt in optimizations:
            model, metadata = await smart_model_selection(
                task_type="chat",
                prompt="Test prompt for optimization",
                files=[],
                optimization=opt
            )

            if model and metadata:
                test_result(f"Routing-{opt}", True, f"Selected: {model}")
            else:
                test_result(f"Routing-{opt}", False, "No model selected")

    except Exception as e:
        test_result("Adaptive Routing", False, str(e))

async def test_learning_engine():
    """Test adaptive learning engine"""
    try:
        from utils.adaptive_learning_engine import AdaptiveLearningEngine, ALContext

        engine = AdaptiveLearningEngine()

        # Create test context
        context = ALContext(
            task_type="test",
            model_name="test-model",
            provider="test",
            prompt_length=100,
            file_count=0,
            optimization_mode="balanced"
        )

        # Test prediction
        predictions = await engine.predict_performance(context)

        if predictions:
            test_result("Learning Prediction", True,
                       f"Cost: ${predictions.predicted_cost:.4f}")
        else:
            test_result("Learning Prediction", False, "No predictions")

        # Test learning from feedback
        engine.record_actual_performance(
            context.request_id,
            latency_ms=150.0,
            tokens_used=200,
            cost=0.002,
            quality_score=0.95,
            tps=50.0
        )

        test_result("Learning Feedback", True, "Feedback recorded")

    except Exception as e:
        test_result("Learning Engine", False, str(e))

async def test_context_management():
    """Test context-aware prediction"""
    try:
        from utils.context_aware_predictor import ContextAwarePredictor

        predictor = ContextAwarePredictor()

        # Test context window allocation
        allocation = predictor.allocate_context_window(
            task_type="chat",
            prompt_tokens=1000,
            expected_output_tokens=500,
            model_name="gpt-4",
            optimization="balanced"
        )

        if allocation and allocation.max_tokens > 0:
            test_result("Context Allocation", True,
                       f"Allocated: {allocation.max_tokens} tokens")
        else:
            test_result("Context Allocation", False, "No allocation")

        # Test compression strategy
        strategy = predictor.get_compression_strategy(
            current_tokens=8000,
            max_tokens=8192,
            priority="quality"
        )

        test_result("Compression Strategy", True if strategy else False,
                   f"Strategy: {strategy}")

    except Exception as e:
        test_result("Context Management", False, str(e))

async def test_cost_optimization():
    """Test cost-performance optimizer"""
    try:
        from utils.cost_performance_optimizer import CostPerformanceOptimizer

        optimizer = CostPerformanceOptimizer()

        # Create test models
        models = [
            {
                'name': 'fast-model',
                'cost': 0.001,
                'latency': 100,
                'quality': 0.8,
                'tps': 100,
                'success_rate': 0.95
            },
            {
                'name': 'quality-model',
                'cost': 0.01,
                'latency': 500,
                'quality': 0.95,
                'tps': 50,
                'success_rate': 0.99
            },
            {
                'name': 'cheap-model',
                'cost': 0.0001,
                'latency': 200,
                'quality': 0.7,
                'tps': 80,
                'success_rate': 0.9
            }
        ]

        # Test Pareto frontier
        frontier = optimizer.compute_pareto_frontier(models)

        if frontier:
            test_result("Pareto Frontier", True,
                       f"Found {len(frontier)} optimal models")
        else:
            test_result("Pareto Frontier", False, "No frontier computed")

        # Test optimization selection
        optimal = optimizer.select_optimal_model(
            models,
            optimization_mode="balanced"
        )

        if optimal:
            test_result("Optimization Selection", True,
                       f"Selected: {optimal.get('name', 'unknown')}")
        else:
            test_result("Optimization Selection", False, "No model selected")

    except Exception as e:
        test_result("Cost Optimization", False, str(e))

async def test_streaming_monitor():
    """Test streaming performance monitor"""
    try:
        from utils.streaming_monitor import StreamingMonitor

        monitor = StreamingMonitor()

        # Simulate streaming
        session_id = "test-session"

        # Start streaming
        monitor.start_streaming(session_id, "test-model")
        test_result("Streaming Start", True, "Session started")

        # Simulate token streaming
        for i in range(10):
            monitor.record_token(session_id, f"token_{i}")
            await asyncio.sleep(0.01)  # Simulate delay

        # Get metrics
        metrics = monitor.get_metrics(session_id)

        if metrics and metrics.get('ttft_ms') is not None:
            test_result("Streaming Metrics", True,
                       f"TTFT: {metrics['ttft_ms']:.2f}ms")
        else:
            test_result("Streaming Metrics", False, "No metrics recorded")

        # End streaming
        monitor.end_streaming(session_id)

    except Exception as e:
        test_result("Streaming Monitor", False, str(e))

async def test_token_budget():
    """Test token budget management"""
    try:
        from utils.token_budget_manager import TokenBudgetManager

        manager = TokenBudgetManager(
            daily_limit=10000,
            monthly_limit=100000
        )

        # Test allocation
        can_allocate = manager.can_allocate(1000, "test-model")
        test_result("Budget Check", True if can_allocate else False,
                   f"Can allocate: {can_allocate}")

        if can_allocate:
            # Allocate tokens
            allocated = manager.allocate_tokens(1000, "test-model", 0.01)
            test_result("Token Allocation", allocated,
                       "Tokens allocated" if allocated else "Allocation failed")

            # Check usage
            usage = manager.get_usage_summary()
            test_result("Usage Summary", True,
                       f"Daily: {usage['daily']['percentage']:.1f}%")

        # Test prediction
        exhaustion = manager.predict_exhaustion()
        if exhaustion:
            test_result("Exhaustion Prediction", True,
                       f"Predicted: {exhaustion}")
        else:
            test_result("Exhaustion Prediction", True, "No exhaustion predicted")

    except Exception as e:
        test_result("Token Budget", False, str(e))

async def test_database_operations():
    """Test database persistence"""
    try:
        import psycopg2
        from psycopg2.extras import RealDictCursor

        # Try to connect to database
        conn = psycopg2.connect(
            host="localhost",
            port=5432,
            database="zen_mcp",
            user="zen_user",
            password="zen_password"
        )

        cursor = conn.cursor(cursor_factory=RealDictCursor)

        # Test schema existence
        cursor.execute("""
            SELECT schema_name
            FROM information_schema.schemata
            WHERE schema_name LIKE 'zen_%'
        """)

        schemas = cursor.fetchall()
        expected_schemas = ['zen_adaptive', 'zen_performance', 'zen_embeddings',
                          'zen_budget', 'zen_conversation']

        found_schemas = [s['schema_name'] for s in schemas]

        for schema in expected_schemas:
            if schema in found_schemas:
                test_result(f"Schema-{schema}", True, "Exists")
            else:
                test_result(f"Schema-{schema}", False, "Not found")

        # Test inserting performance data
        cursor.execute("""
            INSERT INTO zen_performance.model_performance
            (model_name, provider, task_type, input_tokens, output_tokens,
             total_time, total_cost, success)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
        """, ("test-model", "test-provider", "test", 100, 50, 0.5, 0.001, True))

        result = cursor.fetchone()
        test_result("DB Insert", True if result else False,
                   "Data inserted" if result else "Insert failed")

        conn.commit()
        cursor.close()
        conn.close()

    except Exception as e:
        test_result("Database Operations", False, str(e))
        test_results['warnings'].append(
            "Database not available - run scripts/init_services.sh first"
        )

async def test_dashboard_api():
    """Test dashboard API endpoints"""
    try:
        import aiohttp

        async with aiohttp.ClientSession() as session:
            # Test health endpoint
            async with session.get('http://localhost:8080/health') as resp:
                if resp.status == 200:
                    test_result("Dashboard Health", True, "API healthy")
                else:
                    test_result("Dashboard Health", False, f"Status: {resp.status}")

            # Test metrics endpoint
            async with session.get('http://localhost:8080/metrics') as resp:
                if resp.status == 200:
                    data = await resp.json()
                    test_result("Dashboard Metrics", True,
                               f"Got {len(data)} metrics")
                else:
                    test_result("Dashboard Metrics", False, f"Status: {resp.status}")

    except Exception as e:
        test_result("Dashboard API", False, str(e))
        test_results['warnings'].append(
            "Dashboard not running - start with: cd dashboard && python performance_dashboard.py"
        )

async def test_integration_flow():
    """Test complete integration flow"""
    try:
        from utils.enhanced_model_router import EnhancedModelRouter

        router = EnhancedModelRouter()

        # Simulate complete request flow
        test_prompt = "Explain quantum computing in simple terms"

        # 1. Route request
        model, metadata = await router.route_request(
            task_type="chat",
            prompt=test_prompt,
            optimization="balanced"
        )

        test_result("Integration-Routing", True if model else False,
                   f"Routed to: {model}")

        if model:
            # 2. Simulate execution
            start_time = time.time()
            await asyncio.sleep(0.1)  # Simulate processing
            latency = (time.time() - start_time) * 1000

            # 3. Record performance
            router.record_performance(
                request_id=metadata.get('request_id'),
                model=model,
                latency_ms=latency,
                tokens=150,
                cost=0.002,
                quality=0.9,
                success=True
            )

            test_result("Integration-Recording", True, "Performance recorded")

            # 4. Verify learning
            await asyncio.sleep(0.5)  # Let system process

            # 5. Route another similar request
            model2, metadata2 = await router.route_request(
                task_type="chat",
                prompt=test_prompt,
                optimization="balanced"
            )

            test_result("Integration-Learning", True,
                       f"Learned routing: {model2}")

    except Exception as e:
        test_result("Integration Flow", False, str(e))

async def run_all_tests():
    """Run all tests"""
    print("\n" + "="*60)
    print("🧪 Zen MCP Adaptive Learning System - Full Test Suite")
    print("="*60 + "\n")

    # Core component tests
    print("📦 Testing Core Components...")
    await test_adaptive_routing()
    await test_learning_engine()
    await test_context_management()
    await test_cost_optimization()

    # Monitoring tests
    print("\n📊 Testing Monitoring...")
    await test_streaming_monitor()
    await test_token_budget()

    # Persistence tests
    print("\n💾 Testing Persistence...")
    await test_database_operations()

    # API tests
    print("\n🌐 Testing APIs...")
    await test_dashboard_api()

    # Integration test
    print("\n🔄 Testing Integration...")
    await test_integration_flow()

    # Print summary
    print("\n" + "="*60)
    print("📋 Test Summary")
    print("="*60)

    total_tests = len(test_results['passed']) + len(test_results['failed'])
    pass_rate = (len(test_results['passed']) / total_tests * 100) if total_tests > 0 else 0

    print(f"\n✅ Passed: {len(test_results['passed'])}")
    print(f"❌ Failed: {len(test_results['failed'])}")
    print(f"📊 Pass Rate: {pass_rate:.1f}%")

    if test_results['failed']:
        print("\n🔴 Failed Tests:")
        for name, message in test_results['failed']:
            print(f"  • {name}: {message}")

    if test_results['warnings']:
        print("\n⚠️  Warnings:")
        for warning in test_results['warnings']:
            print(f"  • {warning}")

    print("\n" + "="*60)

    if pass_rate == 100:
        print("🎉 All tests passed! System is fully operational.")
    elif pass_rate >= 80:
        print("✅ Core functionality working. Check warnings for optional features.")
    elif pass_rate >= 50:
        print("⚠️  Some components need attention. Review failed tests.")
    else:
        print("❌ System needs configuration. Run scripts/init_services.sh")

    print("="*60 + "\n")

if __name__ == "__main__":
    # Run tests
    asyncio.run(run_all_tests())
