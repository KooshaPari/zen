"""
Performance Dashboard for Adaptive Learning System

This module provides a web-based dashboard for monitoring the adaptive learning
system's performance, model selections, and optimization metrics.

Key Features:
- Real-time performance metrics
# ruff: noqa: W291,W293

- Model selection visualization
- Cost/performance analysis
- Learning progress tracking
- Token budget monitoring
- Streaming metrics display
"""

import logging
import os

# Import monitoring components
import sys
from datetime import datetime, timezone
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.adaptive_learning_engine import get_adaptive_engine
from utils.enhanced_model_router import get_enhanced_router
from utils.model_performance_tracker import get_performance_tracker
from utils.streaming_monitor import get_streaming_monitor
from utils.token_budget_manager import BudgetPeriod, get_budget_manager

logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Zen MCP Adaptive Learning Dashboard",
    description="Performance monitoring and optimization dashboard",
    version="1.0.0"
)

# Enable CORS for web access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# === API ENDPOINTS ===

@app.get("/")
async def root():
    """Serve the dashboard HTML page."""
    return HTMLResponse(content=DASHBOARD_HTML)


@app.get("/api/status")
async def get_system_status():
    """Get overall system status."""
    try:
        adaptive_engine = get_adaptive_engine()
        streaming_monitor = get_streaming_monitor()
        budget_manager = get_budget_manager()

        return {
            "status": "online",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "components": {
                "adaptive_learning": {
                    "enabled": adaptive_engine is not None,
                    "learning_metrics": adaptive_engine.learning_metrics if adaptive_engine else None
                },
                "streaming_monitor": {
                    "active_streams": len(streaming_monitor.active_streams),
                    "completed_streams": len(streaming_monitor.completed_streams)
                },
                "budget_manager": {
                    "budgets_configured": len(budget_manager.budgets),
                    "active_allocations": len(budget_manager.allocations)
                }
            }
        }
    except Exception as e:
        logger.error(f"Failed to get system status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/performance/summary")
async def get_performance_summary():
    """Get performance summary from adaptive learning engine."""
    try:
        engine = get_adaptive_engine()
        if not engine:
            return {"status": "adaptive_learning_disabled"}

        return engine.get_performance_summary()
    except Exception as e:
        logger.error(f"Failed to get performance summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/performance/models")
async def get_model_performance(
    days: int = 7,
    model_name: Optional[str] = None,
    task_type: Optional[str] = None
):
    """Get detailed model performance metrics."""
    try:
        _tracker = get_performance_tracker()
        router = get_enhanced_router()

        report = await router.get_performance_report(
            model_name=model_name,
            task_type=task_type,
            days=days
        )

        return report
    except Exception as e:
        logger.error(f"Failed to get model performance: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/streaming/statistics")
async def get_streaming_statistics():
    """Get streaming performance statistics."""
    try:
        monitor = get_streaming_monitor()
        return monitor.get_statistics()
    except Exception as e:
        logger.error(f"Failed to get streaming statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/streaming/active")
async def get_active_streams():
    """Get currently active streams."""
    try:
        monitor = get_streaming_monitor()
        active = monitor.get_active_streams()

        return {
            "count": len(active),
            "streams": [
                {
                    "request_id": s.request_id,
                    "model_name": s.model_name,
                    "phase": s.phase.value,
                    "output_tokens": s.output_tokens,
                    "current_tps": s.current_tps,
                    "running_cost": s.running_cost,
                    "ttft_ms": s.time_to_first_token_ms
                }
                for s in active
            ]
        }
    except Exception as e:
        logger.error(f"Failed to get active streams: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/budget/status")
async def get_budget_status(period: Optional[str] = None):
    """Get token budget status."""
    try:
        manager = get_budget_manager()

        if period:
            period_enum = BudgetPeriod(period)
            return manager.get_budget_status(period_enum)
        else:
            return manager.get_budget_status()
    except Exception as e:
        logger.error(f"Failed to get budget status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/budget/allocations")
async def get_budget_allocations():
    """Get token allocation statistics."""
    try:
        manager = get_budget_manager()
        return manager.get_allocation_statistics()
    except Exception as e:
        logger.error(f"Failed to get allocation statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/budget/exhaustion")
async def predict_budget_exhaustion(period: str = "daily"):
    """Predict when budget will be exhausted."""
    try:
        manager = get_budget_manager()
        period_enum = BudgetPeriod(period)

        exhaustion_time = manager.predict_budget_exhaustion(period_enum)

        if exhaustion_time:
            return {
                "period": period,
                "predicted_exhaustion": exhaustion_time.isoformat(),
                "hours_remaining": (exhaustion_time - datetime.now(timezone.utc)).total_seconds() / 3600
            }
        else:
            return {
                "period": period,
                "predicted_exhaustion": None,
                "message": "Budget not at risk of exhaustion"
            }
    except Exception as e:
        logger.error(f"Failed to predict budget exhaustion: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/recommendations/{task_type}")
async def get_model_recommendations(task_type: str):
    """Get model recommendations for a task type."""
    try:
        router = get_enhanced_router()
        recommendations = await router.get_model_recommendations(task_type)

        return {
            "task_type": task_type,
            "recommendations": recommendations,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# === WEBSOCKET FOR REAL-TIME UPDATES ===

@app.websocket("/ws/metrics")
async def websocket_metrics(websocket: WebSocket):
    """WebSocket endpoint for real-time metrics streaming."""
    await websocket.accept()

    try:
        import asyncio

        while True:
            # Send real-time updates every 2 seconds
            monitor = get_streaming_monitor()
            manager = get_budget_manager()

            update = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "active_streams": len(monitor.active_streams),
                "active_allocations": len([a for a in manager.allocations.values() if a.is_active]),
                "metrics": monitor.get_statistics()
            }

            await websocket.send_json(update)
            await asyncio.sleep(2)

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.close()


# === DASHBOARD HTML ===

DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Zen MCP Adaptive Learning Dashboard</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
        }

        h1 {
            color: white;
            text-align: center;
            margin-bottom: 30px;
            font-size: 2.5em;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        }

        .dashboard-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
        }

        .card {
            background: white;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            transition: transform 0.3s ease;
        }

        .card:hover {
            transform: translateY(-5px);
        }

        .card h2 {
            color: #333;
            margin-bottom: 15px;
            font-size: 1.3em;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }

        .metric {
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            border-bottom: 1px solid #eee;
        }

        .metric:last-child {
            border-bottom: none;
        }

        .metric-label {
            color: #666;
            font-weight: 500;
        }

        .metric-value {
            color: #333;
            font-weight: bold;
        }

        .status-indicator {
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            margin-right: 5px;
        }

        .status-online {
            background-color: #4caf50;
        }

        .status-warning {
            background-color: #ff9800;
        }

        .status-error {
            background-color: #f44336;
        }

        .progress-bar {
            width: 100%;
            height: 20px;
            background-color: #e0e0e0;
            border-radius: 10px;
            overflow: hidden;
            margin-top: 10px;
        }

        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            transition: width 0.3s ease;
        }

        .chart-container {
            margin-top: 15px;
            height: 200px;
        }

        #connection-status {
            position: fixed;
            top: 20px;
            right: 20px;
            background: white;
            padding: 10px 20px;
            border-radius: 20px;
            box-shadow: 0 4px 10px rgba(0,0,0,0.1);
            font-weight: bold;
        }

        .loading {
            text-align: center;
            color: #999;
            padding: 20px;
        }

        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }

        .updating {
            animation: pulse 1s infinite;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 Zen MCP Adaptive Learning Dashboard</h1>

        <div id="connection-status">
            <span class="status-indicator status-online"></span>
            <span id="status-text">Connecting...</span>
        </div>

        <div class="dashboard-grid">
            <!-- System Status Card -->
            <div class="card">
                <h2>System Status</h2>
                <div id="system-status" class="loading">Loading...</div>
            </div>

            <!-- Performance Summary Card -->
            <div class="card">
                <h2>Performance Summary</h2>
                <div id="performance-summary" class="loading">Loading...</div>
            </div>

            <!-- Active Streams Card -->
            <div class="card">
                <h2>Active Streams</h2>
                <div id="active-streams" class="loading">Loading...</div>
            </div>

            <!-- Budget Status Card -->
            <div class="card">
                <h2>Token Budget</h2>
                <div id="budget-status" class="loading">Loading...</div>
            </div>

            <!-- Model Performance Card -->
            <div class="card">
                <h2>Model Performance (7 days)</h2>
                <div id="model-performance" class="loading">Loading...</div>
            </div>

            <!-- Streaming Statistics Card -->
            <div class="card">
                <h2>Streaming Statistics</h2>
                <div id="streaming-stats" class="loading">Loading...</div>
            </div>
        </div>
    </div>

    <script>
        // Dashboard state
        let ws = null;
        let updateInterval = null;

        // Initialize dashboard
        async function initDashboard() {
            // Set up WebSocket connection
            connectWebSocket();

            // Load initial data
            await loadAllData();

            // Set up periodic updates for non-websocket data
            updateInterval = setInterval(loadAllData, 10000);
        }

        // WebSocket connection
        function connectWebSocket() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${protocol}//${window.location.host}/ws/metrics`;

            ws = new WebSocket(wsUrl);

            ws.onopen = () => {
                updateConnectionStatus('Connected', 'online');
            };

            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                updateRealTimeMetrics(data);
            };

            ws.onerror = () => {
                updateConnectionStatus('Connection Error', 'error');
            };

            ws.onclose = () => {
                updateConnectionStatus('Disconnected', 'error');
                // Attempt to reconnect after 5 seconds
                setTimeout(connectWebSocket, 5000);
            };
        }

        // Update connection status
        function updateConnectionStatus(text, status) {
            const statusText = document.getElementById('status-text');
            const indicator = document.querySelector('#connection-status .status-indicator');

            statusText.textContent = text;
            indicator.className = `status-indicator status-${status}`;
        }

        // Load all dashboard data
        async function loadAllData() {
            try {
                await Promise.all([
                    loadSystemStatus(),
                    loadPerformanceSummary(),
                    loadActiveStreams(),
                    loadBudgetStatus(),
                    loadModelPerformance(),
                    loadStreamingStats()
                ]);
            } catch (error) {
                console.error('Failed to load dashboard data:', error);
            }
        }

        // Load system status
        async function loadSystemStatus() {
            try {
                const response = await fetch('/api/status');
                const data = await response.json();

                const container = document.getElementById('system-status');
                container.innerHTML = `
                    <div class="metric">
                        <span class="metric-label">Status</span>
                        <span class="metric-value">
                            <span class="status-indicator status-online"></span>
                            ${data.status}
                        </span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Adaptive Learning</span>
                        <span class="metric-value">${data.components.adaptive_learning.enabled ? 'Enabled' : 'Disabled'}</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Active Streams</span>
                        <span class="metric-value">${data.components.streaming_monitor.active_streams}</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Active Allocations</span>
                        <span class="metric-value">${data.components.budget_manager.active_allocations}</span>
                    </div>
                `;
            } catch (error) {
                console.error('Failed to load system status:', error);
            }
        }

        // Load performance summary
        async function loadPerformanceSummary() {
            try {
                const response = await fetch('/api/performance/summary');
                const data = await response.json();

                const container = document.getElementById('performance-summary');

                if (data.status === 'no_data') {
                    container.innerHTML = '<div class="loading">No data available yet</div>';
                    return;
                }

                container.innerHTML = `
                    <div class="metric">
                        <span class="metric-label">Total Selections</span>
                        <span class="metric-value">${data.recent_selections || 0}</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Avg Overall Score</span>
                        <span class="metric-value">${(data.avg_scores?.overall || 0).toFixed(2)}</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Avg Cost Efficiency</span>
                        <span class="metric-value">${(data.avg_scores?.cost_efficiency || 0).toFixed(2)}</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Avg Performance</span>
                        <span class="metric-value">${(data.avg_scores?.performance || 0).toFixed(2)}</span>
                    </div>
                `;
            } catch (error) {
                console.error('Failed to load performance summary:', error);
            }
        }

        // Load active streams
        async function loadActiveStreams() {
            try {
                const response = await fetch('/api/streaming/active');
                const data = await response.json();

                const container = document.getElementById('active-streams');

                if (data.count === 0) {
                    container.innerHTML = '<div class="loading">No active streams</div>';
                    return;
                }

                let html = '';
                data.streams.slice(0, 3).forEach(stream => {
                    html += `
                        <div class="metric">
                            <span class="metric-label">${stream.model_name}</span>
                            <span class="metric-value">${stream.current_tps.toFixed(1)} TPS</span>
                        </div>
                    `;
                });

                container.innerHTML = html;
            } catch (error) {
                console.error('Failed to load active streams:', error);
            }
        }

        // Load budget status
        async function loadBudgetStatus() {
            try {
                const response = await fetch('/api/budget/status?period=daily');
                const data = await response.json();

                const container = document.getElementById('budget-status');

                if (data.error) {
                    container.innerHTML = '<div class="loading">No budget configured</div>';
                    return;
                }

                const utilizationPercent = (data.utilization_rate * 100).toFixed(1);

                container.innerHTML = `
                    <div class="metric">
                        <span class="metric-label">Daily Budget</span>
                        <span class="metric-value">${data.total_tokens?.toLocaleString() || 0} tokens</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Used</span>
                        <span class="metric-value">${data.used_tokens?.toLocaleString() || 0} tokens</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Available</span>
                        <span class="metric-value">${data.available_tokens?.toLocaleString() || 0} tokens</span>
                    </div>
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: ${utilizationPercent}%"></div>
                    </div>
                    <div style="text-align: center; margin-top: 5px; color: #666;">
                        ${utilizationPercent}% utilized
                    </div>
                `;
            } catch (error) {
                console.error('Failed to load budget status:', error);
            }
        }

        // Load model performance
        async function loadModelPerformance() {
            try {
                const response = await fetch('/api/performance/models?days=7');
                const data = await response.json();

                const container = document.getElementById('model-performance');

                if (!data.summary || !data.summary.total_requests) {
                    container.innerHTML = '<div class="loading">No performance data yet</div>';
                    return;
                }

                container.innerHTML = `
                    <div class="metric">
                        <span class="metric-label">Total Requests</span>
                        <span class="metric-value">${data.summary.total_requests}</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Success Rate</span>
                        <span class="metric-value">${(data.summary.success_rate * 100).toFixed(1)}%</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Avg Cost</span>
                        <span class="metric-value">$${data.summary.avg_cost_usd?.toFixed(4) || 0}</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Avg Latency</span>
                        <span class="metric-value">${data.summary.avg_latency_ms?.toFixed(0) || 0}ms</span>
                    </div>
                `;
            } catch (error) {
                console.error('Failed to load model performance:', error);
            }
        }

        // Load streaming statistics
        async function loadStreamingStats() {
            try {
                const response = await fetch('/api/streaming/statistics');
                const data = await response.json();

                const container = document.getElementById('streaming-stats');

                if (data.status === 'no_data') {
                    container.innerHTML = '<div class="loading">No streaming data yet</div>';
                    return;
                }

                container.innerHTML = `
                    <div class="metric">
                        <span class="metric-label">Completed Streams</span>
                        <span class="metric-value">${data.completed_streams}</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Success Rate</span>
                        <span class="metric-value">${(data.success_rate * 100).toFixed(1)}%</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Avg TTFT</span>
                        <span class="metric-value">${data.ttft_stats?.avg_ms?.toFixed(0) || 0}ms</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Avg TPS</span>
                        <span class="metric-value">${data.tps_stats?.avg?.toFixed(1) || 0}</span>
                    </div>
                `;
            } catch (error) {
                console.error('Failed to load streaming stats:', error);
            }
        }

        // Update real-time metrics from WebSocket
        function updateRealTimeMetrics(data) {
            // Add updating animation
            const cards = document.querySelectorAll('.card');
            cards.forEach(card => {
                card.classList.add('updating');
                setTimeout(() => card.classList.remove('updating'), 300);
            });
        }

        // Initialize on page load
        document.addEventListener('DOMContentLoaded', initDashboard);

        // Cleanup on page unload
        window.addEventListener('beforeunload', () => {
            if (ws) ws.close();
            if (updateInterval) clearInterval(updateInterval);
        });
    </script>
</body>
</html>
"""


# === MAIN ENTRY POINT ===

def start_dashboard(host: str = "0.0.0.0", port: int = 8080):
    """
    Start the performance dashboard server.

    Args:
        host: Host to bind to
        port: Port to listen on
    """
    logger.info(f"Starting performance dashboard on http://{host}:{port}")

    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
        access_log=True
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Zen MCP Performance Dashboard")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8080, help="Port to listen on")

    args = parser.parse_args()

    start_dashboard(args.host, args.port)
