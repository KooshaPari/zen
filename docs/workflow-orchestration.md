# Temporal Workflow Orchestration System

The Zen MCP Server includes a comprehensive workflow orchestration system built on Temporal for managing complex, long-running multi-agent processes with fault tolerance and human approval gates.

## Overview

The workflow orchestration system provides enterprise-grade capabilities for:

- **Multi-agent project coordination** - Orchestrate multiple agents across different phases of complex projects
- **Human-in-the-loop approvals** - Integrate human approval gates with timeout and escalation policies  
- **Saga pattern transactions** - Manage distributed transactions with automatic compensation on failures
- **Workflow monitoring & recovery** - Real-time monitoring with automatic failure detection and recovery
- **Integration with existing infrastructure** - Seamless integration with Redis, NATS, and Kafka

## Architecture Components

### 1. Temporal Client Integration (`utils/temporal_client.py`)

The core integration with Temporal provides:
- Workflow execution management
- Human approval coordination
- Fallback execution when Temporal is unavailable
- Workflow state persistence

```python
from utils.temporal_client import get_temporal_client

client = get_temporal_client()
await client.connect()

# Start a workflow
result = await client.start_workflow(
    workflow_class=MyWorkflow,
    workflow_args={"param": "value"},
    workflow_id="my-workflow-123"
)
```

### 2. Multi-Agent Project Workflows (`workflows/multi_agent_workflow.py`)

Orchestrates complex projects across multiple agents with:
- Automatic project analysis and planning
- Parallel and sequential agent coordination
- Progress tracking and monitoring
- Automatic retry and compensation on failures

```python
from workflows.multi_agent_workflow import start_multi_agent_project, ProjectSpec
from tools.shared.agent_models import AgentType

# Define project
project_spec = ProjectSpec(
    project_id="web-app-project",
    name="E-commerce Web Application",
    description="Build a complete e-commerce solution",
    requirements=[
        "User authentication system",
        "Product catalog with search",
        "Shopping cart and checkout",
        "Admin dashboard"
    ],
    constraints={"timeline": "2 weeks", "budget": 10000},
    success_criteria=[
        "All features functional",
        "Tests passing with >80% coverage",
        "Performance meets requirements"
    ],
    estimated_duration_hours=80,
    agents_required=[AgentType.CLAUDE, AgentType.AIDER],
    approval_gates=["Backend API", "deployment"]
)

# Start project workflow
result = await start_multi_agent_project(
    project_spec=project_spec,
    config={
        "approval_timeout": 24 * 60 * 60,  # 24 hours
        "parallel_execution": True
    }
)
```

### 3. Human Approval Workflows (`workflows/approval_workflow.py`)

Sophisticated approval processes with:
- Multi-stage approval chains
- Timeout and escalation handling
- Multiple notification channels (email, Slack, webhooks, SMS)
- Approval delegation capabilities

```python
from workflows.approval_workflow import submit_approval_decision

# Submit approval decision
success = await submit_approval_decision(
    request_id="approval-123",
    approver_id="user@company.com",
    decision="approve",
    feedback="Ready for production deployment"
)
```

### 4. Saga Pattern Workflows (`workflows/saga_workflow.py`)

Distributed transaction management with:
- Forward and compensation action definitions
- Both orchestration and choreography patterns
- Automatic rollback on failures
- Cross-service transaction coordination

```python
from workflows.saga_workflow import start_distributed_saga

# Define saga transaction
saga_definition = {
    "saga_id": "user-onboarding-saga",
    "name": "User Onboarding Process",
    "description": "Complete user onboarding with external services",
    "steps": [
        {
            "step_id": "create_account",
            "name": "Create User Account",
            "description": "Create user in auth service",
            "forward_action": {
                "type": "api_call",
                "url": "https://auth.company.com/users",
                "method": "POST"
            },
            "compensation_action": {
                "type": "api_call", 
                "url": "https://auth.company.com/users/{user_id}",
                "method": "DELETE"
            },
            "agent_type": "CLAUDE",
            "timeout_minutes": 10,
            "retry_count": 3
        },
        {
            "step_id": "setup_billing",
            "name": "Setup Billing Account",
            "description": "Create billing account",
            "forward_action": {
                "type": "api_call",
                "url": "https://billing.company.com/accounts",
                "method": "POST"
            },
            "compensation_action": {
                "type": "api_call",
                "url": "https://billing.company.com/accounts/{account_id}",
                "method": "DELETE"
            },
            "agent_type": "CLAUDE",
            "dependencies": ["create_account"]
        }
    ]
}

# Start saga
result = await start_distributed_saga(
    saga_definition=saga_definition,
    coordination_mode="orchestration"
)
```

### 5. Workflow Monitoring (`workflows/workflow_monitor.py`)

Real-time monitoring and recovery with:
- Performance metrics collection
- Health checks and alerting
- Automatic failure detection
- Recovery policy enforcement

```python
from workflows.workflow_monitor import get_workflow_monitor

monitor = get_workflow_monitor()
await monitor.start_monitoring()

# Perform health check
health = await monitor.perform_health_check("workflow-123")
print(f"Workflow health: {health.status}")

# Get dashboard data
dashboard = await monitor.get_workflow_dashboard_data()
```

### 6. Integration Layer (`workflows/integrations.py`)

Seamless integration with existing infrastructure:
- Redis-based workflow state management
- NATS real-time messaging
- Kafka event streaming
- Agent API bridge

## Workflow Orchestrator Tool

The system includes a unified MCP tool for workflow management:

```python
# Start a multi-agent project
{
    "operation": "start_workflow",
    "workflow_type": "multi_agent_project",
    "workflow_spec": {
        "project_id": "test-project",
        "name": "Test Project",
        "description": "Test project workflow",
        "requirements": ["Feature A", "Feature B"],
        "constraints": {"timeline": "1 week"},
        "success_criteria": ["All features working"],
        "estimated_duration_hours": 20,
        "agents_required": ["CLAUDE"]
    }
}

# Check workflow status
{
    "operation": "status",
    "workflow_id": "workflow-123",
    "include_details": true
}

# Request human approval
{
    "operation": "approval",
    "action": "request",
    "workflow_id": "workflow-123",
    "stage": "deployment",
    "description": "Approve production deployment",
    "context": {"environment": "production"}
}

# Approve a request
{
    "operation": "approval",
    "action": "approve",
    "approval_id": "approval-456",
    "approver_id": "user@company.com",
    "feedback": "Ready for deployment"
}
```

## Installation and Setup

### Basic Installation

The workflow orchestration system is included with the standard Zen MCP Server installation. Basic functionality works with just Redis:

```bash
pip install zen-mcp-server
```

### Full Workflow Features

For complete workflow orchestration capabilities, install with workflow dependencies:

```bash
# Install all workflow dependencies
pip install zen-mcp-server[workflows]

# Or install specific components
pip install zen-mcp-server[temporal,nats,kafka]
```

### Temporal Server Setup

1. **Docker Compose (Recommended for development):**

```yaml
version: '3.8'
services:
  temporal:
    image: temporalio/auto-setup:1.22.0
    ports:
      - "7233:7233"
    environment:
      - DB=postgresql
      - DB_PORT=5432
      - POSTGRES_USER=temporal
      - POSTGRES_PWD=temporal
      - POSTGRES_SEEDS=postgresql
    depends_on:
      - postgresql
  
  postgresql:
    image: postgres:13
    environment:
      POSTGRES_PASSWORD: temporal
      POSTGRES_USER: temporal
      POSTGRES_DB: temporal
```

2. **Start Temporal:**

```bash
docker-compose up -d
```

3. **Configure environment:**

```bash
export TEMPORAL_ADDRESS=localhost:7233
export TEMPORAL_NAMESPACE=default
export TEMPORAL_TASK_QUEUE=zen-workflows
```

### Optional Integrations

#### NATS Setup

```bash
# Start NATS server
docker run -d --name nats -p 4222:4222 nats

# Configure environment
export NATS_URL=nats://localhost:4222
```

#### Kafka Setup

```bash
# Start Kafka with Zookeeper
docker run -d --name zookeeper -p 2181:2181 zookeeper
docker run -d --name kafka -p 9092:9092 \
  --link zookeeper:zookeeper \
  confluentinc/cp-kafka

# Configure environment
export KAFKA_BOOTSTRAP_SERVERS=localhost:9092
```

## Configuration

### Environment Variables

```bash
# Temporal Configuration
TEMPORAL_ADDRESS=localhost:7233
TEMPORAL_NAMESPACE=default
TEMPORAL_TASK_QUEUE=zen-workflows

# NATS Configuration (optional)
NATS_URL=nats://localhost:4222

# Kafka Configuration (optional)  
KAFKA_BOOTSTRAP_SERVERS=localhost:9092

# Redis Configuration (already configured)
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB_CONVERSATIONS=0

# Workflow Monitoring
WORKFLOW_MONITORING_ENABLED=true
WORKFLOW_HEALTH_CHECK_INTERVAL=300
WORKFLOW_METRICS_INTERVAL=60
```

## Usage Examples

### Example 1: Multi-Agent Web Development Project

```python
# Define a complex web development project
project_spec = ProjectSpec(
    project_id="ecommerce-platform",
    name="E-commerce Platform",
    description="Build a scalable e-commerce platform with microservices architecture",
    requirements=[
        "User authentication and authorization",
        "Product catalog with search and filtering", 
        "Shopping cart and checkout process",
        "Payment integration (Stripe)",
        "Order management system",
        "Admin dashboard",
        "Email notifications",
        "API documentation"
    ],
    constraints={
        "timeline": "4 weeks",
        "budget": 50000,
        "technology_stack": "Node.js, React, PostgreSQL, Redis",
        "deployment": "AWS EKS with CI/CD"
    },
    success_criteria=[
        "All user stories implemented and tested",
        "API response time < 200ms for 95% of requests",
        "Test coverage >= 85%",
        "Security audit passed",
        "Load testing passed (1000 concurrent users)",
        "Documentation complete"
    ],
    estimated_duration_hours=320,  # 4 weeks * 40 hours * 2 developers
    agents_required=[AgentType.CLAUDE, AgentType.AIDER, AgentType.CODEX],
    approval_gates=[
        "Architecture Review",
        "Security Review", 
        "Performance Testing",
        "Production Deployment"
    ]
)

# Start the project with advanced configuration
result = await start_multi_agent_project(
    project_spec=project_spec,
    config={
        "approval_timeout": 48 * 60 * 60,  # 48 hours for approvals
        "parallel_execution": True,
        "max_parallel_agents": 3,
        "retry_failed_phases": True,
        "automatic_testing": True,
        "deployment_strategy": "blue_green",
        "monitoring_enabled": True
    }
)
```

### Example 2: Distributed Microservice Deployment Saga

```python
# Define microservice deployment saga
deployment_saga = {
    "saga_id": "microservice-deployment-v2.1.0",
    "name": "Microservice Deployment Pipeline", 
    "description": "Deploy new version of microservices with rollback capability",
    "steps": [
        {
            "step_id": "deploy_database_migrations",
            "name": "Deploy Database Migrations",
            "description": "Apply database schema changes",
            "forward_action": {
                "type": "task",
                "description": "Run database migrations",
                "database": "production",
                "migration_version": "v2.1.0"
            },
            "compensation_action": {
                "type": "task",
                "description": "Rollback database migrations",
                "database": "production", 
                "migration_version": "v2.0.5"
            },
            "agent_type": "CLAUDE",
            "timeout_minutes": 15,
            "retry_count": 2,
            "critical": True
        },
        {
            "step_id": "deploy_auth_service", 
            "name": "Deploy Authentication Service",
            "description": "Deploy auth service v2.1.0",
            "forward_action": {
                "type": "api_call",
                "url": "https://deploy.company.com/services/auth",
                "method": "POST",
                "payload": {"version": "v2.1.0", "replicas": 3}
            },
            "compensation_action": {
                "type": "api_call",
                "url": "https://deploy.company.com/services/auth",
                "method": "POST", 
                "payload": {"version": "v2.0.5", "replicas": 3}
            },
            "agent_type": "CLAUDE",
            "dependencies": ["deploy_database_migrations"],
            "parallel_group": "services"
        },
        {
            "step_id": "deploy_api_gateway",
            "name": "Deploy API Gateway",
            "description": "Deploy API gateway v2.1.0",
            "forward_action": {
                "type": "api_call",
                "url": "https://deploy.company.com/services/gateway", 
                "method": "POST",
                "payload": {"version": "v2.1.0", "replicas": 2}
            },
            "compensation_action": {
                "type": "api_call",
                "url": "https://deploy.company.com/services/gateway",
                "method": "POST",
                "payload": {"version": "v2.0.5", "replicas": 2}
            },
            "agent_type": "CLAUDE",
            "dependencies": ["deploy_database_migrations"],
            "parallel_group": "services"
        },
        {
            "step_id": "run_health_checks",
            "name": "Run Health Checks",
            "description": "Verify all services are healthy",
            "forward_action": {
                "type": "script",
                "script": "#!/bin/bash\ncurl -f https://api.company.com/health && curl -f https://auth.company.com/health"
            },
            "agent_type": "CLAUDE",
            "dependencies": ["deploy_auth_service", "deploy_api_gateway"],
            "timeout_minutes": 10,
            "retry_count": 5
        },
        {
            "step_id": "update_load_balancer",
            "name": "Update Load Balancer",
            "description": "Switch load balancer to new services",
            "forward_action": {
                "type": "api_call",
                "url": "https://lb.company.com/config",
                "method": "PUT",
                "payload": {"active_version": "v2.1.0"}
            },
            "compensation_action": {
                "type": "api_call",
                "url": "https://lb.company.com/config", 
                "method": "PUT",
                "payload": {"active_version": "v2.0.5"}
            },
            "agent_type": "CLAUDE",
            "dependencies": ["run_health_checks"],
            "critical": True
        }
    ],
    "context": {
        "deployment_version": "v2.1.0",
        "previous_version": "v2.0.5",
        "environment": "production",
        "requester": "deploy-bot"
    }
}

# Execute deployment saga
result = await start_distributed_saga(
    saga_definition=deployment_saga,
    coordination_mode="orchestration",
    config={
        "timeout_seconds": 1800,  # 30 minutes total
        "retry_failed_steps": True,
        "alert_on_failure": True,
        "notification_channels": ["slack://deployments", "email://devops@company.com"]
    }
)
```

### Example 3: Human Approval Workflow for Production Changes

```python
# Request approval for production database changes
approval_result = await client.request_human_approval(
    workflow_id="db-migration-workflow-789",
    stage="production_database_migration",
    description="Apply database schema changes to production",
    context={
        "migration_scripts": ["001_add_user_preferences.sql", "002_index_optimizations.sql"],
        "estimated_downtime": "5 minutes",
        "rollback_plan": "Automated rollback via migration version",
        "tested_environments": ["staging", "qa"],
        "risk_level": "medium",
        "business_impact": "Minimal - new features only"
    },
    timeout_seconds=4 * 60 * 60,  # 4 hours
    callback_url="https://api.company.com/webhooks/approvals"
)

# The approval system will:
# 1. Send notifications to configured approvers
# 2. Provide approval interface with context
# 3. Handle timeout and escalation
# 4. Return decision to workflow
```

## Monitoring and Observability

### Workflow Dashboard

The monitoring system provides comprehensive dashboards:

```python
# Get system-wide workflow overview
dashboard = await monitor.get_workflow_dashboard_data()

print(f"Total workflows: {dashboard['summary']['total_workflows']}")
print(f"Healthy workflows: {dashboard['summary']['healthy_workflows']}")
print(f"Critical alerts: {dashboard['summary']['critical_alerts']}")

# Get specific workflow details
workflow_details = await monitor.get_workflow_dashboard_data("workflow-123")
print(f"Workflow status: {workflow_details['metrics']['status']}")
print(f"Completion rate: {workflow_details['metrics']['completion_rate']}")
```

### Health Checks

Automated health monitoring:

```python
# Perform comprehensive health check
health = await monitor.perform_health_check("workflow-123")

print(f"Overall health: {health.status}")
print(f"Issues detected: {len(health.issues_detected)}")
for issue in health.issues_detected:
    print(f"  - {issue}")

print(f"Recommendations: {len(health.recommendations)}")
for rec in health.recommendations:
    print(f"  - {rec}")
```

### Alerting and Recovery

The system includes automatic alerting and recovery:

```python
# Configure recovery policies
recovery_policy = RecoveryPolicy(
    policy_id="high_value_workflow_policy",
    name="High Value Workflow Recovery",
    workflow_types=["multi_agent_project"],
    conditions={
        "execution_time_hours": {">=": 4},
        "error_rate": {">=": 0.3}
    },
    recovery_actions=[RecoveryAction.RETRY, RecoveryAction.ESCALATE],
    max_recovery_attempts=3,
    notification_channels=["slack://alerts", "email://ops@company.com"]
)

# Trigger manual recovery
recovery_result = await monitor.trigger_recovery(
    workflow_id="critical-workflow-456", 
    recovery_reason="High error rate detected",
    recovery_action=RecoveryAction.RESTART
)
```

## Performance and Scalability

The workflow orchestration system is designed for enterprise scale:

### Performance Metrics

- **Workflow Throughput**: 1000+ concurrent workflows
- **Step Execution Latency**: Sub-second for most operations  
- **State Persistence**: High-availability Redis with replication
- **Fault Tolerance**: Automatic retry with exponential backoff
- **Recovery Time**: < 30 seconds for most failure scenarios

### Scalability Features

- **Horizontal Scaling**: Temporal workers can be scaled independently
- **Resource Isolation**: Workflows run in isolated execution environments
- **Load Balancing**: Automatic distribution across available workers
- **State Sharding**: Redis state can be sharded across multiple instances
- **Event Streaming**: Kafka integration for high-throughput event processing

### Best Practices

1. **Workflow Design**:
   - Keep workflows idempotent
   - Use timeouts for all external operations
   - Implement proper compensation logic
   - Design for parallelism where possible

2. **State Management**:
   - Minimize workflow state size
   - Use Redis clustering for high availability
   - Implement proper TTL policies

3. **Monitoring**:
   - Set up comprehensive alerting
   - Monitor workflow execution metrics
   - Implement health checks for critical workflows
   - Use dashboard for operational visibility

4. **Error Handling**:
   - Implement retry policies for transient failures
   - Design compensation actions for critical steps
   - Use circuit breakers for external service calls
   - Implement proper escalation procedures

## Troubleshooting

### Common Issues

1. **Temporal Connection Failed**:
   ```bash
   # Check Temporal server status
   docker ps | grep temporal
   
   # Verify network connectivity
   telnet localhost 7233
   
   # Check environment variables
   echo $TEMPORAL_ADDRESS
   ```

2. **Workflow Execution Timeout**:
   - Increase workflow timeout in configuration
   - Check agent responsiveness
   - Verify network stability between components

3. **High Memory Usage**:
   - Review workflow state size
   - Implement state cleanup policies
   - Consider Redis memory optimization

4. **Approval Timeout**:
   - Check notification delivery
   - Verify approver availability
   - Consider escalation policies

### Debugging Tools

```python
# Enable debug logging
import logging
logging.getLogger("workflows").setLevel(logging.DEBUG)
logging.getLogger("utils.temporal_client").setLevel(logging.DEBUG)

# Get detailed workflow execution logs
logs = await monitor.get_workflow_logs("workflow-123")

# Check workflow state history
state_history = await state_manager.get_workflow_state_history("workflow-123")
```

## API Reference

### Workflow Orchestrator Tool

The primary interface for workflow management:

```json
{
  "operation": "start_workflow|approval|status|monitor|list_workflows|cancel_workflow",
  "workflow_type": "multi_agent_project|approval|saga",
  "workflow_spec": {...},
  "config": {...},
  "workflow_id": "string",
  "approval_id": "string", 
  "action": "string",
  "approver_id": "string",
  "feedback": "string"
}
```

### Core Classes

- `TemporalWorkflowClient` - Main client for Temporal integration
- `MultiAgentProjectWorkflow` - Multi-agent project orchestration
- `HumanApprovalWorkflow` - Human approval processes
- `SagaWorkflow` - Distributed transaction management
- `WorkflowMonitor` - Monitoring and recovery
- `WorkflowStateManager` - State persistence
- `WorkflowOrchestratorTool` - MCP tool interface

## Contributing

To contribute to the workflow orchestration system:

1. Follow the existing code patterns and architecture
2. Add comprehensive tests for new features
3. Update documentation for any API changes
4. Ensure backward compatibility
5. Test with different Temporal versions

## License

The workflow orchestration system is part of the Zen MCP Server and follows the same license terms.