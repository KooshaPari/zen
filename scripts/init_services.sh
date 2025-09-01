#!/bin/bash

# Initialize Zen MCP Services
# This script sets up all required services for the adaptive learning system

set -e

echo "🚀 Initializing Zen MCP Services..."

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to wait for service
wait_for_service() {
    local service=$1
    local port=$2
    local max_attempts=30
    local attempt=0
    
    echo -n "⏳ Waiting for $service on port $port..."
    
    while ! nc -z localhost $port 2>/dev/null; do
        attempt=$((attempt + 1))
        if [ $attempt -eq $max_attempts ]; then
            echo -e " ${RED}✗${NC}"
            echo -e "${RED}Error: $service failed to start on port $port${NC}"
            return 1
        fi
        sleep 2
        echo -n "."
    done
    
    echo -e " ${GREEN}✓${NC}"
    return 0
}

# Check prerequisites
echo "📋 Checking prerequisites..."

if ! command_exists docker; then
    echo -e "${RED}Error: Docker is not installed${NC}"
    echo "Please install Docker from https://www.docker.com/get-started"
    exit 1
fi

if ! command_exists docker-compose; then
    echo -e "${RED}Error: Docker Compose is not installed${NC}"
    echo "Please install Docker Compose"
    exit 1
fi

echo -e "${GREEN}✓${NC} Prerequisites satisfied"

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p config/grafana/dashboards
mkdir -p config/grafana/datasources
mkdir -p logs
mkdir -p data

# Start core services first
echo "🐳 Starting core services..."
docker-compose -f docker-compose.full.yml up -d postgres redis

# Wait for PostgreSQL
wait_for_service "PostgreSQL" 5432

# Wait for Redis
wait_for_service "Redis" 6379

# Initialize database
echo "🗄️ Initializing database..."
sleep 5  # Give PostgreSQL extra time to fully initialize

# Check if database exists
if docker exec zen-postgres psql -U zen_user -lqt | cut -d \| -f 1 | grep -qw zen_mcp; then
    echo -e "${GREEN}✓${NC} Database already exists"
else
    echo "Creating database..."
    docker exec zen-postgres psql -U postgres -c "CREATE DATABASE zen_mcp;"
    docker exec zen-postgres psql -U postgres -d zen_mcp -c "CREATE EXTENSION IF NOT EXISTS pgvector;"
fi

# Run SQL scripts
echo "📝 Running SQL migrations..."
for sql_file in sql/*.sql; do
    if [ -f "$sql_file" ]; then
        echo "  Running $(basename $sql_file)..."
        docker exec -i zen-postgres psql -U zen_user -d zen_mcp < "$sql_file" 2>/dev/null || true
    fi
done

# Start embedding service
echo "🧠 Starting embedding service..."
docker-compose -f docker-compose.full.yml up -d embedding-service

# Start optional services
echo "📡 Starting optional services..."
read -p "Start NATS messaging? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    docker-compose -f docker-compose.full.yml up -d nats
    wait_for_service "NATS" 4222
fi

read -p "Start Kafka event streaming? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    docker-compose -f docker-compose.full.yml up -d zookeeper kafka
    wait_for_service "Kafka" 9092
fi

read -p "Start Ollama for local models? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    docker-compose -f docker-compose.full.yml up -d ollama
    wait_for_service "Ollama" 11434
    
    # Pull a default model
    echo "📥 Pulling default Ollama model (llama3.2)..."
    docker exec zen-ollama ollama pull llama3.2
fi

# Start monitoring services
read -p "Start monitoring services (Grafana, Prometheus)? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    docker-compose -f docker-compose.full.yml up -d prometheus grafana
    wait_for_service "Prometheus" 9090
    wait_for_service "Grafana" 3000
    echo -e "${GREEN}✓${NC} Grafana available at http://localhost:3000 (admin/admin)"
fi

# Start pgAdmin
read -p "Start pgAdmin for database management? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    docker-compose -f docker-compose.full.yml up -d pgadmin
    wait_for_service "pgAdmin" 5050
    echo -e "${GREEN}✓${NC} pgAdmin available at http://localhost:5050 (admin@zen.local/admin)"
fi

# Update .env file
echo "🔧 Updating environment configuration..."
cat >> .env << EOF

# Database Configuration
DATABASE_URL=postgresql://zen_user:zen_password@localhost:5432/zen_mcp
REDIS_URL=redis://localhost:6379

# Embedding Service
EMBEDDING_SERVICE_URL=http://localhost:8090

# Optional Services
NATS_URL=nats://zen_user:zen_password@localhost:4222
KAFKA_BOOTSTRAP_SERVERS=localhost:9092
OLLAMA_URL=http://localhost:11434

# Monitoring
PROMETHEUS_URL=http://localhost:9090
GRAFANA_URL=http://localhost:3000
EOF

# Start the dashboard
echo "🎯 Starting adaptive learning dashboard..."
cd dashboard && python performance_dashboard.py &
DASHBOARD_PID=$!

# Wait for dashboard
wait_for_service "Dashboard" 8080

# Show service status
echo ""
echo "✅ All services initialized successfully!"
echo ""
echo "📊 Service Status:"
docker-compose -f docker-compose.full.yml ps

echo ""
echo "🌐 Available Endpoints:"
echo "  • Dashboard: http://localhost:8080"
echo "  • PostgreSQL: localhost:5432 (zen_user/zen_password)"
echo "  • Redis: localhost:6379"
echo "  • Embedding Service: http://localhost:8090"

if docker-compose -f docker-compose.full.yml ps | grep -q zen-grafana; then
    echo "  • Grafana: http://localhost:3000 (admin/admin)"
fi

if docker-compose -f docker-compose.full.yml ps | grep -q zen-prometheus; then
    echo "  • Prometheus: http://localhost:9090"
fi

if docker-compose -f docker-compose.full.yml ps | grep -q zen-pgadmin; then
    echo "  • pgAdmin: http://localhost:5050 (admin@zen.local/admin)"
fi

if docker-compose -f docker-compose.full.yml ps | grep -q zen-nats; then
    echo "  • NATS: localhost:4222"
    echo "  • NATS Monitoring: http://localhost:8222"
fi

if docker-compose -f docker-compose.full.yml ps | grep -q zen-kafka; then
    echo "  • Kafka: localhost:9092"
fi

if docker-compose -f docker-compose.full.yml ps | grep -q zen-ollama; then
    echo "  • Ollama: http://localhost:11434"
fi

echo ""
echo "💡 Tips:"
echo "  • View logs: docker-compose -f docker-compose.full.yml logs -f [service]"
echo "  • Stop services: docker-compose -f docker-compose.full.yml down"
echo "  • Remove volumes: docker-compose -f docker-compose.full.yml down -v"
echo "  • Test routing: python test_adaptive_routing.py"
echo ""
echo "🎉 Setup complete! The adaptive learning system is ready."