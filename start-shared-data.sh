#!/bin/bash

# Startup script for Zen MCP Shared Data Infrastructure
# Automatically detects architecture and uses appropriate configuration

set -e

echo "🚀 Starting Zen MCP Shared Data Infrastructure..."

# Detect architecture
ARCH=$(uname -m)
echo "Detected architecture: $ARCH"

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker Desktop first."
    exit 1
fi

# Choose appropriate compose file
if [[ "$ARCH" == "arm64" ]] || [[ "$ARCH" == "aarch64" ]]; then
    echo "Using ARM64 optimized configuration..."
    COMPOSE_FILE="docker-compose.shared-data-arm64.yml"
    
    # Build the Python embedding service for ARM64
    echo "Building Python embedding service for ARM64..."
    docker build -f Dockerfile.embeddings -t zen-embeddings:latest .
    
    # Add the embedding service to the compose
    cat >> docker-compose.shared-data-arm64.yml << 'EOF'

  # Python-based embedding service for ARM64
  embeddings:
    image: zen-embeddings:latest
    container_name: zen-embeddings
    ports:
      - "8090:8090"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8090/health"]
      interval: 10s
      timeout: 5s
      retries: 5
    networks:
      - zen-shared-data
EOF
    
else
    echo "Using x86_64 configuration..."
    COMPOSE_FILE="docker-compose.shared-data.yml"
fi

# Create required directories
mkdir -p scripts

# Start services
echo "Starting services with $COMPOSE_FILE..."
docker-compose -f $COMPOSE_FILE up -d

# Wait for services to be healthy
echo "Waiting for services to be healthy..."
sleep 5

# Check service status
echo ""
echo "📊 Service Status:"
echo "=================="

# PostgreSQL
if docker exec zen-postgres-vector pg_isready -U zen_user -d zen_vector > /dev/null 2>&1; then
    echo "✅ PostgreSQL with pgvector: Running"
else
    echo "⚠️  PostgreSQL with pgvector: Starting..."
fi

# Redis
if docker exec zen-redis redis-cli ping > /dev/null 2>&1; then
    echo "✅ Redis: Running"
else
    echo "⚠️  Redis: Starting..."
fi

# ArangoDB
if curl -s http://localhost:8529/_api/version > /dev/null 2>&1; then
    echo "✅ ArangoDB: Running"
else
    echo "⚠️  ArangoDB: Starting..."
fi

# Embeddings
if curl -s http://localhost:8090/health > /dev/null 2>&1; then
    echo "✅ Embeddings Service: Running"
else
    echo "⚠️  Embeddings Service: Starting..."
fi

# Redpanda
if docker exec zen-redpanda rpk cluster health 2>/dev/null | grep -q "Healthy: true"; then
    echo "✅ Redpanda (Kafka): Running"
else
    echo "⚠️  Redpanda (Kafka): Starting..."
fi

# MinIO
if docker exec zen-minio mc ready local > /dev/null 2>&1; then
    echo "✅ MinIO: Running"
else
    echo "⚠️  MinIO: Starting..."
fi

echo ""
echo "🎉 Shared Data Infrastructure is starting up!"
echo ""
echo "Access points:"
echo "============="
echo "PostgreSQL:    localhost:5432 (user: zen_user, pass: zen_secure_pass_2025)"
echo "ArangoDB:      http://localhost:8529 (user: root, pass: zen_arango_2025)"
echo "Redis:         localhost:6379"
echo "Embeddings:    http://localhost:8090"
echo "Redpanda:      localhost:19092 (Kafka API)"
echo "MinIO:         http://localhost:9001 (Console - user: zen_minio_admin, pass: zen_minio_secure_2025)"
echo ""
echo "To stop all services: docker-compose -f $COMPOSE_FILE down"
echo "To view logs: docker-compose -f $COMPOSE_FILE logs -f [service-name]"