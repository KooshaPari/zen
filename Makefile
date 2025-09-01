.PHONY: dev-all-lite dev-all-lite-down dev-all-lite-logs dev-all-lite-restart dev-down down logs dev-db pg-wipe ingest-nomic

COMPOSE := docker compose
COMPOSE_LITE := COMPOSE_PROJECT_NAME=zen-lite docker compose

# Minimal dev stack for ARM64: NATS (with JetStream) + Redis
dev-all-lite:
	@echo "[dev] Starting NATS (JetStream) and Redis for ARM64..."
	$(COMPOSE_LITE) -f docker-compose.dev-lite.arm64.yml up -d --remove-orphans
	@echo "[dev] Waiting for services to be healthy..."
	@sleep 2
	@echo "[dev] Export these env vars in your shell for local runs:"
	@echo "    export NATS_SERVERS=nats://localhost:$${NATS_PORT:-4222}"
	@echo "    export REDIS_URL=redis://localhost:$${REDIS_PORT:-6379}/0"
	@echo "    export ENFORCE_ACL=0 ENABLE_HTTP_OAUTH=0; unset HTTP_HMAC_SECRET"
	@echo "[dev] NATS monitor: http://localhost:$${NATS_MONITOR_PORT:-8222}  | Redis: localhost:$${REDIS_PORT:-6379}"
	@echo "[dev] RabbitMQ: amqp://guest:guest@localhost:$${RABBITMQ_PORT:-5672}/ (mgmt http://localhost:$${RABBITMQ_MGMT_PORT:-15672})"

dev-all-lite-down:
	$(COMPOSE_LITE) -f docker-compose.dev-lite.arm64.yml down -v --remove-orphans

dev-all-lite-logs:
	$(COMPOSE_LITE) -f docker-compose.dev-lite.arm64.yml logs -f

dev-all-lite-restart: dev-all-lite-down dev-all-lite

# Aliases and additional helpers
dev-down: dev-all-lite-down
down: dev-all-lite-down
logs: dev-all-lite-logs

# Start only Postgres (for pgvector) from the shared dev compose
dev-db:
	@echo "[dev] Starting Postgres (pgvector)"
	$(COMPOSE) -f docker-compose.shared-dev.yml up -d --remove-orphans postgres

pg-wipe:
	@echo "[dev] Stopping Postgres and removing pgdata volume(s)"
	-$(COMPOSE) -f docker-compose.shared-dev.yml stop postgres
	-@VOLS=$$(docker volume ls -q | grep -E '(^|_)pgdata$$' | grep -E 'zen-mcp' || true); \
	if [ -z "$$VOLS" ]; then \
	  echo "No zen-mcp pgdata volumes found"; \
	else \
	  for v in $$VOLS; do echo "Removing volume $$v"; docker volume rm $$v || true; done; \
	fi; \
	echo "Done. Start DB with 'make dev-db'"

# Ingest current repo markdown into pgvector using Ollama embeddings (nomic-embed-text)
ingest-nomic:
	@echo "[ingest] Using EMBEDDINGS_PROVIDER=$${EMBEDDINGS_PROVIDER:-ollama} model=$${OLLAMA_EMBED_MODEL:-nomic-embed-text} dim=$${RAG_VECTOR_DIM:-768}"
	PYTHONPATH=. EMBEDDINGS_PROVIDER=$${EMBEDDINGS_PROVIDER:-ollama} \
		OLLAMA_URL=$${OLLAMA_URL:-http://localhost:11434} \
		OLLAMA_EMBED_MODEL=$${OLLAMA_EMBED_MODEL:-nomic-embed-text} \
		RAG_VECTOR_DIM=$${RAG_VECTOR_DIM:-768} \
		ZEN_PG_DSN=$${ZEN_PG_DSN:-postgresql://postgres:postgres@localhost:5432/zen} \
		python scripts/semtools_cli.py --work-dir . --glob "**/*.md" --collection knowledge --provider $${EMBEDDINGS_PROVIDER:-ollama} --model $${OLLAMA_EMBED_MODEL:-nomic-embed-text} --dim $${RAG_VECTOR_DIM:-768}

# Convenience targets for RabbitMQ + tests
.PHONY: dev-lite-up-rabbitmq test-unit test-http test-rabbitmq

dev-lite-up-rabbitmq:
	$(COMPOSE_LITE) -f docker-compose.dev-lite.arm64.yml up -d rabbitmq

test-unit:
	PYTHONPATH=. pytest -m "not integration" -q

test-http:
	PYTHONPATH=. pytest -q -k "http_endpoints_router_messaging or http_tasks_csv or models_catalog_endpoint"

test-rabbitmq:
	PYTHONPATH=. pytest -q tests/integration/test_rabbitmq_queue.py
