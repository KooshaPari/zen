-- Initialize Zen MCP Database
-- This script sets up all required schemas and tables

-- Create extensions
CREATE EXTENSION IF NOT EXISTS pgvector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Create user if not exists (for local development)
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_user WHERE usename = 'zen_user') THEN
        CREATE USER zen_user WITH PASSWORD 'zen_password';
    END IF;
END $$;

-- Grant privileges
GRANT ALL PRIVILEGES ON DATABASE zen_mcp TO zen_user;
GRANT CREATE ON SCHEMA public TO zen_user;

-- Create schemas
CREATE SCHEMA IF NOT EXISTS zen_adaptive;
CREATE SCHEMA IF NOT EXISTS zen_performance;
CREATE SCHEMA IF NOT EXISTS zen_embeddings;
CREATE SCHEMA IF NOT EXISTS zen_budget;
CREATE SCHEMA IF NOT EXISTS zen_conversation;

-- Grant schema privileges
GRANT ALL ON SCHEMA zen_adaptive TO zen_user;
GRANT ALL ON SCHEMA zen_performance TO zen_user;
GRANT ALL ON SCHEMA zen_embeddings TO zen_user;
GRANT ALL ON SCHEMA zen_budget TO zen_user;
GRANT ALL ON SCHEMA zen_conversation TO zen_user;

-- Set search path
SET search_path TO zen_adaptive, zen_performance, zen_embeddings, zen_budget, zen_conversation, public;

-- Import individual schema files (these will be created next)
\i /docker-entrypoint-initdb.d/01_performance_schema.sql
\i /docker-entrypoint-initdb.d/02_adaptive_learning_schema.sql
\i /docker-entrypoint-initdb.d/03_embeddings_schema.sql
\i /docker-entrypoint-initdb.d/04_budget_schema.sql
\i /docker-entrypoint-initdb.d/05_conversation_schema.sql

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_performance_created ON zen_performance.model_performance(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_adaptive_predictions_created ON zen_adaptive.predictions(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_budget_allocations_created ON zen_budget.token_allocations(created_at DESC);

-- Set up row-level security (optional)
ALTER TABLE zen_adaptive.predictions ENABLE ROW LEVEL SECURITY;
ALTER TABLE zen_performance.model_performance ENABLE ROW LEVEL SECURITY;
ALTER TABLE zen_budget.token_budgets ENABLE ROW LEVEL SECURITY;

-- Grant final permissions
GRANT ALL ON ALL TABLES IN SCHEMA zen_adaptive TO zen_user;
GRANT ALL ON ALL SEQUENCES IN SCHEMA zen_adaptive TO zen_user;
GRANT ALL ON ALL FUNCTIONS IN SCHEMA zen_adaptive TO zen_user;

GRANT ALL ON ALL TABLES IN SCHEMA zen_performance TO zen_user;
GRANT ALL ON ALL SEQUENCES IN SCHEMA zen_performance TO zen_user;
GRANT ALL ON ALL FUNCTIONS IN SCHEMA zen_performance TO zen_user;

GRANT ALL ON ALL TABLES IN SCHEMA zen_embeddings TO zen_user;
GRANT ALL ON ALL SEQUENCES IN SCHEMA zen_embeddings TO zen_user;
GRANT ALL ON ALL FUNCTIONS IN SCHEMA zen_embeddings TO zen_user;

GRANT ALL ON ALL TABLES IN SCHEMA zen_budget TO zen_user;
GRANT ALL ON ALL SEQUENCES IN SCHEMA zen_budget TO zen_user;
GRANT ALL ON ALL FUNCTIONS IN SCHEMA zen_budget TO zen_user;

GRANT ALL ON ALL TABLES IN SCHEMA zen_conversation TO zen_user;
GRANT ALL ON ALL SEQUENCES IN SCHEMA zen_conversation TO zen_user;
GRANT ALL ON ALL FUNCTIONS IN SCHEMA zen_conversation TO zen_user;

-- Verify setup
DO $$
BEGIN
    RAISE NOTICE 'Database initialization complete';
    RAISE NOTICE 'Schemas created: zen_adaptive, zen_performance, zen_embeddings, zen_budget, zen_conversation';
END $$;