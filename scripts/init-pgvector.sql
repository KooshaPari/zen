-- Initialize pgvector database for Zen MCP Server
-- This script sets up the vector storage schema with multi-tenant support

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";  -- For text search

-- Create schema for zen shared data
CREATE SCHEMA IF NOT EXISTS zen_shared;

-- Organizations table (top-level tenant)
CREATE TABLE IF NOT EXISTS zen_shared.organizations (
    org_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    org_name VARCHAR(255) UNIQUE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Projects table (under organizations)
CREATE TABLE IF NOT EXISTS zen_shared.projects (
    project_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    org_id UUID NOT NULL REFERENCES zen_shared.organizations(org_id) ON DELETE CASCADE,
    project_name VARCHAR(255) NOT NULL,
    repo_url VARCHAR(500),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'::jsonb,
    UNIQUE(org_id, project_name)
);

-- Work directories table (scoped access boundaries)
CREATE TABLE IF NOT EXISTS zen_shared.work_dirs (
    work_dir_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    project_id UUID NOT NULL REFERENCES zen_shared.projects(project_id) ON DELETE CASCADE,
    work_dir_path VARCHAR(500) NOT NULL,
    allowed_agents TEXT[],
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'::jsonb,
    UNIQUE(project_id, work_dir_path)
);

-- Vector collections table (namespaced vector storage)
CREATE TABLE IF NOT EXISTS zen_shared.vector_collections (
    collection_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    work_dir_id UUID NOT NULL REFERENCES zen_shared.work_dirs(work_dir_id) ON DELETE CASCADE,
    collection_name VARCHAR(100) NOT NULL,
    collection_type VARCHAR(50) NOT NULL CHECK (collection_type IN ('code', 'knowledge', 'memory', 'other')),
    dimension INTEGER NOT NULL DEFAULT 768,  -- Nomic Embed Text dimension (adjust if you change model)
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'::jsonb,
    UNIQUE(work_dir_id, collection_name)
);

-- Main vector storage table with tenant isolation
CREATE TABLE IF NOT EXISTS zen_shared.vectors (
    vector_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    collection_id UUID NOT NULL REFERENCES zen_shared.vector_collections(collection_id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    embedding vector(768),  -- Nomic Embed Text uses 768 dimensions
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    access_count INTEGER DEFAULT 0,
    last_accessed TIMESTAMP WITH TIME ZONE
);

-- Create indexes for efficient vector search
CREATE INDEX IF NOT EXISTS idx_vectors_embedding ON zen_shared.vectors 
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

CREATE INDEX IF NOT EXISTS idx_vectors_collection ON zen_shared.vectors(collection_id);
CREATE INDEX IF NOT EXISTS idx_vectors_metadata ON zen_shared.vectors USING GIN (metadata);
CREATE INDEX IF NOT EXISTS idx_vectors_content_trgm ON zen_shared.vectors USING GIN (content gin_trgm_ops);

-- Model performance history table (for enhanced model router)
CREATE TABLE IF NOT EXISTS zen_shared.model_performance (
    performance_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    work_dir_id UUID NOT NULL REFERENCES zen_shared.work_dirs(work_dir_id) ON DELETE CASCADE,
    model_name VARCHAR(255) NOT NULL,
    task_type VARCHAR(100) NOT NULL,
    complexity_tier VARCHAR(50),
    success BOOLEAN NOT NULL,
    latency_ms INTEGER,
    token_count INTEGER,
    error_message TEXT,
    task_embedding vector(768),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_model_perf_work_dir ON zen_shared.model_performance(work_dir_id);
CREATE INDEX IF NOT EXISTS idx_model_perf_model ON zen_shared.model_performance(model_name);
CREATE INDEX IF NOT EXISTS idx_model_perf_task_type ON zen_shared.model_performance(task_type);
CREATE INDEX IF NOT EXISTS idx_model_perf_embedding ON zen_shared.model_performance 
    USING ivfflat (task_embedding vector_cosine_ops)
    WITH (lists = 50);

-- Conversation memory table with embeddings
CREATE TABLE IF NOT EXISTS zen_shared.conversation_memory (
    memory_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    work_dir_id UUID NOT NULL REFERENCES zen_shared.work_dirs(work_dir_id) ON DELETE CASCADE,
    thread_id UUID NOT NULL,
    agent_id VARCHAR(255) NOT NULL,
    turn_number INTEGER NOT NULL,
    role VARCHAR(50) NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
    content TEXT NOT NULL,
    content_embedding vector(768),
    tool_name VARCHAR(100),
    files TEXT[],
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'::jsonb,
    UNIQUE(thread_id, turn_number)
);

CREATE INDEX IF NOT EXISTS idx_conv_memory_thread ON zen_shared.conversation_memory(thread_id);
CREATE INDEX IF NOT EXISTS idx_conv_memory_work_dir ON zen_shared.conversation_memory(work_dir_id);
CREATE INDEX IF NOT EXISTS idx_conv_memory_agent ON zen_shared.conversation_memory(agent_id);
CREATE INDEX IF NOT EXISTS idx_conv_memory_embedding ON zen_shared.conversation_memory 
    USING ivfflat (content_embedding vector_cosine_ops)
    WITH (lists = 100);

-- Agent memory table with hierarchical layers
CREATE TABLE IF NOT EXISTS zen_shared.agent_memory (
    memory_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    work_dir_id UUID NOT NULL REFERENCES zen_shared.work_dirs(work_dir_id) ON DELETE CASCADE,
    agent_id VARCHAR(255) NOT NULL,
    memory_type VARCHAR(50) NOT NULL CHECK (memory_type IN ('short_term', 'working', 'long_term', 'shared', 'procedural', 'episodic', 'semantic')),
    priority VARCHAR(20) NOT NULL CHECK (priority IN ('critical', 'high', 'medium', 'low', 'temporary')),
    content JSONB NOT NULL,
    content_embedding vector(768),
    tags TEXT[],
    related_memories UUID[],
    access_count INTEGER DEFAULT 0,
    last_accessed TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE,
    metadata JSONB DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_agent_memory_work_dir ON zen_shared.agent_memory(work_dir_id);
CREATE INDEX IF NOT EXISTS idx_agent_memory_agent ON zen_shared.agent_memory(agent_id);
CREATE INDEX IF NOT EXISTS idx_agent_memory_type ON zen_shared.agent_memory(memory_type);
CREATE INDEX IF NOT EXISTS idx_agent_memory_embedding ON zen_shared.agent_memory 
    USING ivfflat (content_embedding vector_cosine_ops)
    WITH (lists = 100);
CREATE INDEX IF NOT EXISTS idx_agent_memory_tags ON zen_shared.agent_memory USING GIN (tags);

-- Create helper functions for vector similarity search
CREATE OR REPLACE FUNCTION zen_shared.search_similar_vectors(
    p_collection_id UUID,
    p_query_embedding vector(768),
    p_limit INTEGER DEFAULT 10,
    p_threshold FLOAT DEFAULT 0.7
)
RETURNS TABLE (
    vector_id UUID,
    content TEXT,
    similarity FLOAT,
    metadata JSONB
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        v.vector_id,
        v.content,
        1 - (v.embedding <=> p_query_embedding) AS similarity,
        v.metadata
    FROM zen_shared.vectors v
    WHERE v.collection_id = p_collection_id
        AND 1 - (v.embedding <=> p_query_embedding) >= p_threshold
    ORDER BY v.embedding <=> p_query_embedding
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql;

-- Create function for hybrid search (vector + text)
CREATE OR REPLACE FUNCTION zen_shared.hybrid_search(
    p_collection_id UUID,
    p_query_embedding vector(768),
    p_text_query TEXT,
    p_limit INTEGER DEFAULT 10,
    p_vector_weight FLOAT DEFAULT 0.7
)
RETURNS TABLE (
    vector_id UUID,
    content TEXT,
    combined_score FLOAT,
    metadata JSONB
) AS $$
BEGIN
    RETURN QUERY
    WITH vector_scores AS (
        SELECT 
            v.vector_id,
            1 - (v.embedding <=> p_query_embedding) AS vector_score
        FROM zen_shared.vectors v
        WHERE v.collection_id = p_collection_id
    ),
    text_scores AS (
        SELECT 
            v.vector_id,
            ts_rank_cd(to_tsvector('english', v.content), plainto_tsquery('english', p_text_query)) AS text_score
        FROM zen_shared.vectors v
        WHERE v.collection_id = p_collection_id
            AND v.content @@ plainto_tsquery('english', p_text_query)
    )
    SELECT 
        v.vector_id,
        v.content,
        (COALESCE(vs.vector_score, 0) * p_vector_weight + 
         COALESCE(ts.text_score, 0) * (1 - p_vector_weight)) AS combined_score,
        v.metadata
    FROM zen_shared.vectors v
    LEFT JOIN vector_scores vs ON v.vector_id = vs.vector_id
    LEFT JOIN text_scores ts ON v.vector_id = ts.vector_id
    WHERE v.collection_id = p_collection_id
        AND (vs.vector_score IS NOT NULL OR ts.text_score IS NOT NULL)
    ORDER BY combined_score DESC
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql;

-- Grant appropriate permissions
GRANT USAGE ON SCHEMA zen_shared TO zen_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA zen_shared TO zen_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA zen_shared TO zen_user;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA zen_shared TO zen_user;
