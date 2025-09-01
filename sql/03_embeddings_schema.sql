-- Embeddings Schema for Semantic Search and Similarity
-- Uses pgvector for efficient vector operations

SET search_path TO zen_embeddings, public;

-- Document embeddings for semantic search
CREATE TABLE IF NOT EXISTS document_embeddings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Document identification
    document_id VARCHAR(255) NOT NULL,
    document_type VARCHAR(50) NOT NULL, -- 'code', 'prompt', 'response', 'file'
    document_hash VARCHAR(64) NOT NULL,
    
    -- Content
    content TEXT NOT NULL,
    content_tokens INTEGER,
    
    -- Embedding
    embedding vector(384), -- BGE-small dimension
    embedding_model VARCHAR(100) DEFAULT 'BAAI/bge-small-en-v1.5',
    
    -- Metadata
    metadata JSONB,
    task_type VARCHAR(100),
    model_name VARCHAR(255),
    quality_score FLOAT,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    accessed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    access_count INTEGER DEFAULT 1,
    
    -- Indexes
    INDEX idx_emb_document (document_id),
    INDEX idx_emb_type (document_type),
    INDEX idx_emb_hash (document_hash),
    INDEX idx_emb_created (created_at DESC)
);

-- Create vector index for similarity search
CREATE INDEX idx_emb_vector ON document_embeddings 
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- Task pattern embeddings for similarity matching
CREATE TABLE IF NOT EXISTS task_embeddings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Task identification
    task_type VARCHAR(100) NOT NULL,
    task_description TEXT,
    
    -- Embedding
    embedding vector(384),
    
    -- Performance characteristics
    best_model VARCHAR(255),
    avg_tokens INTEGER,
    avg_cost DECIMAL(10, 6),
    avg_latency_ms INTEGER,
    success_rate FLOAT,
    
    -- Usage
    match_count INTEGER DEFAULT 0,
    last_matched TIMESTAMP WITH TIME ZONE,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create vector index for task similarity
CREATE INDEX idx_task_vector ON task_embeddings 
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 50);

-- Conversation embeddings for context retrieval
CREATE TABLE IF NOT EXISTS conversation_embeddings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Conversation identification
    conversation_id VARCHAR(255) NOT NULL,
    turn_number INTEGER NOT NULL,
    role VARCHAR(20) NOT NULL, -- 'user', 'assistant', 'system'
    
    -- Content
    content TEXT NOT NULL,
    embedding vector(384),
    
    -- Context
    tool_name VARCHAR(100),
    model_name VARCHAR(255),
    
    -- Metadata
    metadata JSONB,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Constraints
    UNIQUE(conversation_id, turn_number)
);

-- Create vector index for conversation search
CREATE INDEX idx_conv_vector ON conversation_embeddings 
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 50);

-- Function to find similar documents
CREATE OR REPLACE FUNCTION find_similar_documents(
    query_embedding vector(384),
    limit_count INTEGER DEFAULT 10,
    min_similarity FLOAT DEFAULT 0.7
)
RETURNS TABLE(
    document_id VARCHAR,
    content TEXT,
    similarity FLOAT,
    metadata JSONB
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        de.document_id,
        de.content,
        1 - (de.embedding <=> query_embedding) as similarity,
        de.metadata
    FROM document_embeddings de
    WHERE 1 - (de.embedding <=> query_embedding) > min_similarity
    ORDER BY de.embedding <=> query_embedding
    LIMIT limit_count;
END;
$$ LANGUAGE plpgsql;

-- Function to find similar tasks
CREATE OR REPLACE FUNCTION find_similar_tasks(
    query_embedding vector(384),
    limit_count INTEGER DEFAULT 5
)
RETURNS TABLE(
    task_type VARCHAR,
    best_model VARCHAR,
    similarity FLOAT,
    avg_cost DECIMAL,
    avg_latency_ms INTEGER,
    success_rate FLOAT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        te.task_type,
        te.best_model,
        1 - (te.embedding <=> query_embedding) as similarity,
        te.avg_cost,
        te.avg_latency_ms,
        te.success_rate
    FROM task_embeddings te
    ORDER BY te.embedding <=> query_embedding
    LIMIT limit_count;
    
    -- Update match count
    UPDATE task_embeddings
    SET match_count = match_count + 1,
        last_matched = NOW()
    WHERE task_embeddings.id IN (
        SELECT id FROM task_embeddings
        ORDER BY embedding <=> query_embedding
        LIMIT 1
    );
END;
$$ LANGUAGE plpgsql;

-- Function to clean old embeddings
CREATE OR REPLACE FUNCTION clean_old_embeddings(
    days_to_keep INTEGER DEFAULT 30
)
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM document_embeddings
    WHERE accessed_at < NOW() - INTERVAL '1 day' * days_to_keep
    AND access_count < 5;
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;