-- Performance Tracking Schema
-- Stores model performance metrics and historical data

SET search_path TO zen_performance, public;

-- Model performance metrics table
CREATE TABLE IF NOT EXISTS model_performance (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_name VARCHAR(255) NOT NULL,
    provider VARCHAR(100) NOT NULL,
    task_type VARCHAR(100) NOT NULL,
    task_hash VARCHAR(64),
    
    -- Token metrics
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    cached_tokens INTEGER DEFAULT 0,
    total_tokens INTEGER GENERATED ALWAYS AS (input_tokens + output_tokens) STORED,
    
    -- Time metrics (in seconds)
    first_token_time FLOAT,
    total_time FLOAT NOT NULL,
    queue_time FLOAT DEFAULT 0,
    
    -- Cost metrics (in USD)
    input_cost DECIMAL(10, 6),
    output_cost DECIMAL(10, 6),
    total_cost DECIMAL(10, 6),
    cost_per_token DECIMAL(10, 8),
    
    -- Performance metrics
    tokens_per_second FLOAT,
    success BOOLEAN NOT NULL DEFAULT true,
    error_type VARCHAR(100),
    quality_score FLOAT DEFAULT 1.0 CHECK (quality_score >= 0 AND quality_score <= 1),
    
    -- Context information
    context_length INTEGER,
    max_context INTEGER,
    temperature FLOAT,
    
    -- Request metadata
    session_id VARCHAR(255),
    request_id VARCHAR(255),
    work_dir_id UUID,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Indexes
    INDEX idx_perf_model (model_name),
    INDEX idx_perf_provider (provider),
    INDEX idx_perf_task (task_type),
    INDEX idx_perf_created (created_at DESC),
    INDEX idx_perf_session (session_id),
    INDEX idx_perf_work_dir (work_dir_id)
);

-- Model pricing information
CREATE TABLE IF NOT EXISTS model_pricing (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_name VARCHAR(255) NOT NULL,
    provider VARCHAR(100) NOT NULL,
    
    -- Pricing per million tokens
    input_price_per_1m DECIMAL(10, 4) NOT NULL,
    output_price_per_1m DECIMAL(10, 4) NOT NULL,
    cached_price_per_1m DECIMAL(10, 4),
    
    -- Model characteristics
    context_window INTEGER,
    max_output_tokens INTEGER,
    supports_vision BOOLEAN DEFAULT false,
    supports_functions BOOLEAN DEFAULT false,
    supports_streaming BOOLEAN DEFAULT true,
    
    -- Metadata
    effective_date DATE NOT NULL DEFAULT CURRENT_DATE,
    expires_date DATE,
    is_active BOOLEAN DEFAULT true,
    notes TEXT,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Constraints
    UNIQUE(model_name, provider, effective_date),
    CHECK (expires_date IS NULL OR expires_date > effective_date)
);

-- Aggregated performance statistics (materialized view)
CREATE MATERIALIZED VIEW IF NOT EXISTS model_performance_stats AS
SELECT 
    model_name,
    provider,
    task_type,
    COUNT(*) as request_count,
    SUM(CASE WHEN success THEN 1 ELSE 0 END)::FLOAT / COUNT(*) as success_rate,
    
    -- Token statistics
    AVG(input_tokens) as avg_input_tokens,
    AVG(output_tokens) as avg_output_tokens,
    SUM(total_tokens) as total_tokens_used,
    
    -- Time statistics (in milliseconds)
    AVG(first_token_time * 1000) as avg_ttft_ms,
    AVG(total_time * 1000) as avg_latency_ms,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY total_time * 1000) as p50_latency_ms,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY total_time * 1000) as p95_latency_ms,
    PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY total_time * 1000) as p99_latency_ms,
    
    -- Cost statistics
    AVG(total_cost) as avg_cost_usd,
    SUM(total_cost) as total_cost_usd,
    AVG(cost_per_token * 1000000) as avg_cost_per_1m_tokens,
    
    -- Performance statistics
    AVG(tokens_per_second) as avg_tps,
    AVG(quality_score) as avg_quality,
    
    -- Time window
    MIN(created_at) as first_request,
    MAX(created_at) as last_request
FROM model_performance
GROUP BY model_name, provider, task_type;

-- Create index on materialized view
CREATE INDEX idx_perf_stats_model ON model_performance_stats(model_name);
CREATE INDEX idx_perf_stats_task ON model_performance_stats(task_type);

-- Function to refresh performance statistics
CREATE OR REPLACE FUNCTION refresh_performance_stats()
RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY model_performance_stats;
END;
$$ LANGUAGE plpgsql;

-- Trigger to update timestamps
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_model_pricing_updated_at
BEFORE UPDATE ON model_pricing
FOR EACH ROW
EXECUTE FUNCTION update_updated_at();