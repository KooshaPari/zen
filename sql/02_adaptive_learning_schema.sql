-- Adaptive Learning Database Schema
-- This schema tracks model predictions, actual performance, and learning metrics

-- Create schema if not exists
CREATE SCHEMA IF NOT EXISTS zen_adaptive;

-- Prediction tracking table
CREATE TABLE IF NOT EXISTS zen_adaptive.predictions (
    prediction_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    request_id VARCHAR(255) NOT NULL,
    model_name VARCHAR(255) NOT NULL,
    task_type VARCHAR(100) NOT NULL,
    
    -- Predicted metrics
    predicted_cost DECIMAL(10, 6),
    predicted_latency_ms INTEGER,
    predicted_quality DECIMAL(3, 2),
    predicted_tps DECIMAL(10, 2),
    predicted_success_rate DECIMAL(3, 2),
    confidence_score DECIMAL(3, 2),
    
    -- Context information
    input_tokens INTEGER,
    expected_output_tokens INTEGER,
    context_window_size INTEGER,
    optimization_objective VARCHAR(50),
    
    -- Constraints
    max_cost_per_request DECIMAL(10, 4),
    max_latency_ms INTEGER,
    min_quality_score DECIMAL(3, 2),
    
    -- Risk assessments
    context_overflow_risk DECIMAL(3, 2),
    timeout_risk DECIMAL(3, 2),
    quality_risk DECIMAL(3, 2),
    
    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    work_dir_id UUID,
    session_id VARCHAR(255),
    
    -- Indexes
    INDEX idx_predictions_request (request_id),
    INDEX idx_predictions_model (model_name),
    INDEX idx_predictions_created (created_at),
    INDEX idx_predictions_work_dir (work_dir_id)
);

-- Actual performance reconciliation table
CREATE TABLE IF NOT EXISTS zen_adaptive.actual_performance (
    performance_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    prediction_id UUID REFERENCES zen_adaptive.predictions(prediction_id),
    request_id VARCHAR(255) NOT NULL,
    
    -- Actual metrics
    actual_cost DECIMAL(10, 6),
    actual_latency_ms INTEGER,
    actual_quality DECIMAL(3, 2),
    actual_tps DECIMAL(10, 2),
    actual_success BOOLEAN,
    
    -- Token counts
    actual_input_tokens INTEGER,
    actual_output_tokens INTEGER,
    cached_tokens INTEGER,
    
    -- Performance details
    time_to_first_token_ms INTEGER,
    total_time_seconds DECIMAL(10, 3),
    error_type VARCHAR(100),
    
    -- Prediction errors
    cost_error_rate DECIMAL(5, 4),
    latency_error_rate DECIMAL(5, 4),
    quality_error_rate DECIMAL(5, 4),
    tps_error_rate DECIMAL(5, 4),
    mean_error_rate DECIMAL(5, 4),
    
    -- Metadata
    reconciled_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Indexes
    INDEX idx_actual_request (request_id),
    INDEX idx_actual_prediction (prediction_id),
    INDEX idx_actual_reconciled (reconciled_at)
);

-- Model learning history
CREATE TABLE IF NOT EXISTS zen_adaptive.learning_history (
    learning_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_name VARCHAR(255) NOT NULL,
    task_type VARCHAR(100),
    
    -- Learning metrics
    training_samples INTEGER,
    mean_absolute_error DECIMAL(5, 4),
    cost_prediction_accuracy DECIMAL(5, 4),
    latency_prediction_accuracy DECIMAL(5, 4),
    quality_prediction_accuracy DECIMAL(5, 4),
    
    -- Model performance over time
    selection_count INTEGER DEFAULT 0,
    success_count INTEGER DEFAULT 0,
    failure_count INTEGER DEFAULT 0,
    avg_confidence DECIMAL(3, 2),
    
    -- Optimization scores
    avg_cost_efficiency DECIMAL(5, 4),
    avg_performance_score DECIMAL(5, 4),
    avg_overall_score DECIMAL(5, 4),
    
    -- Time window
    period_start TIMESTAMP WITH TIME ZONE,
    period_end TIMESTAMP WITH TIME ZONE,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Indexes
    INDEX idx_learning_model (model_name),
    INDEX idx_learning_task (task_type),
    INDEX idx_learning_period (period_start, period_end)
);

-- Context patterns for learning
CREATE TABLE IF NOT EXISTS zen_adaptive.context_patterns (
    pattern_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    pattern_hash VARCHAR(64) NOT NULL,
    
    -- Pattern characteristics
    task_type VARCHAR(100),
    complexity_range VARCHAR(20),
    token_range VARCHAR(20),
    optimization_objective VARCHAR(50),
    
    -- Best performing models
    best_model_for_cost VARCHAR(255),
    best_model_for_speed VARCHAR(255),
    best_model_for_quality VARCHAR(255),
    best_model_overall VARCHAR(255),
    
    -- Performance statistics
    sample_count INTEGER DEFAULT 1,
    avg_cost DECIMAL(10, 6),
    avg_latency_ms INTEGER,
    avg_quality DECIMAL(3, 2),
    success_rate DECIMAL(3, 2),
    
    -- Metadata
    first_seen TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_seen TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_updated TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Indexes
    UNIQUE INDEX idx_pattern_hash (pattern_hash),
    INDEX idx_pattern_task (task_type),
    INDEX idx_pattern_updated (last_updated)
);

-- Training data for neural network
CREATE TABLE IF NOT EXISTS zen_adaptive.training_data (
    training_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Feature vector (stored as JSONB for flexibility)
    features JSONB NOT NULL,
    
    -- Target values
    target_cost DECIMAL(10, 6),
    target_latency_ms INTEGER,
    target_quality DECIMAL(3, 2),
    target_tps DECIMAL(10, 2),
    target_success BOOLEAN,
    
    -- Data quality
    is_validated BOOLEAN DEFAULT FALSE,
    is_anomaly BOOLEAN DEFAULT FALSE,
    confidence_weight DECIMAL(3, 2) DEFAULT 1.0,
    
    -- Metadata
    model_name VARCHAR(255),
    task_type VARCHAR(100),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    used_in_training BOOLEAN DEFAULT FALSE,
    training_batch_id VARCHAR(100),
    
    -- Indexes
    INDEX idx_training_created (created_at),
    INDEX idx_training_used (used_in_training),
    INDEX idx_training_batch (training_batch_id)
);

-- Model weights and checkpoints
CREATE TABLE IF NOT EXISTS zen_adaptive.model_checkpoints (
    checkpoint_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_version VARCHAR(50) NOT NULL,
    
    -- Model state (stored as binary)
    model_weights BYTEA,
    scaler_state JSONB,
    
    -- Performance metrics at checkpoint
    validation_loss DECIMAL(10, 6),
    training_loss DECIMAL(10, 6),
    epoch INTEGER,
    training_samples INTEGER,
    
    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    is_active BOOLEAN DEFAULT FALSE,
    description TEXT,
    
    -- Indexes
    INDEX idx_checkpoint_version (model_version),
    INDEX idx_checkpoint_active (is_active),
    INDEX idx_checkpoint_created (created_at)
);

-- Create views for analytics
CREATE OR REPLACE VIEW zen_adaptive.prediction_accuracy AS
SELECT 
    p.model_name,
    p.task_type,
    COUNT(*) as total_predictions,
    AVG(ap.cost_error_rate) as avg_cost_error,
    AVG(ap.latency_error_rate) as avg_latency_error,
    AVG(ap.quality_error_rate) as avg_quality_error,
    AVG(ap.mean_error_rate) as avg_overall_error,
    SUM(CASE WHEN ap.actual_success THEN 1 ELSE 0 END)::FLOAT / COUNT(*) as success_rate
FROM zen_adaptive.predictions p
JOIN zen_adaptive.actual_performance ap ON p.prediction_id = ap.prediction_id
WHERE p.created_at > NOW() - INTERVAL '30 days'
GROUP BY p.model_name, p.task_type;

CREATE OR REPLACE VIEW zen_adaptive.model_selection_trends AS
SELECT 
    DATE_TRUNC('hour', created_at) as hour,
    model_name,
    optimization_objective,
    COUNT(*) as selections,
    AVG(confidence_score) as avg_confidence,
    AVG(predicted_cost) as avg_predicted_cost,
    AVG(predicted_quality) as avg_predicted_quality
FROM zen_adaptive.predictions
WHERE created_at > NOW() - INTERVAL '7 days'
GROUP BY DATE_TRUNC('hour', created_at), model_name, optimization_objective
ORDER BY hour DESC;

-- Function to calculate learning rate
CREATE OR REPLACE FUNCTION zen_adaptive.calculate_learning_rate(
    model_name_param VARCHAR,
    lookback_days INTEGER DEFAULT 7
) RETURNS TABLE (
    improvement_rate DECIMAL,
    current_accuracy DECIMAL,
    previous_accuracy DECIMAL,
    sample_count INTEGER
) AS $$
BEGIN
    RETURN QUERY
    WITH current_period AS (
        SELECT 
            AVG(ap.mean_error_rate) as error_rate,
            COUNT(*) as samples
        FROM zen_adaptive.predictions p
        JOIN zen_adaptive.actual_performance ap ON p.prediction_id = ap.prediction_id
        WHERE p.model_name = model_name_param
        AND p.created_at > NOW() - INTERVAL '1 day' * (lookback_days / 2)
    ),
    previous_period AS (
        SELECT 
            AVG(ap.mean_error_rate) as error_rate
        FROM zen_adaptive.predictions p
        JOIN zen_adaptive.actual_performance ap ON p.prediction_id = ap.prediction_id
        WHERE p.model_name = model_name_param
        AND p.created_at BETWEEN NOW() - INTERVAL '1 day' * lookback_days 
                              AND NOW() - INTERVAL '1 day' * (lookback_days / 2)
    )
    SELECT 
        CASE 
            WHEN previous_period.error_rate > 0 
            THEN ((previous_period.error_rate - current_period.error_rate) / previous_period.error_rate)::DECIMAL
            ELSE 0
        END as improvement_rate,
        (1 - current_period.error_rate)::DECIMAL as current_accuracy,
        (1 - previous_period.error_rate)::DECIMAL as previous_accuracy,
        current_period.samples::INTEGER
    FROM current_period, previous_period;
END;
$$ LANGUAGE plpgsql;

-- Trigger to update learning history
CREATE OR REPLACE FUNCTION zen_adaptive.update_learning_history()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO zen_adaptive.learning_history (
        model_name,
        task_type,
        training_samples,
        mean_absolute_error,
        selection_count,
        success_count,
        period_start,
        period_end
    )
    SELECT 
        p.model_name,
        p.task_type,
        COUNT(*),
        AVG(NEW.mean_error_rate),
        COUNT(*),
        SUM(CASE WHEN NEW.actual_success THEN 1 ELSE 0 END),
        DATE_TRUNC('hour', NOW() - INTERVAL '1 hour'),
        DATE_TRUNC('hour', NOW())
    FROM zen_adaptive.predictions p
    WHERE p.prediction_id = NEW.prediction_id
    GROUP BY p.model_name, p.task_type
    ON CONFLICT (model_name, task_type, period_start) 
    DO UPDATE SET
        training_samples = EXCLUDED.training_samples,
        mean_absolute_error = EXCLUDED.mean_absolute_error,
        selection_count = EXCLUDED.selection_count,
        success_count = EXCLUDED.success_count,
        updated_at = NOW();
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_learning_history_trigger
AFTER INSERT ON zen_adaptive.actual_performance
FOR EACH ROW
EXECUTE FUNCTION zen_adaptive.update_learning_history();

-- Grant permissions
GRANT ALL ON SCHEMA zen_adaptive TO zen_user;
GRANT ALL ON ALL TABLES IN SCHEMA zen_adaptive TO zen_user;
GRANT ALL ON ALL SEQUENCES IN SCHEMA zen_adaptive TO zen_user;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA zen_adaptive TO zen_user;