-- Token Budget Management Schema
-- Tracks and manages token usage across different time periods

SET search_path TO zen_budget, public;

-- Token budget configuration
CREATE TABLE IF NOT EXISTS token_budgets (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Budget identification
    budget_name VARCHAR(100) NOT NULL,
    budget_type VARCHAR(50) NOT NULL, -- 'global', 'model', 'provider', 'task', 'user'
    budget_scope VARCHAR(255), -- specific model/provider/task/user identifier
    
    -- Budget limits
    hourly_limit INTEGER,
    daily_limit INTEGER,
    weekly_limit INTEGER,
    monthly_limit INTEGER,
    
    -- Cost limits (in USD)
    hourly_cost_limit DECIMAL(10, 2),
    daily_cost_limit DECIMAL(10, 2),
    weekly_cost_limit DECIMAL(10, 2),
    monthly_cost_limit DECIMAL(10, 2),
    
    -- Current usage (updated in real-time)
    hourly_tokens_used INTEGER DEFAULT 0,
    daily_tokens_used INTEGER DEFAULT 0,
    weekly_tokens_used INTEGER DEFAULT 0,
    monthly_tokens_used INTEGER DEFAULT 0,
    
    hourly_cost_used DECIMAL(10, 4) DEFAULT 0,
    daily_cost_used DECIMAL(10, 4) DEFAULT 0,
    weekly_cost_used DECIMAL(10, 4) DEFAULT 0,
    monthly_cost_used DECIMAL(10, 4) DEFAULT 0,
    
    -- Reset timestamps
    hourly_reset_at TIMESTAMP WITH TIME ZONE,
    daily_reset_at TIMESTAMP WITH TIME ZONE,
    weekly_reset_at TIMESTAMP WITH TIME ZONE,
    monthly_reset_at TIMESTAMP WITH TIME ZONE,
    
    -- Configuration
    is_active BOOLEAN DEFAULT true,
    priority INTEGER DEFAULT 5, -- 1-10, higher is more important
    alert_threshold FLOAT DEFAULT 0.8, -- Alert when usage reaches 80%
    
    -- Metadata
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Constraints
    UNIQUE(budget_name, budget_type, budget_scope),
    CHECK (priority >= 1 AND priority <= 10),
    CHECK (alert_threshold >= 0 AND alert_threshold <= 1)
);

-- Token allocations tracking
CREATE TABLE IF NOT EXISTS token_allocations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Allocation identification
    budget_id UUID REFERENCES token_budgets(id) ON DELETE CASCADE,
    request_id VARCHAR(255) NOT NULL,
    session_id VARCHAR(255),
    
    -- Model and task info
    model_name VARCHAR(255) NOT NULL,
    provider VARCHAR(100) NOT NULL,
    task_type VARCHAR(100),
    
    -- Token usage
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    total_tokens INTEGER GENERATED ALWAYS AS (input_tokens + output_tokens) STORED,
    
    -- Cost tracking
    input_cost DECIMAL(10, 6),
    output_cost DECIMAL(10, 6),
    total_cost DECIMAL(10, 6),
    
    -- Allocation status
    status VARCHAR(50) NOT NULL, -- 'allocated', 'used', 'exceeded', 'rejected'
    rejection_reason TEXT,
    
    -- Timestamps
    allocated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    used_at TIMESTAMP WITH TIME ZONE,
    
    -- Indexes
    INDEX idx_alloc_budget (budget_id),
    INDEX idx_alloc_request (request_id),
    INDEX idx_alloc_session (session_id),
    INDEX idx_alloc_allocated (allocated_at DESC),
    INDEX idx_alloc_status (status)
);

-- Budget alerts
CREATE TABLE IF NOT EXISTS budget_alerts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Alert identification
    budget_id UUID REFERENCES token_budgets(id) ON DELETE CASCADE,
    alert_type VARCHAR(50) NOT NULL, -- 'threshold', 'exceeded', 'reset', 'prediction'
    
    -- Alert details
    period VARCHAR(20) NOT NULL, -- 'hourly', 'daily', 'weekly', 'monthly'
    usage_percentage FLOAT NOT NULL,
    tokens_used INTEGER,
    tokens_limit INTEGER,
    cost_used DECIMAL(10, 4),
    cost_limit DECIMAL(10, 2),
    
    -- Prediction (for predictive alerts)
    predicted_exhaustion TIMESTAMP WITH TIME ZONE,
    confidence_score FLOAT,
    
    -- Alert metadata
    message TEXT NOT NULL,
    severity VARCHAR(20) DEFAULT 'warning', -- 'info', 'warning', 'critical'
    acknowledged BOOLEAN DEFAULT false,
    acknowledged_by VARCHAR(255),
    acknowledged_at TIMESTAMP WITH TIME ZONE,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Indexes
    INDEX idx_alerts_budget (budget_id),
    INDEX idx_alerts_created (created_at DESC),
    INDEX idx_alerts_severity (severity),
    INDEX idx_alerts_ack (acknowledged)
);

-- Usage history for analytics
CREATE TABLE IF NOT EXISTS usage_history (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Period identification
    period_type VARCHAR(20) NOT NULL, -- 'hour', 'day', 'week', 'month'
    period_start TIMESTAMP WITH TIME ZONE NOT NULL,
    period_end TIMESTAMP WITH TIME ZONE NOT NULL,
    
    -- Aggregated metrics
    total_tokens INTEGER NOT NULL,
    total_cost DECIMAL(10, 4) NOT NULL,
    request_count INTEGER NOT NULL,
    unique_sessions INTEGER,
    unique_models INTEGER,
    
    -- Model breakdown
    model_usage JSONB, -- {model_name: {tokens: x, cost: y, requests: z}}
    
    -- Task breakdown
    task_usage JSONB, -- {task_type: {tokens: x, cost: y, requests: z}}
    
    -- Provider breakdown
    provider_usage JSONB, -- {provider: {tokens: x, cost: y, requests: z}}
    
    -- Performance metrics
    avg_tokens_per_request FLOAT,
    avg_cost_per_request DECIMAL(10, 6),
    success_rate FLOAT,
    
    -- Created timestamp
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Constraints
    UNIQUE(period_type, period_start),
    
    -- Indexes
    INDEX idx_history_period (period_type, period_start DESC),
    INDEX idx_history_start (period_start DESC)
);

-- Function to check budget availability
CREATE OR REPLACE FUNCTION check_budget_availability(
    p_budget_id UUID,
    p_tokens INTEGER,
    p_cost DECIMAL(10, 6)
)
RETURNS TABLE(
    is_available BOOLEAN,
    reason TEXT,
    remaining_tokens INTEGER,
    remaining_cost DECIMAL(10, 4)
) AS $$
DECLARE
    v_budget token_budgets%ROWTYPE;
    v_is_available BOOLEAN := true;
    v_reason TEXT := 'Budget available';
    v_remaining_tokens INTEGER;
    v_remaining_cost DECIMAL(10, 4);
BEGIN
    -- Get budget
    SELECT * INTO v_budget FROM token_budgets WHERE id = p_budget_id;
    
    IF NOT FOUND OR NOT v_budget.is_active THEN
        RETURN QUERY SELECT false, 'Budget not found or inactive', 0, 0.0;
        RETURN;
    END IF;
    
    -- Check hourly limits
    IF v_budget.hourly_limit IS NOT NULL THEN
        v_remaining_tokens := v_budget.hourly_limit - v_budget.hourly_tokens_used;
        IF v_budget.hourly_tokens_used + p_tokens > v_budget.hourly_limit THEN
            v_is_available := false;
            v_reason := 'Hourly token limit exceeded';
        END IF;
    END IF;
    
    -- Check daily limits
    IF v_is_available AND v_budget.daily_limit IS NOT NULL THEN
        v_remaining_tokens := LEAST(v_remaining_tokens, v_budget.daily_limit - v_budget.daily_tokens_used);
        IF v_budget.daily_tokens_used + p_tokens > v_budget.daily_limit THEN
            v_is_available := false;
            v_reason := 'Daily token limit exceeded';
        END IF;
    END IF;
    
    -- Check cost limits
    IF v_is_available AND v_budget.daily_cost_limit IS NOT NULL THEN
        v_remaining_cost := v_budget.daily_cost_limit - v_budget.daily_cost_used;
        IF v_budget.daily_cost_used + p_cost > v_budget.daily_cost_limit THEN
            v_is_available := false;
            v_reason := 'Daily cost limit exceeded';
        END IF;
    END IF;
    
    RETURN QUERY SELECT v_is_available, v_reason, v_remaining_tokens, v_remaining_cost;
END;
$$ LANGUAGE plpgsql;

-- Function to allocate tokens
CREATE OR REPLACE FUNCTION allocate_tokens(
    p_budget_id UUID,
    p_request_id VARCHAR(255),
    p_model_name VARCHAR(255),
    p_provider VARCHAR(100),
    p_tokens INTEGER,
    p_cost DECIMAL(10, 6)
)
RETURNS UUID AS $$
DECLARE
    v_allocation_id UUID;
    v_is_available BOOLEAN;
    v_reason TEXT;
BEGIN
    -- Check availability
    SELECT is_available, reason INTO v_is_available, v_reason
    FROM check_budget_availability(p_budget_id, p_tokens, p_cost);
    
    -- Create allocation record
    INSERT INTO token_allocations (
        budget_id, request_id, model_name, provider,
        input_tokens, output_tokens, total_cost,
        status, rejection_reason
    ) VALUES (
        p_budget_id, p_request_id, p_model_name, p_provider,
        p_tokens, 0, p_cost,
        CASE WHEN v_is_available THEN 'allocated' ELSE 'rejected' END,
        CASE WHEN NOT v_is_available THEN v_reason ELSE NULL END
    ) RETURNING id INTO v_allocation_id;
    
    -- Update budget usage if allocated
    IF v_is_available THEN
        UPDATE token_budgets
        SET hourly_tokens_used = hourly_tokens_used + p_tokens,
            daily_tokens_used = daily_tokens_used + p_tokens,
            weekly_tokens_used = weekly_tokens_used + p_tokens,
            monthly_tokens_used = monthly_tokens_used + p_tokens,
            hourly_cost_used = hourly_cost_used + p_cost,
            daily_cost_used = daily_cost_used + p_cost,
            weekly_cost_used = weekly_cost_used + p_cost,
            monthly_cost_used = monthly_cost_used + p_cost,
            updated_at = NOW()
        WHERE id = p_budget_id;
        
        -- Check for alerts
        PERFORM check_budget_alerts(p_budget_id);
    END IF;
    
    RETURN v_allocation_id;
END;
$$ LANGUAGE plpgsql;

-- Function to check and create budget alerts
CREATE OR REPLACE FUNCTION check_budget_alerts(p_budget_id UUID)
RETURNS void AS $$
DECLARE
    v_budget token_budgets%ROWTYPE;
    v_usage_pct FLOAT;
    v_period TEXT;
BEGIN
    SELECT * INTO v_budget FROM token_budgets WHERE id = p_budget_id;
    
    -- Check each period
    FOR v_period IN SELECT unnest(ARRAY['hourly', 'daily', 'weekly', 'monthly'])
    LOOP
        -- Calculate usage percentage
        v_usage_pct := CASE v_period
            WHEN 'hourly' THEN 
                CASE WHEN v_budget.hourly_limit > 0 
                THEN v_budget.hourly_tokens_used::FLOAT / v_budget.hourly_limit 
                ELSE 0 END
            WHEN 'daily' THEN 
                CASE WHEN v_budget.daily_limit > 0 
                THEN v_budget.daily_tokens_used::FLOAT / v_budget.daily_limit 
                ELSE 0 END
            WHEN 'weekly' THEN 
                CASE WHEN v_budget.weekly_limit > 0 
                THEN v_budget.weekly_tokens_used::FLOAT / v_budget.weekly_limit 
                ELSE 0 END
            WHEN 'monthly' THEN 
                CASE WHEN v_budget.monthly_limit > 0 
                THEN v_budget.monthly_tokens_used::FLOAT / v_budget.monthly_limit 
                ELSE 0 END
        END;
        
        -- Create alert if threshold exceeded
        IF v_usage_pct >= v_budget.alert_threshold THEN
            INSERT INTO budget_alerts (
                budget_id, alert_type, period, usage_percentage,
                message, severity
            ) VALUES (
                p_budget_id, 'threshold', v_period, v_usage_pct,
                format('Budget %s has reached %.1f%% of %s limit', 
                    v_budget.budget_name, v_usage_pct * 100, v_period),
                CASE 
                    WHEN v_usage_pct >= 1.0 THEN 'critical'
                    WHEN v_usage_pct >= 0.9 THEN 'warning'
                    ELSE 'info'
                END
            ) ON CONFLICT DO NOTHING;
        END IF;
    END LOOP;
END;
$$ LANGUAGE plpgsql;

-- Function to reset budget periods
CREATE OR REPLACE FUNCTION reset_budget_periods()
RETURNS void AS $$
BEGIN
    -- Reset hourly budgets
    UPDATE token_budgets
    SET hourly_tokens_used = 0,
        hourly_cost_used = 0,
        hourly_reset_at = NOW() + INTERVAL '1 hour'
    WHERE hourly_reset_at <= NOW();
    
    -- Reset daily budgets
    UPDATE token_budgets
    SET daily_tokens_used = 0,
        daily_cost_used = 0,
        daily_reset_at = NOW() + INTERVAL '1 day'
    WHERE daily_reset_at <= NOW();
    
    -- Reset weekly budgets
    UPDATE token_budgets
    SET weekly_tokens_used = 0,
        weekly_cost_used = 0,
        weekly_reset_at = NOW() + INTERVAL '1 week'
    WHERE weekly_reset_at <= NOW();
    
    -- Reset monthly budgets
    UPDATE token_budgets
    SET monthly_tokens_used = 0,
        monthly_cost_used = 0,
        monthly_reset_at = NOW() + INTERVAL '1 month'
    WHERE monthly_reset_at <= NOW();
END;
$$ LANGUAGE plpgsql;

-- Create default global budget
INSERT INTO token_budgets (
    budget_name, budget_type, budget_scope,
    daily_limit, monthly_limit,
    daily_cost_limit, monthly_cost_limit,
    hourly_reset_at, daily_reset_at, weekly_reset_at, monthly_reset_at
) VALUES (
    'Global Budget', 'global', 'all',
    1000000, 10000000,  -- 1M daily, 10M monthly tokens
    100.00, 1000.00,     -- $100 daily, $1000 monthly
    NOW() + INTERVAL '1 hour',
    NOW() + INTERVAL '1 day',
    NOW() + INTERVAL '1 week',
    NOW() + INTERVAL '1 month'
) ON CONFLICT (budget_name, budget_type, budget_scope) DO NOTHING;