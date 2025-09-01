-- Conversation Memory and Context Management Schema
-- Stores conversation history and manages context across tools

SET search_path TO zen_conversation, public;

-- Conversations table
CREATE TABLE IF NOT EXISTS conversations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Conversation identification
    conversation_id VARCHAR(255) NOT NULL UNIQUE,
    session_id VARCHAR(255),
    user_id VARCHAR(255),
    
    -- Conversation metadata
    title TEXT,
    description TEXT,
    tags TEXT[],
    
    -- State management
    is_active BOOLEAN DEFAULT true,
    last_activity TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    turn_count INTEGER DEFAULT 0,
    total_tokens INTEGER DEFAULT 0,
    total_cost DECIMAL(10, 6) DEFAULT 0,
    
    -- Context window management
    context_strategy VARCHAR(50) DEFAULT 'sliding', -- 'sliding', 'summary', 'hybrid'
    max_context_tokens INTEGER DEFAULT 8000,
    current_context_tokens INTEGER DEFAULT 0,
    
    -- Metadata
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Indexes
    INDEX idx_conv_session (session_id),
    INDEX idx_conv_user (user_id),
    INDEX idx_conv_active (is_active, last_activity DESC),
    INDEX idx_conv_created (created_at DESC)
);

-- Conversation turns (messages)
CREATE TABLE IF NOT EXISTS conversation_turns (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Turn identification
    conversation_id VARCHAR(255) NOT NULL REFERENCES conversations(conversation_id) ON DELETE CASCADE,
    turn_number INTEGER NOT NULL,
    parent_turn_id UUID REFERENCES conversation_turns(id),
    
    -- Message content
    role VARCHAR(20) NOT NULL, -- 'user', 'assistant', 'system', 'tool'
    content TEXT NOT NULL,
    content_tokens INTEGER,
    
    -- Tool information
    tool_name VARCHAR(100),
    tool_input JSONB,
    tool_output JSONB,
    tool_error TEXT,
    
    -- Model information
    model_name VARCHAR(255),
    provider VARCHAR(100),
    
    -- Token usage
    input_tokens INTEGER,
    output_tokens INTEGER,
    total_tokens INTEGER GENERATED ALWAYS AS (COALESCE(input_tokens, 0) + COALESCE(output_tokens, 0)) STORED,
    
    -- Cost tracking
    cost DECIMAL(10, 6),
    
    -- Quality and performance
    quality_score FLOAT,
    latency_ms INTEGER,
    ttft_ms INTEGER, -- Time to first token
    tps FLOAT, -- Tokens per second
    
    -- Context management
    in_context BOOLEAN DEFAULT true,
    context_priority INTEGER DEFAULT 5, -- 1-10, higher is more important
    summarized BOOLEAN DEFAULT false,
    summary TEXT,
    
    -- Metadata
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Constraints
    UNIQUE(conversation_id, turn_number),
    CHECK (role IN ('user', 'assistant', 'system', 'tool')),
    CHECK (context_priority >= 1 AND context_priority <= 10),
    
    -- Indexes
    INDEX idx_turn_conv (conversation_id, turn_number),
    INDEX idx_turn_parent (parent_turn_id),
    INDEX idx_turn_tool (tool_name),
    INDEX idx_turn_created (created_at DESC),
    INDEX idx_turn_context (in_context, context_priority DESC)
);

-- Tool memory across conversations
CREATE TABLE IF NOT EXISTS tool_memory (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Memory identification
    tool_name VARCHAR(100) NOT NULL,
    memory_key VARCHAR(255) NOT NULL,
    
    -- Memory content
    memory_value JSONB NOT NULL,
    memory_type VARCHAR(50), -- 'preference', 'pattern', 'result', 'state'
    
    -- Scope
    scope VARCHAR(50) DEFAULT 'global', -- 'global', 'session', 'user', 'conversation'
    scope_id VARCHAR(255), -- session_id, user_id, or conversation_id
    
    -- Metadata
    access_count INTEGER DEFAULT 0,
    last_accessed TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Constraints
    UNIQUE(tool_name, memory_key, scope, scope_id),
    
    -- Indexes
    INDEX idx_toolmem_tool (tool_name),
    INDEX idx_toolmem_scope (scope, scope_id),
    INDEX idx_toolmem_accessed (last_accessed DESC),
    INDEX idx_toolmem_expires (expires_at)
);

-- Context summaries for long conversations
CREATE TABLE IF NOT EXISTS context_summaries (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Summary identification
    conversation_id VARCHAR(255) NOT NULL REFERENCES conversations(conversation_id) ON DELETE CASCADE,
    start_turn INTEGER NOT NULL,
    end_turn INTEGER NOT NULL,
    
    -- Summary content
    summary TEXT NOT NULL,
    summary_tokens INTEGER,
    
    -- Key points and entities
    key_points TEXT[],
    entities JSONB, -- Named entities, topics, etc.
    
    -- Metadata
    created_by VARCHAR(255), -- Model that created the summary
    quality_score FLOAT,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Constraints
    CHECK (end_turn >= start_turn),
    
    -- Indexes
    INDEX idx_summary_conv (conversation_id),
    INDEX idx_summary_turns (conversation_id, start_turn, end_turn)
);

-- Cross-tool conversation flow
CREATE TABLE IF NOT EXISTS tool_transitions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Transition identification
    conversation_id VARCHAR(255) NOT NULL,
    from_tool VARCHAR(100),
    to_tool VARCHAR(100) NOT NULL,
    turn_number INTEGER NOT NULL,
    
    -- Context passed
    context_passed JSONB,
    context_tokens INTEGER,
    
    -- Transition metadata
    transition_type VARCHAR(50), -- 'continuation', 'delegation', 'escalation', 'fallback'
    reason TEXT,
    success BOOLEAN DEFAULT true,
    
    -- Performance
    transition_latency_ms INTEGER,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Indexes
    INDEX idx_trans_conv (conversation_id),
    INDEX idx_trans_tools (from_tool, to_tool),
    INDEX idx_trans_created (created_at DESC)
);

-- Function to get conversation context
CREATE OR REPLACE FUNCTION get_conversation_context(
    p_conversation_id VARCHAR(255),
    p_max_tokens INTEGER DEFAULT 8000
)
RETURNS TABLE(
    turn_number INTEGER,
    role VARCHAR,
    content TEXT,
    tool_name VARCHAR,
    tokens INTEGER
) AS $$
DECLARE
    v_total_tokens INTEGER := 0;
BEGIN
    RETURN QUERY
    WITH prioritized_turns AS (
        SELECT 
            ct.turn_number,
            ct.role,
            ct.content,
            ct.tool_name,
            COALESCE(ct.content_tokens, length(ct.content) / 4) as tokens,
            ct.context_priority,
            ct.created_at
        FROM conversation_turns ct
        WHERE ct.conversation_id = p_conversation_id
            AND ct.in_context = true
            AND NOT ct.summarized
        ORDER BY ct.context_priority DESC, ct.created_at DESC
    ),
    selected_turns AS (
        SELECT 
            pt.turn_number,
            pt.role,
            pt.content,
            pt.tool_name,
            pt.tokens,
            SUM(pt.tokens) OVER (ORDER BY pt.context_priority DESC, pt.created_at DESC) as running_total
        FROM prioritized_turns pt
    )
    SELECT 
        st.turn_number,
        st.role,
        st.content,
        st.tool_name,
        st.tokens::INTEGER
    FROM selected_turns st
    WHERE st.running_total <= p_max_tokens
    ORDER BY st.turn_number;
END;
$$ LANGUAGE plpgsql;

-- Function to add conversation turn
CREATE OR REPLACE FUNCTION add_conversation_turn(
    p_conversation_id VARCHAR(255),
    p_role VARCHAR(20),
    p_content TEXT,
    p_tool_name VARCHAR(100) DEFAULT NULL,
    p_model_name VARCHAR(255) DEFAULT NULL,
    p_tokens INTEGER DEFAULT NULL,
    p_metadata JSONB DEFAULT NULL
)
RETURNS UUID AS $$
DECLARE
    v_turn_id UUID;
    v_turn_number INTEGER;
    v_conversation_exists BOOLEAN;
BEGIN
    -- Check if conversation exists, create if not
    SELECT EXISTS(SELECT 1 FROM conversations WHERE conversation_id = p_conversation_id)
    INTO v_conversation_exists;
    
    IF NOT v_conversation_exists THEN
        INSERT INTO conversations (conversation_id)
        VALUES (p_conversation_id);
    END IF;
    
    -- Get next turn number
    SELECT COALESCE(MAX(turn_number), 0) + 1
    INTO v_turn_number
    FROM conversation_turns
    WHERE conversation_id = p_conversation_id;
    
    -- Insert turn
    INSERT INTO conversation_turns (
        conversation_id, turn_number, role, content,
        tool_name, model_name, content_tokens, metadata
    ) VALUES (
        p_conversation_id, v_turn_number, p_role, p_content,
        p_tool_name, p_model_name, p_tokens, p_metadata
    ) RETURNING id INTO v_turn_id;
    
    -- Update conversation
    UPDATE conversations
    SET turn_count = turn_count + 1,
        last_activity = NOW(),
        current_context_tokens = current_context_tokens + COALESCE(p_tokens, length(p_content) / 4),
        updated_at = NOW()
    WHERE conversation_id = p_conversation_id;
    
    -- Check if context compression needed
    PERFORM compress_context_if_needed(p_conversation_id);
    
    RETURN v_turn_id;
END;
$$ LANGUAGE plpgsql;

-- Function to compress context when needed
CREATE OR REPLACE FUNCTION compress_context_if_needed(
    p_conversation_id VARCHAR(255)
)
RETURNS void AS $$
DECLARE
    v_max_tokens INTEGER;
    v_current_tokens INTEGER;
    v_strategy VARCHAR(50);
BEGIN
    -- Get conversation settings
    SELECT max_context_tokens, current_context_tokens, context_strategy
    INTO v_max_tokens, v_current_tokens, v_strategy
    FROM conversations
    WHERE conversation_id = p_conversation_id;
    
    -- Check if compression needed
    IF v_current_tokens > v_max_tokens * 0.9 THEN
        CASE v_strategy
            WHEN 'sliding' THEN
                -- Mark old turns as out of context
                UPDATE conversation_turns
                SET in_context = false
                WHERE conversation_id = p_conversation_id
                    AND turn_number < (
                        SELECT MAX(turn_number) - 20
                        FROM conversation_turns
                        WHERE conversation_id = p_conversation_id
                    );
                    
            WHEN 'summary' THEN
                -- Create summary of old turns
                PERFORM create_context_summary(p_conversation_id);
                
            WHEN 'hybrid' THEN
                -- Both sliding and summary
                PERFORM create_context_summary(p_conversation_id);
                UPDATE conversation_turns
                SET in_context = false
                WHERE conversation_id = p_conversation_id
                    AND summarized = true;
        END CASE;
        
        -- Update token count
        UPDATE conversations
        SET current_context_tokens = (
            SELECT COALESCE(SUM(content_tokens), 0)
            FROM conversation_turns
            WHERE conversation_id = p_conversation_id
                AND in_context = true
        )
        WHERE conversation_id = p_conversation_id;
    END IF;
END;
$$ LANGUAGE plpgsql;

-- Function to create context summary
CREATE OR REPLACE FUNCTION create_context_summary(
    p_conversation_id VARCHAR(255)
)
RETURNS UUID AS $$
DECLARE
    v_summary_id UUID;
    v_start_turn INTEGER;
    v_end_turn INTEGER;
BEGIN
    -- Get turn range to summarize
    SELECT MIN(turn_number), MAX(turn_number) - 10
    INTO v_start_turn, v_end_turn
    FROM conversation_turns
    WHERE conversation_id = p_conversation_id
        AND in_context = true
        AND NOT summarized;
    
    IF v_end_turn > v_start_turn THEN
        -- Note: Actual summarization would be done by the application
        -- This just creates a placeholder
        INSERT INTO context_summaries (
            conversation_id, start_turn, end_turn,
            summary, summary_tokens
        ) VALUES (
            p_conversation_id, v_start_turn, v_end_turn,
            'Summary pending generation', 100
        ) RETURNING id INTO v_summary_id;
        
        -- Mark turns as summarized
        UPDATE conversation_turns
        SET summarized = true
        WHERE conversation_id = p_conversation_id
            AND turn_number BETWEEN v_start_turn AND v_end_turn;
    END IF;
    
    RETURN v_summary_id;
END;
$$ LANGUAGE plpgsql;

-- Function to get tool memory
CREATE OR REPLACE FUNCTION get_tool_memory(
    p_tool_name VARCHAR(100),
    p_memory_key VARCHAR(255),
    p_scope VARCHAR(50) DEFAULT 'global',
    p_scope_id VARCHAR(255) DEFAULT NULL
)
RETURNS JSONB AS $$
DECLARE
    v_memory_value JSONB;
BEGIN
    -- Get memory value
    SELECT memory_value INTO v_memory_value
    FROM tool_memory
    WHERE tool_name = p_tool_name
        AND memory_key = p_memory_key
        AND scope = p_scope
        AND (scope_id = p_scope_id OR (p_scope_id IS NULL AND scope_id IS NULL));
    
    -- Update access tracking
    IF FOUND THEN
        UPDATE tool_memory
        SET access_count = access_count + 1,
            last_accessed = NOW()
        WHERE tool_name = p_tool_name
            AND memory_key = p_memory_key
            AND scope = p_scope
            AND (scope_id = p_scope_id OR (p_scope_id IS NULL AND scope_id IS NULL));
    END IF;
    
    RETURN v_memory_value;
END;
$$ LANGUAGE plpgsql;

-- Function to set tool memory
CREATE OR REPLACE FUNCTION set_tool_memory(
    p_tool_name VARCHAR(100),
    p_memory_key VARCHAR(255),
    p_memory_value JSONB,
    p_scope VARCHAR(50) DEFAULT 'global',
    p_scope_id VARCHAR(255) DEFAULT NULL,
    p_memory_type VARCHAR(50) DEFAULT NULL,
    p_expires_in_seconds INTEGER DEFAULT NULL
)
RETURNS void AS $$
BEGIN
    INSERT INTO tool_memory (
        tool_name, memory_key, memory_value,
        scope, scope_id, memory_type,
        expires_at
    ) VALUES (
        p_tool_name, p_memory_key, p_memory_value,
        p_scope, p_scope_id, p_memory_type,
        CASE 
            WHEN p_expires_in_seconds IS NOT NULL 
            THEN NOW() + (p_expires_in_seconds || ' seconds')::INTERVAL 
            ELSE NULL 
        END
    )
    ON CONFLICT (tool_name, memory_key, scope, scope_id)
    DO UPDATE SET
        memory_value = EXCLUDED.memory_value,
        memory_type = COALESCE(EXCLUDED.memory_type, tool_memory.memory_type),
        expires_at = EXCLUDED.expires_at,
        access_count = tool_memory.access_count + 1,
        last_accessed = NOW(),
        updated_at = NOW();
END;
$$ LANGUAGE plpgsql;

-- Function to clean expired memories
CREATE OR REPLACE FUNCTION clean_expired_memories()
RETURNS INTEGER AS $$
DECLARE
    v_deleted_count INTEGER;
BEGIN
    DELETE FROM tool_memory
    WHERE expires_at < NOW();
    
    GET DIAGNOSTICS v_deleted_count = ROW_COUNT;
    RETURN v_deleted_count;
END;
$$ LANGUAGE plpgsql;