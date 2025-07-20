-- Migration: Add User-Level Master Summaries for Cross-Conversation Context
-- This enables maintaining comprehensive user context across all conversations

-- Create user master summaries table
CREATE TABLE IF NOT EXISTS user_master_summaries (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id TEXT UNIQUE NOT NULL,
    
    -- Master summary content
    master_summary TEXT NOT NULL,
    master_embedding VECTOR(1536),
    
    -- Consolidation metadata
    total_conversations INTEGER DEFAULT 0,
    total_messages INTEGER DEFAULT 0,
    conversations_included UUID[], -- Array of conversation IDs included
    last_conversation_id UUID,     -- Most recent conversation
    
    -- Context windows
    recent_context TEXT,           -- Summary of last 3 conversations
    historical_context TEXT,       -- Consolidated older conversations
    key_insights TEXT[],          -- Extracted key insights about user
    preferences JSONB DEFAULT '{}'::jsonb, -- User preferences learned
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_consolidation_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_user_master_summaries_user_id ON user_master_summaries(user_id);
CREATE INDEX IF NOT EXISTS idx_user_master_summaries_updated_at ON user_master_summaries(updated_at);
CREATE INDEX IF NOT EXISTS idx_user_master_summaries_embedding ON user_master_summaries USING ivfflat (master_embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_user_master_summaries_conversations ON user_master_summaries USING GIN(conversations_included);

-- Add trigger for updated_at
DROP TRIGGER IF EXISTS update_user_master_summaries_updated_at ON user_master_summaries;
CREATE TRIGGER update_user_master_summaries_updated_at
    BEFORE UPDATE ON user_master_summaries
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Function to get or create user master summary
CREATE OR REPLACE FUNCTION get_or_create_user_master_summary(
    target_user_id TEXT
)
RETURNS UUID
LANGUAGE plpgsql
AS $$
DECLARE
    summary_id UUID;
BEGIN
    -- Try to get existing master summary
    SELECT id INTO summary_id
    FROM user_master_summaries
    WHERE user_id = target_user_id;
    
    -- Create if doesn't exist
    IF summary_id IS NULL THEN
        INSERT INTO user_master_summaries (
            user_id, 
            master_summary, 
            recent_context, 
            historical_context
        )
        VALUES (
            target_user_id,
            'New user - no conversation history yet.',
            '',
            ''
        )
        RETURNING id INTO summary_id;
    END IF;
    
    RETURN summary_id;
END;
$$;

-- Function to consolidate user summary across conversations
CREATE OR REPLACE FUNCTION consolidate_user_summary(
    target_user_id TEXT,
    new_conversation_id UUID DEFAULT NULL
)
RETURNS BOOLEAN
LANGUAGE plpgsql
AS $$
DECLARE
    master_summary_id UUID;
    recent_conversations UUID[];
    conversation_summaries TEXT[];
    consolidated_text TEXT;
BEGIN
    -- Get or create master summary
    master_summary_id := get_or_create_user_master_summary(target_user_id);
    
    -- Get recent conversations (last 5)
    SELECT ARRAY_AGG(id ORDER BY updated_at DESC)
    INTO recent_conversations
    FROM (
        SELECT id, updated_at
        FROM conversations
        WHERE user_id = target_user_id
        ORDER BY updated_at DESC
        LIMIT 5
    ) recent;
    
    -- Get summaries for recent conversations
    SELECT ARRAY_AGG(summary_text ORDER BY created_at DESC)
    INTO conversation_summaries
    FROM conversation_summaries cs
    WHERE cs.user_id = target_user_id
      AND cs.conversation_id = ANY(recent_conversations)
      AND cs.consolidation_status = 'active'
    ORDER BY cs.created_at DESC
    LIMIT 10; -- Last 10 summaries across recent conversations
    
    -- Create consolidated text (this will be enhanced with LLM processing)
    consolidated_text := COALESCE(
        array_to_string(conversation_summaries, E'\n--- CONVERSATION BREAK ---\n'),
        'No conversation summaries available yet.'
    );
    
    -- Update master summary
    UPDATE user_master_summaries
    SET 
        master_summary = consolidated_text,
        conversations_included = recent_conversations,
        total_conversations = (
            SELECT COUNT(*) 
            FROM conversations 
            WHERE user_id = target_user_id
        ),
        total_messages = (
            SELECT COUNT(*) 
            FROM messages m
            JOIN conversations c ON m.conversation_id = c.id
            WHERE c.user_id = target_user_id
        ),
        last_conversation_id = new_conversation_id,
        last_consolidation_at = NOW(),
        updated_at = NOW()
    WHERE id = master_summary_id;
    
    RETURN TRUE;
END;
$$;

-- Function to get user context for new conversations
CREATE OR REPLACE FUNCTION get_user_context_for_conversation(
    target_user_id TEXT
)
RETURNS TABLE (
    master_summary TEXT,
    recent_context TEXT,
    key_insights TEXT[],
    total_conversations INTEGER,
    last_conversation_id UUID
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        ums.master_summary,
        ums.recent_context,
        ums.key_insights,
        ums.total_conversations,
        ums.last_conversation_id
    FROM user_master_summaries ums
    WHERE ums.user_id = target_user_id;
END;
$$;

-- Note: User summary consolidation is handled in Python code, not database triggers
-- This ensures we have proper LLM integration and error handling 