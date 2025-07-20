-- Migration: Remove Master Summary System
-- Simplify memory system by removing user_master_summaries table and related functions
-- The system will now rely solely on conversation summaries, using the latest conversation summary as the main user summary

-- Drop functions that depend on user_master_summaries table
DROP FUNCTION IF EXISTS get_user_context_for_conversation(TEXT);
DROP FUNCTION IF EXISTS consolidate_user_summary(TEXT, UUID);
DROP FUNCTION IF EXISTS get_or_create_user_master_summary(TEXT);

-- Drop triggers and indexes related to user_master_summaries
DROP TRIGGER IF EXISTS update_user_master_summaries_updated_at ON user_master_summaries;
DROP INDEX IF EXISTS idx_user_master_summaries_user_id;
DROP INDEX IF EXISTS idx_user_master_summaries_updated_at;
DROP INDEX IF EXISTS idx_user_master_summaries_embedding;
DROP INDEX IF EXISTS idx_user_master_summaries_conversations;

-- Drop the user_master_summaries table
DROP TABLE IF EXISTS user_master_summaries;

-- Create simplified function to get latest conversation summary for user
CREATE OR REPLACE FUNCTION get_latest_user_conversation_summary(
    target_user_id UUID
)
RETURNS TABLE (
    summary_text TEXT,
    conversation_id UUID,
    created_at TIMESTAMP WITH TIME ZONE,
    message_count INTEGER,
    summary_type VARCHAR
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        cs.summary_text,
        cs.conversation_id,
        cs.created_at,
        cs.message_count,
        cs.summary_type
    FROM conversation_summaries cs
    WHERE cs.user_id = target_user_id
        AND cs.consolidation_status = 'active'
    ORDER BY cs.created_at DESC
    LIMIT 1;
END;
$$;

-- Create function to get user context based on latest conversation summary
CREATE OR REPLACE FUNCTION get_user_context_from_conversations(
    target_user_id UUID
)
RETURNS TABLE (
    latest_summary TEXT,
    conversation_count INTEGER,
    latest_conversation_id UUID,
    has_history BOOLEAN
)
LANGUAGE plpgsql
AS $$
DECLARE
    conv_count INTEGER;
BEGIN
    -- Get conversation count
    SELECT COUNT(*) INTO conv_count
    FROM conversations
    WHERE user_id = target_user_id;
    
    RETURN QUERY
    SELECT
        COALESCE(cs.summary_text, 'No conversation history yet.') as latest_summary,
        conv_count as conversation_count,
        cs.conversation_id as latest_conversation_id,
        (conv_count > 0) as has_history
    FROM conversation_summaries cs
    WHERE cs.user_id = target_user_id
        AND cs.consolidation_status = 'active'
    ORDER BY cs.created_at DESC
    LIMIT 1;
    
    -- If no conversation summaries exist but conversations do, return basic info
    IF NOT FOUND AND conv_count > 0 THEN
        RETURN QUERY
        SELECT
            'User has conversations but no summaries yet.' as latest_summary,
            conv_count as conversation_count,
            (SELECT id FROM conversations WHERE user_id = target_user_id ORDER BY updated_at DESC LIMIT 1) as latest_conversation_id,
            true as has_history;
    END IF;
    
    -- If no conversations exist at all
    IF NOT FOUND AND conv_count = 0 THEN
        RETURN QUERY
        SELECT
            'New user - no conversation history yet.' as latest_summary,
            0 as conversation_count,
            NULL::UUID as latest_conversation_id,
            false as has_history;
    END IF;
END;
$$;

-- Grant necessary permissions
GRANT EXECUTE ON FUNCTION get_latest_user_conversation_summary(UUID) TO authenticated;
GRANT EXECUTE ON FUNCTION get_user_context_from_conversations(UUID) TO authenticated;

-- Log completion
DO $$ 
BEGIN
    RAISE NOTICE '✅ Master summary system removed successfully';
    RAISE NOTICE '✅ Dropped user_master_summaries table';
    RAISE NOTICE '✅ Dropped related functions: get_or_create_user_master_summary, consolidate_user_summary, get_user_context_for_conversation';
    RAISE NOTICE '✅ Created simplified functions: get_latest_user_conversation_summary, get_user_context_from_conversations';
    RAISE NOTICE '✅ System now uses conversation summaries as the primary memory source';
END
$$; 