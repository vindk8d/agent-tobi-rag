-- Fix UUID casting issue in get_user_context_from_conversations function
-- The conversation_summaries.user_id column is UUID type but the function parameter is TEXT
-- This causes comparison failures and prevents conversation summaries from being returned

CREATE OR REPLACE FUNCTION get_user_context_from_conversations(target_user_id TEXT)
RETURNS TABLE (
    user_id TEXT,
    latest_summary TEXT,
    conversation_count INTEGER,
    has_history BOOLEAN,
    latest_conversation_id TEXT
) 
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    WITH conversation_stats AS (
        SELECT 
            cs.user_id,
            COUNT(*) as total_conversations,
            CASE WHEN COUNT(*) > 0 THEN TRUE ELSE FALSE END as has_history
        FROM conversation_summaries cs
        WHERE cs.user_id = target_user_id::UUID
        GROUP BY cs.user_id
    ),
    latest_summary AS (
        SELECT 
            cs.user_id,
            cs.summary_text,
            cs.conversation_id,
            ROW_NUMBER() OVER (ORDER BY cs.created_at DESC) as rn
        FROM conversation_summaries cs
        WHERE cs.user_id = target_user_id::UUID
    )
    SELECT 
        COALESCE(stats.user_id::TEXT, target_user_id) as user_id,
        COALESCE(latest.summary_text, 'No conversation history available')::TEXT as latest_summary,
        COALESCE(stats.total_conversations, 0)::INTEGER as conversation_count,
        COALESCE(stats.has_history, FALSE)::BOOLEAN as has_history,
        latest.conversation_id::TEXT as latest_conversation_id
    FROM conversation_stats stats
    FULL OUTER JOIN latest_summary latest ON stats.user_id = latest.user_id AND latest.rn = 1;
END;
$$;
