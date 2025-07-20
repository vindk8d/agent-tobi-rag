-- Fix user context function to properly handle TEXT user_id to UUID lookup
-- The conversation_summaries.user_id field stores UUIDs from users.id, 
-- but the API passes TEXT user_id values like "customer-robert-004"

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
DECLARE
    target_uuid UUID;
BEGIN
    -- First, convert the TEXT user_id to UUID by looking up in users table
    SELECT u.id INTO target_uuid 
    FROM users u 
    WHERE u.user_id = target_user_id OR u.id::TEXT = target_user_id;
    
    -- If no user found, return empty result for this specific user_id
    IF target_uuid IS NULL THEN
        RETURN QUERY
        SELECT 
            target_user_id as user_id,
            'No conversation history available'::TEXT as latest_summary,
            0::INTEGER as conversation_count,
            FALSE::BOOLEAN as has_history,
            NULL::TEXT as latest_conversation_id;
        RETURN;
    END IF;

    -- Now query conversation_summaries using the UUID
    RETURN QUERY
    WITH conversation_stats AS (
        SELECT 
            cs.user_id,
            COUNT(*) as total_conversations,
            CASE WHEN COUNT(*) > 0 THEN TRUE ELSE FALSE END as has_history
        FROM conversation_summaries cs
        WHERE cs.user_id = target_uuid
        GROUP BY cs.user_id
    ),
    latest_summary AS (
        SELECT 
            cs.user_id,
            cs.summary_text,
            cs.conversation_id,
            ROW_NUMBER() OVER (ORDER BY cs.created_at DESC) as rn
        FROM conversation_summaries cs
        WHERE cs.user_id = target_uuid
    )
    SELECT 
        target_user_id as user_id,  -- Return the original input user_id
        COALESCE(latest.summary_text, 'No conversation history available')::TEXT as latest_summary,
        COALESCE(stats.total_conversations, 0)::INTEGER as conversation_count,
        COALESCE(stats.has_history, FALSE)::BOOLEAN as has_history,
        latest.conversation_id::TEXT as latest_conversation_id
    FROM conversation_stats stats
    FULL OUTER JOIN latest_summary latest ON stats.user_id = latest.user_id AND latest.rn = 1;
END;
$$;
