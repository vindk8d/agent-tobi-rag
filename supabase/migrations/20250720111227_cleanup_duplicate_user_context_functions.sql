-- Clean up duplicate function definitions that are causing overloading conflicts
-- Drop all existing versions of get_user_context_from_conversations

DROP FUNCTION IF EXISTS get_user_context_from_conversations(TEXT);
DROP FUNCTION IF EXISTS get_user_context_from_conversations(UUID);
DROP FUNCTION IF EXISTS get_user_context_from_conversations(target_user_id TEXT);
DROP FUNCTION IF EXISTS get_user_context_from_conversations(target_user_id UUID);

-- Create single clean version that handles TEXT user_id input
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
    summary_text TEXT;
    conv_count INTEGER;
    latest_conv_id TEXT;
BEGIN
    -- Convert TEXT user_id to UUID by looking up in users table
    SELECT u.id INTO target_uuid 
    FROM users u 
    WHERE u.user_id = target_user_id OR u.id::TEXT = target_user_id;
    
    -- If no user found, return empty result
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

    -- Get conversation count and latest summary
    SELECT COUNT(*) INTO conv_count
    FROM conversation_summaries cs
    WHERE cs.user_id = target_uuid;
    
    SELECT cs.summary_text, cs.conversation_id INTO summary_text, latest_conv_id
    FROM conversation_summaries cs
    WHERE cs.user_id = target_uuid
    ORDER BY cs.created_at DESC
    LIMIT 1;

    -- Return single result
    RETURN QUERY
    SELECT 
        target_user_id as user_id,
        COALESCE(summary_text, 'No conversation history available')::TEXT as latest_summary,
        COALESCE(conv_count, 0)::INTEGER as conversation_count,
        CASE WHEN conv_count > 0 THEN TRUE ELSE FALSE END::BOOLEAN as has_history,
        latest_conv_id::TEXT as latest_conversation_id;
END;
$$;

-- Grant permissions
GRANT EXECUTE ON FUNCTION get_user_context_from_conversations(TEXT) TO anon, authenticated;
