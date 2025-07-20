-- Fix get_user_context_from_conversations function after user_id → username rename
-- This function is used by the memory debug API and was referencing the old user_id column

CREATE OR REPLACE FUNCTION public.get_user_context_from_conversations(target_user_id text)
RETURNS TABLE(user_id text, latest_summary text, conversation_count integer, has_history boolean, latest_conversation_id text)
LANGUAGE plpgsql
AS $function$
DECLARE
    target_uuid UUID;
    summary_text TEXT;
    conv_count INTEGER;
    latest_conv_id TEXT;
BEGIN
    -- Convert TEXT user_id to UUID by looking up in users table
    -- Check both id (UUID) and username (string) since system passes UUIDs but may also get usernames
    SELECT u.id INTO target_uuid 
    FROM users u 
    WHERE u.username = target_user_id OR u.id::TEXT = target_user_id;
    
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
$function$;

-- Add comment to clarify the function's purpose
COMMENT ON FUNCTION get_user_context_from_conversations(text) IS 'Get user context from conversations. Accepts either users.id (UUID) or users.username (string)';
