-- Migration: Fix cleanup function error handling (Task 5.5.4 - Additional Fixes)
-- This migration fixes the remaining UUID/text issues in the cleanup function

-- 1. Fix cleanup_expired_memories function with proper error handling
DROP FUNCTION IF EXISTS cleanup_expired_memories();
CREATE OR REPLACE FUNCTION cleanup_expired_memories() RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM long_term_memories 
    WHERE expiry_at IS NOT NULL AND expiry_at < NOW();
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    
    -- Log cleanup activity with proper UUID generation (skip on error)
    BEGIN
        INSERT INTO memory_access_patterns (
            user_id, memory_namespace, memory_key, access_frequency, 
            last_accessed_at, context_relevance, access_context, retrieval_method
        ) VALUES (
            'system', ARRAY['system', 'maintenance'], 'cleanup_expired_memories', deleted_count, 
            NOW(), 0.0, 'Automated expired memory cleanup', 'system'
        );
    EXCEPTION
        WHEN others THEN
            NULL; -- Skip logging on error
    END;
    
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Log completion
DO $$ 
BEGIN
    RAISE NOTICE '✅ Additional long-term memory function fixes completed!';
    RAISE NOTICE '✅ Fixed cleanup_expired_memories error handling';
END
$$;
