-- Fix column naming issues and UUID format problems

-- 1. Fix get_long_term_memory function - use correct column name
DROP FUNCTION IF EXISTS get_long_term_memory(text[], text);
CREATE OR REPLACE FUNCTION get_long_term_memory(
    p_namespace TEXT[],
    p_key TEXT
) RETURNS TABLE (
    namespace TEXT[],
    key TEXT,
    value JSONB,
    created_at TIMESTAMP WITH TIME ZONE,
    updated_at TIMESTAMP WITH TIME ZONE
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        ltm.namespace,
        ltm.key,
        ltm.value,
        ltm.created_at,
        ltm.updated_at
    FROM long_term_memories ltm
    WHERE 
        ltm.namespace = p_namespace
        AND ltm.key = p_key
        AND (ltm.expiry_at IS NULL OR ltm.expiry_at > NOW())  -- Fixed: expiry_at not expires_at
    ORDER BY ltm.updated_at DESC
    LIMIT 1;
END;
$$ LANGUAGE plpgsql;

-- 2. Fix put_long_term_memory function - use correct column name
DROP FUNCTION IF EXISTS put_long_term_memory(text[], text, jsonb, vector, varchar, timestamp with time zone);
CREATE OR REPLACE FUNCTION put_long_term_memory(
    p_namespace TEXT[],
    p_key TEXT,
    p_value JSONB,
    p_embedding VECTOR(1536),
    p_memory_type VARCHAR(50) DEFAULT 'general',
    p_expiry_at TIMESTAMP WITH TIME ZONE DEFAULT NULL
) RETURNS UUID AS $$
DECLARE
    memory_id UUID;
BEGIN
    -- Insert or update the memory
    INSERT INTO long_term_memories (
        namespace,
        key,
        value,
        embedding,
        memory_type,
        expiry_at  -- Fixed: expiry_at not expires_at
    ) VALUES (
        p_namespace,
        p_key,
        p_value,
        p_embedding,
        p_memory_type,
        p_expiry_at
    )
    ON CONFLICT (namespace, key) DO UPDATE SET
        value = p_value,
        embedding = p_embedding,
        memory_type = p_memory_type,
        expiry_at = p_expiry_at,  -- Fixed: expiry_at not expires_at
        updated_at = NOW()
    RETURNING id INTO memory_id;
    
    -- Update access pattern
    PERFORM update_memory_access_pattern(
        p_namespace,
        p_key,
        'write',
        1,
        NOW()
    );
    
    RETURN memory_id;
END;
$$ LANGUAGE plpgsql;

-- 3. Fix memory_exists function - use correct column name
DROP FUNCTION IF EXISTS memory_exists(text[], text);
CREATE OR REPLACE FUNCTION memory_exists(
    p_namespace TEXT[],
    p_key TEXT
) RETURNS BOOLEAN AS $$
DECLARE
    result BOOLEAN;
BEGIN
    SELECT EXISTS(
        SELECT 1 FROM long_term_memories 
        WHERE namespace = p_namespace 
        AND key = p_key
        AND (expiry_at IS NULL OR expiry_at > NOW())  -- Fixed: expiry_at not expires_at
    ) INTO result;
    
    RETURN result;
END;
$$ LANGUAGE plpgsql;

-- 4. Grant permissions
GRANT EXECUTE ON FUNCTION get_long_term_memory(text[], text) TO authenticated;
GRANT EXECUTE ON FUNCTION put_long_term_memory(text[], text, jsonb, vector, varchar, timestamp with time zone) TO authenticated;
GRANT EXECUTE ON FUNCTION memory_exists(text[], text) TO authenticated; 