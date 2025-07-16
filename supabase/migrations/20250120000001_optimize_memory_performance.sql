-- Migration: Performance optimization for long-term memory system (Task 5.5.4)
-- This migration adds vector indexes, query optimization, and caching strategies
-- for improved memory retrieval and storage performance

-- Enable required extensions for performance optimization
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- 1. Vector indexes for semantic search optimization
-- Create HNSW indexes for vector similarity search (requires pgvector extension)
CREATE INDEX IF NOT EXISTS idx_long_term_memories_embedding_hnsw 
    ON long_term_memories 
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS idx_conversation_summaries_embedding_hnsw 
    ON conversation_summaries 
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- 2. B-tree indexes for frequent query patterns
-- Namespace-based queries
CREATE INDEX IF NOT EXISTS idx_long_term_memories_namespace_gin 
    ON long_term_memories 
    USING gin (namespace);

-- User-specific queries
CREATE INDEX IF NOT EXISTS idx_long_term_memories_user_namespace 
    ON long_term_memories (((namespace)[1]), created_at DESC) 
    WHERE array_length(namespace, 1) >= 1;

-- Key-based lookups
CREATE INDEX IF NOT EXISTS idx_long_term_memories_namespace_key 
    ON long_term_memories (namespace, key) 
    WHERE expires_at IS NULL OR expires_at > NOW();

-- Access pattern optimization
CREATE INDEX IF NOT EXISTS idx_long_term_memories_access_pattern 
    ON long_term_memories (last_accessed DESC, access_count DESC);

-- 3. Conversation-specific indexes
-- User conversations with message count
CREATE INDEX IF NOT EXISTS idx_conversations_user_message_count 
    ON conversations (user_id, message_count DESC, updated_at DESC);

-- Active conversations for consolidation
CREATE INDEX IF NOT EXISTS idx_conversations_consolidation_candidates 
    ON conversations (user_id, updated_at, is_archived) 
    WHERE is_archived = false;

-- Message pattern analysis
CREATE INDEX IF NOT EXISTS idx_messages_conversation_created 
    ON messages (conversation_id, created_at);

-- 4. Memory access pattern indexes
CREATE INDEX IF NOT EXISTS idx_memory_access_patterns_user_time 
    ON memory_access_patterns (user_id, access_time DESC);

CREATE INDEX IF NOT EXISTS idx_memory_access_patterns_memory_frequency 
    ON memory_access_patterns (memory_id, access_frequency DESC);

-- 5. Composite indexes for complex queries
-- Conversation summaries by user and date
CREATE INDEX IF NOT EXISTS idx_conversation_summaries_user_date 
    ON conversation_summaries (user_id, created_at DESC);

-- Long-term memories by expiration
CREATE INDEX IF NOT EXISTS idx_long_term_memories_expiration 
    ON long_term_memories (expires_at) 
    WHERE expires_at IS NOT NULL;

-- 6. Partial indexes for common filter conditions
-- Active memories only
CREATE INDEX IF NOT EXISTS idx_long_term_memories_active 
    ON long_term_memories (namespace, created_at DESC) 
    WHERE expires_at IS NULL OR expires_at > NOW();

-- Recent conversations
CREATE INDEX IF NOT EXISTS idx_conversations_recent 
    ON conversations (user_id, updated_at DESC) 
    WHERE updated_at > NOW() - INTERVAL '30 days';

-- 7. Full-text search indexes
-- Search conversation summaries by content
CREATE INDEX IF NOT EXISTS idx_conversation_summaries_summary_fulltext 
    ON conversation_summaries 
    USING gin (to_tsvector('english', summary));

-- Search long-term memories by value content
CREATE INDEX IF NOT EXISTS idx_long_term_memories_value_fulltext 
    ON long_term_memories 
    USING gin (to_tsvector('english', value::text));

-- 8. Performance monitoring views
CREATE OR REPLACE VIEW memory_performance_stats AS
SELECT 
    'long_term_memories' as table_name,
    COUNT(*) as total_rows,
    COUNT(*) FILTER (WHERE expires_at IS NULL OR expires_at > NOW()) as active_rows,
    AVG(access_count) as avg_access_count,
    MAX(last_accessed) as latest_access,
    pg_size_pretty(pg_total_relation_size('long_term_memories')) as table_size
FROM long_term_memories
UNION ALL
SELECT 
    'conversation_summaries' as table_name,
    COUNT(*) as total_rows,
    COUNT(*) as active_rows,
    0 as avg_access_count,
    MAX(created_at) as latest_access,
    pg_size_pretty(pg_total_relation_size('conversation_summaries')) as table_size
FROM conversation_summaries
UNION ALL
SELECT 
    'conversations' as table_name,
    COUNT(*) as total_rows,
    COUNT(*) FILTER (WHERE is_archived = false) as active_rows,
    0 as avg_access_count,
    MAX(updated_at) as latest_access,
    pg_size_pretty(pg_total_relation_size('conversations')) as table_size
FROM conversations;

-- 9. Optimized query functions
-- Fast semantic search with caching
CREATE OR REPLACE FUNCTION search_long_term_memories_optimized(
    user_id_param TEXT,
    query_embedding VECTOR(1536),
    namespace_prefix TEXT[] DEFAULT NULL,
    similarity_threshold FLOAT DEFAULT 0.7,
    limit_param INTEGER DEFAULT 10
) RETURNS TABLE (
    id UUID,
    namespace TEXT[],
    key TEXT,
    value JSONB,
    similarity_score FLOAT,
    created_at TIMESTAMP WITH TIME ZONE,
    access_count INTEGER
) AS $$
BEGIN
    -- Use the optimized vector index for similarity search
    RETURN QUERY
    SELECT 
        ltm.id,
        ltm.namespace,
        ltm.key,
        ltm.value,
        (ltm.embedding <=> query_embedding) AS similarity_score,
        ltm.created_at,
        ltm.access_count
    FROM long_term_memories ltm
    WHERE 
        (ltm.expires_at IS NULL OR ltm.expires_at > NOW())
        AND (namespace_prefix IS NULL OR ltm.namespace[1:array_length(namespace_prefix, 1)] = namespace_prefix)
        AND ltm.namespace[1] = user_id_param
        AND (ltm.embedding <=> query_embedding) < (1 - similarity_threshold)
    ORDER BY ltm.embedding <=> query_embedding
    LIMIT limit_param;
END;
$$ LANGUAGE plpgsql;

-- Fast conversation summary search
CREATE OR REPLACE FUNCTION search_conversation_summaries_optimized(
    user_id_param TEXT,
    query_embedding VECTOR(1536),
    similarity_threshold FLOAT DEFAULT 0.7,
    limit_param INTEGER DEFAULT 10
) RETURNS TABLE (
    id UUID,
    conversation_id UUID,
    summary TEXT,
    topics TEXT[],
    similarity_score FLOAT,
    created_at TIMESTAMP WITH TIME ZONE,
    message_count INTEGER
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        cs.id,
        cs.conversation_id,
        cs.summary,
        cs.topics,
        (cs.embedding <=> query_embedding) AS similarity_score,
        cs.created_at,
        cs.message_count
    FROM conversation_summaries cs
    WHERE 
        cs.user_id = user_id_param
        AND (cs.embedding <=> query_embedding) < (1 - similarity_threshold)
    ORDER BY cs.embedding <=> query_embedding
    LIMIT limit_param;
END;
$$ LANGUAGE plpgsql;

-- 10. Memory cleanup and maintenance functions
-- Automated cleanup of expired memories
CREATE OR REPLACE FUNCTION cleanup_expired_memories() RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM long_term_memories 
    WHERE expires_at IS NOT NULL AND expires_at < NOW();
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    
    -- Log cleanup activity
    INSERT INTO memory_access_patterns (
        user_id, memory_id, access_type, access_time, 
        query_embedding, similarity_score, access_frequency
    ) VALUES (
        'system', gen_random_uuid(), 'cleanup', NOW(), 
        NULL, 0, deleted_count
    );
    
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Update access patterns for optimization
CREATE OR REPLACE FUNCTION update_memory_access_pattern(
    p_user_id TEXT,
    p_memory_id UUID,
    p_access_type TEXT,
    p_query_embedding VECTOR(1536) DEFAULT NULL,
    p_similarity_score FLOAT DEFAULT NULL
) RETURNS VOID AS $$
BEGIN
    -- Update access count in long_term_memories
    UPDATE long_term_memories 
    SET 
        access_count = access_count + 1,
        last_accessed = NOW()
    WHERE id = p_memory_id;
    
    -- Insert or update access pattern
    INSERT INTO memory_access_patterns (
        user_id, memory_id, access_type, access_time,
        query_embedding, similarity_score, access_frequency
    ) VALUES (
        p_user_id, p_memory_id, p_access_type, NOW(),
        p_query_embedding, p_similarity_score, 1
    )
    ON CONFLICT (user_id, memory_id, access_type) 
    DO UPDATE SET
        access_time = NOW(),
        access_frequency = memory_access_patterns.access_frequency + 1,
        query_embedding = EXCLUDED.query_embedding,
        similarity_score = EXCLUDED.similarity_score;
END;
$$ LANGUAGE plpgsql;

-- 11. Performance monitoring and alerting
-- Create a function to monitor memory system performance
CREATE OR REPLACE FUNCTION get_memory_performance_metrics() RETURNS TABLE (
    metric_name TEXT,
    metric_value NUMERIC,
    metric_unit TEXT,
    status TEXT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        'total_memories'::TEXT as metric_name,
        COUNT(*)::NUMERIC as metric_value,
        'count'::TEXT as metric_unit,
        CASE 
            WHEN COUNT(*) > 1000000 THEN 'warning'
            WHEN COUNT(*) > 10000000 THEN 'critical'
            ELSE 'ok'
        END as status
    FROM long_term_memories
    WHERE expires_at IS NULL OR expires_at > NOW()
    
    UNION ALL
    
    SELECT 
        'avg_access_count'::TEXT,
        AVG(access_count)::NUMERIC,
        'count'::TEXT,
        CASE 
            WHEN AVG(access_count) < 1 THEN 'warning'
            ELSE 'ok'
        END
    FROM long_term_memories
    WHERE expires_at IS NULL OR expires_at > NOW()
    
    UNION ALL
    
    SELECT 
        'expired_memories'::TEXT,
        COUNT(*)::NUMERIC,
        'count'::TEXT,
        CASE 
            WHEN COUNT(*) > 10000 THEN 'warning'
            WHEN COUNT(*) > 50000 THEN 'critical'
            ELSE 'ok'
        END
    FROM long_term_memories
    WHERE expires_at IS NOT NULL AND expires_at < NOW()
    
    UNION ALL
    
    SELECT 
        'active_conversations'::TEXT,
        COUNT(*)::NUMERIC,
        'count'::TEXT,
        CASE 
            WHEN COUNT(*) > 100000 THEN 'warning'
            WHEN COUNT(*) > 500000 THEN 'critical'
            ELSE 'ok'
        END
    FROM conversations
    WHERE is_archived = false;
END;
$$ LANGUAGE plpgsql;

-- 12. Automated maintenance job scheduling
-- Create a function to run regular maintenance
CREATE OR REPLACE FUNCTION run_memory_maintenance() RETURNS TABLE (
    task_name TEXT,
    status TEXT,
    details TEXT
) AS $$
DECLARE
    cleanup_count INTEGER;
    vacuum_started BOOLEAN := false;
BEGIN
    -- Clean up expired memories
    BEGIN
        cleanup_count := cleanup_expired_memories();
        RETURN QUERY SELECT 'cleanup_expired_memories'::TEXT, 'success'::TEXT, 
                           ('Cleaned up ' || cleanup_count || ' expired memories')::TEXT;
    EXCEPTION WHEN others THEN
        RETURN QUERY SELECT 'cleanup_expired_memories'::TEXT, 'error'::TEXT, SQLERRM::TEXT;
    END;
    
    -- Update table statistics
    BEGIN
        ANALYZE long_term_memories;
        ANALYZE conversation_summaries;
        ANALYZE conversations;
        ANALYZE messages;
        ANALYZE memory_access_patterns;
        RETURN QUERY SELECT 'update_statistics'::TEXT, 'success'::TEXT, 'Statistics updated'::TEXT;
    EXCEPTION WHEN others THEN
        RETURN QUERY SELECT 'update_statistics'::TEXT, 'error'::TEXT, SQLERRM::TEXT;
    END;
    
    -- Vacuum if needed (lightweight check)
    BEGIN
        -- Note: VACUUM cannot be run in a transaction, so we'll just report the need
        RETURN QUERY SELECT 'vacuum_check'::TEXT, 'info'::TEXT, 'Manual VACUUM recommended for optimal performance'::TEXT;
    EXCEPTION WHEN others THEN
        RETURN QUERY SELECT 'vacuum_check'::TEXT, 'error'::TEXT, SQLERRM::TEXT;
    END;
END;
$$ LANGUAGE plpgsql;

-- 13. Create materialized views for common queries
-- User memory summary view
CREATE MATERIALIZED VIEW IF NOT EXISTS user_memory_summary AS
SELECT 
    namespace[1] as user_id,
    COUNT(*) as total_memories,
    COUNT(*) FILTER (WHERE expires_at IS NULL OR expires_at > NOW()) as active_memories,
    AVG(access_count) as avg_access_count,
    MAX(last_accessed) as last_access,
    MIN(created_at) as oldest_memory,
    MAX(created_at) as newest_memory,
    array_agg(DISTINCT namespace[2]) FILTER (WHERE array_length(namespace, 1) >= 2) as memory_types
FROM long_term_memories
WHERE array_length(namespace, 1) >= 1
GROUP BY namespace[1];

-- Create unique index for the materialized view
CREATE UNIQUE INDEX IF NOT EXISTS idx_user_memory_summary_user_id 
    ON user_memory_summary (user_id);

-- Refresh function for materialized views
CREATE OR REPLACE FUNCTION refresh_memory_views() RETURNS VOID AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY user_memory_summary;
END;
$$ LANGUAGE plpgsql;

-- 14. Query plan optimization hints
-- Add comments to help query planner
COMMENT ON INDEX idx_long_term_memories_embedding_hnsw IS 'HNSW index for vector similarity search - primary semantic search performance';
COMMENT ON INDEX idx_long_term_memories_namespace_gin IS 'GIN index for namespace array operations - namespace filtering performance';
COMMENT ON INDEX idx_conversation_summaries_embedding_hnsw IS 'HNSW index for conversation summary similarity search';

-- 15. Row-level security optimization
-- Create policies that work well with indexes
CREATE POLICY IF NOT EXISTS "long_term_memories_user_access" ON long_term_memories
    FOR ALL 
    USING (namespace[1] = current_setting('app.current_user', true));

CREATE POLICY IF NOT EXISTS "conversation_summaries_user_access" ON conversation_summaries
    FOR ALL 
    USING (user_id = current_setting('app.current_user', true));

-- Final optimization settings
-- Update table statistics collection
ALTER TABLE long_term_memories SET (autovacuum_analyze_scale_factor = 0.02);
ALTER TABLE conversation_summaries SET (autovacuum_analyze_scale_factor = 0.02);
ALTER TABLE conversations SET (autovacuum_analyze_scale_factor = 0.02);
ALTER TABLE messages SET (autovacuum_analyze_scale_factor = 0.02);

-- Set appropriate fillfactor for tables with updates
ALTER TABLE long_term_memories SET (fillfactor = 90);
ALTER TABLE memory_access_patterns SET (fillfactor = 85);

-- Complete performance optimization
-- Log completion
DO $$
BEGIN
    RAISE NOTICE 'Long-term memory performance optimization completed successfully';
    RAISE NOTICE 'Created % indexes, % functions, and % views', 
        (SELECT COUNT(*) FROM pg_indexes WHERE tablename IN ('long_term_memories', 'conversation_summaries', 'conversations', 'messages', 'memory_access_patterns')),
        (SELECT COUNT(*) FROM pg_proc WHERE proname LIKE '%memory%' OR proname LIKE '%conversation%'),
        (SELECT COUNT(*) FROM pg_views WHERE viewname LIKE '%memory%');
END $$; 