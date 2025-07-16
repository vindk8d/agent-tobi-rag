-- Migration: Add long-term memory support for LangGraph Store integration (Task 5.5.4)
-- This migration implements hybrid memory system combining thread-scoped and cross-thread memory
-- Following LangGraph memory best practices from: https://langchain-ai.github.io/langgraph/concepts/memory/

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- 1. Long-term memory storage table for cross-thread access
-- Implements LangGraph Store interface with namespace-based organization
CREATE TABLE IF NOT EXISTS long_term_memories (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    namespace TEXT[] NOT NULL,                    -- Namespace hierarchy (e.g., ['user', 'user123', 'preferences'])
    key TEXT NOT NULL,                           -- Unique key within namespace
    value JSONB NOT NULL,                        -- Structured memory content
    embedding VECTOR(1536),                      -- OpenAI embedding for semantic search
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    accessed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    access_count INTEGER DEFAULT 0,
    memory_type VARCHAR(50) DEFAULT 'semantic',  -- 'semantic', 'episodic', 'procedural'
    source_thread_id TEXT,                       -- Optional source thread for tracking
    expiry_at TIMESTAMP WITH TIME ZONE,          -- Optional expiry for temporary memories
    metadata JSONB DEFAULT '{}'::jsonb,
    UNIQUE(namespace, key)
);

-- 2. Conversation summaries table for episodic memory
-- Stores periodic summaries of conversations for efficient context retrieval
CREATE TABLE IF NOT EXISTS conversation_summaries (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    conversation_id UUID REFERENCES conversations(id) ON DELETE CASCADE,
    user_id TEXT NOT NULL,
    summary_text TEXT NOT NULL,
    summary_type VARCHAR(50) DEFAULT 'periodic',  -- 'periodic', 'final', 'topic_based'
    message_count INTEGER NOT NULL,
    start_message_id UUID,                        -- First message in summary range
    end_message_id UUID,                          -- Last message in summary range
    summary_embedding VECTOR(1536),              -- Embedding for semantic search
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    consolidation_status VARCHAR(20) DEFAULT 'active', -- 'active', 'archived', 'consolidated'
    metadata JSONB DEFAULT '{}'::jsonb
);

-- 3. Memory access patterns table for optimization
-- Tracks access patterns to optimize memory retrieval performance
CREATE TABLE IF NOT EXISTS memory_access_patterns (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id TEXT NOT NULL,
    memory_namespace TEXT[] NOT NULL,
    memory_key TEXT,
    access_frequency INTEGER DEFAULT 1,
    last_accessed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    context_relevance FLOAT DEFAULT 0.0,
    access_context TEXT,                          -- Context when memory was accessed
    retrieval_method VARCHAR(50),                 -- 'semantic', 'direct', 'filtered'
    metadata JSONB DEFAULT '{}'::jsonb
);

-- 4. Enhance existing conversations table with summary support
-- Add fields for conversation summarization and archival
ALTER TABLE conversations 
ADD COLUMN IF NOT EXISTS summary_text TEXT,
ADD COLUMN IF NOT EXISTS summary_embedding VECTOR(1536),
ADD COLUMN IF NOT EXISTS last_summary_at TIMESTAMP WITH TIME ZONE,
ADD COLUMN IF NOT EXISTS message_count INTEGER DEFAULT 0,
ADD COLUMN IF NOT EXISTS archived_at TIMESTAMP WITH TIME ZONE,
ADD COLUMN IF NOT EXISTS archival_status VARCHAR(20) DEFAULT 'active' 
    CHECK (archival_status IN ('active', 'archived', 'consolidated'));

-- 5. Create indexes for performance optimization
-- Long-term memories indexes
CREATE INDEX IF NOT EXISTS idx_long_term_memories_namespace ON long_term_memories USING GIN(namespace);
CREATE INDEX IF NOT EXISTS idx_long_term_memories_embedding ON long_term_memories USING ivfflat (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_long_term_memories_accessed_at ON long_term_memories(accessed_at);
CREATE INDEX IF NOT EXISTS idx_long_term_memories_memory_type ON long_term_memories(memory_type);
CREATE INDEX IF NOT EXISTS idx_long_term_memories_source_thread ON long_term_memories(source_thread_id);
CREATE INDEX IF NOT EXISTS idx_long_term_memories_expiry ON long_term_memories(expiry_at) WHERE expiry_at IS NOT NULL;

-- Conversation summaries indexes
CREATE INDEX IF NOT EXISTS idx_conversation_summaries_user_id ON conversation_summaries(user_id);
CREATE INDEX IF NOT EXISTS idx_conversation_summaries_conversation_id ON conversation_summaries(conversation_id);
CREATE INDEX IF NOT EXISTS idx_conversation_summaries_embedding ON conversation_summaries USING ivfflat (summary_embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_conversation_summaries_created_at ON conversation_summaries(created_at);
CREATE INDEX IF NOT EXISTS idx_conversation_summaries_type ON conversation_summaries(summary_type);
CREATE INDEX IF NOT EXISTS idx_conversation_summaries_status ON conversation_summaries(consolidation_status);

-- Memory access patterns indexes
CREATE INDEX IF NOT EXISTS idx_memory_access_patterns_user_id ON memory_access_patterns(user_id);
CREATE INDEX IF NOT EXISTS idx_memory_access_patterns_namespace ON memory_access_patterns USING GIN(memory_namespace);
CREATE INDEX IF NOT EXISTS idx_memory_access_patterns_frequency ON memory_access_patterns(access_frequency);
CREATE INDEX IF NOT EXISTS idx_memory_access_patterns_last_accessed ON memory_access_patterns(last_accessed_at);

-- Enhanced conversations table indexes
CREATE INDEX IF NOT EXISTS idx_conversations_summary_embedding ON conversations USING ivfflat (summary_embedding vector_cosine_ops) WHERE summary_embedding IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_conversations_archival_status ON conversations(archival_status);
CREATE INDEX IF NOT EXISTS idx_conversations_message_count ON conversations(message_count);
CREATE INDEX IF NOT EXISTS idx_conversations_archived_at ON conversations(archived_at) WHERE archived_at IS NOT NULL;

-- 6. Database functions for memory operations
-- Function for semantic search of long-term memories
CREATE OR REPLACE FUNCTION search_long_term_memories(
    query_embedding VECTOR(1536),
    namespace_filter TEXT[] DEFAULT NULL,
    similarity_threshold FLOAT DEFAULT 0.7,
    match_count INTEGER DEFAULT 10,
    memory_type_filter VARCHAR(50) DEFAULT NULL,
    exclude_expired BOOLEAN DEFAULT TRUE
)
RETURNS TABLE (
    id UUID,
    key TEXT,
    value JSONB,
    similarity FLOAT,
    namespace TEXT[],
    memory_type VARCHAR(50),
    created_at TIMESTAMP WITH TIME ZONE,
    access_count INTEGER
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        ltm.id,
        ltm.key,
        ltm.value,
        (1 - (ltm.embedding <=> query_embedding))::FLOAT AS similarity,
        ltm.namespace,
        ltm.memory_type,
        ltm.created_at,
        ltm.access_count
    FROM long_term_memories ltm
    WHERE ltm.embedding IS NOT NULL
        AND (namespace_filter IS NULL OR ltm.namespace = namespace_filter)
        AND (memory_type_filter IS NULL OR ltm.memory_type = memory_type_filter)
        AND (NOT exclude_expired OR ltm.expiry_at IS NULL OR ltm.expiry_at > NOW())
        AND (1 - (ltm.embedding <=> query_embedding)) > similarity_threshold
    ORDER BY similarity DESC
    LIMIT match_count;
END;
$$;

-- Function to search conversation summaries for context
CREATE OR REPLACE FUNCTION search_conversation_summaries(
    query_embedding VECTOR(1536),
    target_user_id TEXT,
    similarity_threshold FLOAT DEFAULT 0.7,
    match_count INTEGER DEFAULT 5,
    summary_type_filter VARCHAR(50) DEFAULT NULL
)
RETURNS TABLE (
    id UUID,
    conversation_id UUID,
    summary_text TEXT,
    similarity FLOAT,
    message_count INTEGER,
    created_at TIMESTAMP WITH TIME ZONE,
    summary_type VARCHAR(50)
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        cs.id,
        cs.conversation_id,
        cs.summary_text,
        (1 - (cs.summary_embedding <=> query_embedding))::FLOAT AS similarity,
        cs.message_count,
        cs.created_at,
        cs.summary_type
    FROM conversation_summaries cs
    WHERE cs.summary_embedding IS NOT NULL
        AND cs.user_id = target_user_id
        AND cs.consolidation_status = 'active'
        AND (summary_type_filter IS NULL OR cs.summary_type = summary_type_filter)
        AND (1 - (cs.summary_embedding <=> query_embedding)) > similarity_threshold
    ORDER BY similarity DESC
    LIMIT match_count;
END;
$$;

-- Function to get recent conversation summaries for a user
CREATE OR REPLACE FUNCTION get_recent_conversation_summaries(
    target_user_id TEXT,
    limit_count INTEGER DEFAULT 10,
    days_back INTEGER DEFAULT 30
)
RETURNS TABLE (
    conversation_id UUID,
    summary_text TEXT,
    message_count INTEGER,
    created_at TIMESTAMP WITH TIME ZONE,
    summary_type VARCHAR(50)
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        cs.conversation_id,
        cs.summary_text,
        cs.message_count,
        cs.created_at,
        cs.summary_type
    FROM conversation_summaries cs
    WHERE cs.user_id = target_user_id
        AND cs.consolidation_status = 'active'
        AND cs.created_at > NOW() - INTERVAL '1 day' * days_back
    ORDER BY cs.created_at DESC
    LIMIT limit_count;
END;
$$;

-- Function to update memory access patterns
CREATE OR REPLACE FUNCTION update_memory_access_pattern(
    target_user_id TEXT,
    memory_namespace TEXT[],
    memory_key TEXT DEFAULT NULL,
    access_context TEXT DEFAULT NULL,
    retrieval_method VARCHAR(50) DEFAULT 'direct'
)
RETURNS VOID
LANGUAGE plpgsql
AS $$
BEGIN
    -- Update existing pattern or insert new one
    INSERT INTO memory_access_patterns (
        user_id, memory_namespace, memory_key, access_frequency, 
        last_accessed_at, access_context, retrieval_method
    )
    VALUES (
        target_user_id, memory_namespace, memory_key, 1,
        NOW(), access_context, retrieval_method
    )
    ON CONFLICT (user_id, memory_namespace, memory_key)
    DO UPDATE SET
        access_frequency = memory_access_patterns.access_frequency + 1,
        last_accessed_at = NOW(),
        access_context = EXCLUDED.access_context,
        retrieval_method = EXCLUDED.retrieval_method;
END;
$$;

-- 7. Triggers for maintaining data consistency
-- Trigger to update conversation message count
CREATE OR REPLACE FUNCTION update_conversation_message_count()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        UPDATE conversations 
        SET message_count = message_count + 1,
            updated_at = NOW()
        WHERE id = NEW.conversation_id;
        RETURN NEW;
    ELSIF TG_OP = 'DELETE' THEN
        UPDATE conversations 
        SET message_count = GREATEST(message_count - 1, 0),
            updated_at = NOW()
        WHERE id = OLD.conversation_id;
        RETURN OLD;
    END IF;
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

-- Create trigger for message count maintenance
DROP TRIGGER IF EXISTS update_conversation_message_count_trigger ON messages;
CREATE TRIGGER update_conversation_message_count_trigger
    AFTER INSERT OR DELETE ON messages
    FOR EACH ROW
    EXECUTE FUNCTION update_conversation_message_count();

-- Trigger to update memory access timestamp
CREATE OR REPLACE FUNCTION update_memory_access_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.accessed_at = NOW();
    NEW.access_count = NEW.access_count + 1;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger for memory access tracking
DROP TRIGGER IF EXISTS update_memory_access_trigger ON long_term_memories;
CREATE TRIGGER update_memory_access_trigger
    BEFORE UPDATE ON long_term_memories
    FOR EACH ROW
    EXECUTE FUNCTION update_memory_access_timestamp();

-- 8. Row Level Security (RLS) policies for long-term memory
-- Enable RLS on new tables
ALTER TABLE long_term_memories ENABLE ROW LEVEL SECURITY;
ALTER TABLE conversation_summaries ENABLE ROW LEVEL SECURITY;
ALTER TABLE memory_access_patterns ENABLE ROW LEVEL SECURITY;

-- Long-term memories policies
CREATE POLICY "Users can access their own memories"
ON long_term_memories FOR ALL
TO authenticated
USING (
    -- Extract user_id from namespace (assuming format: ['user', 'user123', ...])
    CASE 
        WHEN array_length(namespace, 1) >= 2 AND namespace[1] = 'user' 
        THEN namespace[2] = auth.uid()::text
        ELSE false
    END
);

CREATE POLICY "Service role full access to memories"
ON long_term_memories FOR ALL
TO service_role
USING (true);

-- Conversation summaries policies
CREATE POLICY "Users can access their own conversation summaries"
ON conversation_summaries FOR ALL
TO authenticated
USING (user_id = auth.uid()::text);

CREATE POLICY "Service role full access to conversation summaries"
ON conversation_summaries FOR ALL
TO service_role
USING (true);

-- Memory access patterns policies
CREATE POLICY "Users can access their own memory patterns"
ON memory_access_patterns FOR ALL
TO authenticated
USING (user_id = auth.uid()::text);

CREATE POLICY "Service role full access to memory patterns"
ON memory_access_patterns FOR ALL
TO service_role
USING (true);

-- 9. Comments for documentation
COMMENT ON TABLE long_term_memories IS 'Long-term memory storage for LangGraph Store with cross-thread access capability';
COMMENT ON TABLE conversation_summaries IS 'Episodic memory storage for conversation summaries with semantic search';
COMMENT ON TABLE memory_access_patterns IS 'Optimization tracking for memory access patterns and usage analytics';

COMMENT ON COLUMN long_term_memories.namespace IS 'Hierarchical namespace for memory organization (e.g., [user, user123, preferences])';
COMMENT ON COLUMN long_term_memories.key IS 'Unique identifier within namespace';
COMMENT ON COLUMN long_term_memories.value IS 'Structured memory content as JSON';
COMMENT ON COLUMN long_term_memories.embedding IS 'OpenAI embedding for semantic search';
COMMENT ON COLUMN long_term_memories.memory_type IS 'Type of memory: semantic, episodic, or procedural';

COMMENT ON COLUMN conversation_summaries.summary_type IS 'Type of summary: periodic, final, or topic_based';
COMMENT ON COLUMN conversation_summaries.consolidation_status IS 'Status: active, archived, or consolidated';

-- Log migration completion
DO $$ 
BEGIN
    RAISE NOTICE '✅ Long-term memory tables created successfully!';
    RAISE NOTICE '✅ Added long_term_memories table with namespace organization';
    RAISE NOTICE '✅ Added conversation_summaries table for episodic memory';
    RAISE NOTICE '✅ Added memory_access_patterns table for optimization';
    RAISE NOTICE '✅ Enhanced conversations table with summary support';
    RAISE NOTICE '✅ Created semantic search functions for memory retrieval';
    RAISE NOTICE '✅ Added indexes for performance optimization';
    RAISE NOTICE '✅ Configured RLS policies for secure access';
    RAISE NOTICE '✅ Task 5.5.4 Database Schema - COMPLETED';
END
$$; 