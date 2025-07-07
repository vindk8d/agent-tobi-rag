-- Create table for tracking source conflicts
CREATE TABLE source_conflicts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    query_text TEXT NOT NULL,
    document_ids UUID[] NOT NULL,
    conflict_type VARCHAR(50) NOT NULL,
    confidence_scores FLOAT[] NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Create table for tracking system performance metrics
CREATE TABLE system_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    metric_name VARCHAR(100) NOT NULL,
    metric_value FLOAT NOT NULL,
    metric_type VARCHAR(50) NOT NULL, -- 'response_time', 'accuracy', 'resolution_rate', etc.
    recorded_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Create table for user feedback on responses
CREATE TABLE response_feedback (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    query_log_id UUID REFERENCES query_logs(id) ON DELETE CASCADE,
    user_id TEXT NOT NULL,
    rating INTEGER CHECK (rating >= 1 AND rating <= 5),
    feedback_text TEXT,
    helpful BOOLEAN,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create table for proactive suggestions
CREATE TABLE proactive_suggestions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    conversation_id UUID REFERENCES conversations(id) ON DELETE CASCADE,
    suggestion_text TEXT NOT NULL,
    trigger_context TEXT,
    document_sources UUID[],
    confidence_score FLOAT,
    displayed BOOLEAN DEFAULT FALSE,
    clicked BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for new tables
CREATE INDEX idx_source_conflicts_query_text ON source_conflicts USING gin(to_tsvector('english', query_text));
CREATE INDEX idx_source_conflicts_created_at ON source_conflicts(created_at);

CREATE INDEX idx_system_metrics_metric_name ON system_metrics(metric_name);
CREATE INDEX idx_system_metrics_recorded_at ON system_metrics(recorded_at);
CREATE INDEX idx_system_metrics_metric_type ON system_metrics(metric_type);

CREATE INDEX idx_response_feedback_query_log_id ON response_feedback(query_log_id);
CREATE INDEX idx_response_feedback_user_id ON response_feedback(user_id);
CREATE INDEX idx_response_feedback_rating ON response_feedback(rating);

CREATE INDEX idx_proactive_suggestions_conversation_id ON proactive_suggestions(conversation_id);
CREATE INDEX idx_proactive_suggestions_displayed ON proactive_suggestions(displayed);
CREATE INDEX idx_proactive_suggestions_created_at ON proactive_suggestions(created_at);

-- Create view for system dashboard
CREATE VIEW system_dashboard AS
SELECT
    (SELECT COUNT(*) FROM data_sources WHERE status = 'active') as active_sources,
    (SELECT COUNT(*) FROM documents WHERE status = 'completed') as processed_documents,
    (SELECT COUNT(*) FROM embeddings) as total_embeddings,
    (SELECT COUNT(*) FROM conversations WHERE created_at > NOW() - INTERVAL '24 hours') as daily_conversations,
    (SELECT AVG(response_time_ms) FROM query_logs WHERE created_at > NOW() - INTERVAL '24 hours') as avg_response_time,
    (SELECT AVG(rating) FROM response_feedback WHERE created_at > NOW() - INTERVAL '7 days') as avg_user_rating;

-- Create function to log system metrics
CREATE OR REPLACE FUNCTION log_system_metric(
    p_metric_name VARCHAR(100),
    p_metric_value FLOAT,
    p_metric_type VARCHAR(50),
    p_metadata JSONB DEFAULT '{}'::jsonb
)
RETURNS UUID
LANGUAGE plpgsql
AS $$
DECLARE
    metric_id UUID;
BEGIN
    INSERT INTO system_metrics (metric_name, metric_value, metric_type, metadata)
    VALUES (p_metric_name, p_metric_value, p_metric_type, p_metadata)
    RETURNING id INTO metric_id;
    
    RETURN metric_id;
END;
$$;

-- Create function to check for conflicting documents
CREATE OR REPLACE FUNCTION detect_conflicts(
    p_query_text TEXT,
    p_document_ids UUID[],
    p_confidence_scores FLOAT[],
    p_threshold FLOAT DEFAULT 0.3
)
RETURNS BOOLEAN
LANGUAGE plpgsql
AS $$
DECLARE
    max_score FLOAT;
    min_score FLOAT;
    conflict_detected BOOLEAN := FALSE;
BEGIN
    -- Check if there's significant variation in confidence scores
    max_score := (SELECT MAX(score) FROM unnest(p_confidence_scores) AS score);
    min_score := (SELECT MIN(score) FROM unnest(p_confidence_scores) AS score);
    
    IF (max_score - min_score) > p_threshold THEN
        conflict_detected := TRUE;
        
        -- Log the conflict
        INSERT INTO source_conflicts (query_text, document_ids, conflict_type, confidence_scores)
        VALUES (p_query_text, p_document_ids, 'confidence_variation', p_confidence_scores);
    END IF;
    
    RETURN conflict_detected;
END;
$$;

-- Create function to get relevant proactive suggestions
CREATE OR REPLACE FUNCTION get_proactive_suggestions(
    p_conversation_id UUID,
    p_context TEXT,
    p_limit INTEGER DEFAULT 3
)
RETURNS TABLE (
    id UUID,
    suggestion_text TEXT,
    confidence_score FLOAT,
    document_sources UUID[]
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        ps.id,
        ps.suggestion_text,
        ps.confidence_score,
        ps.document_sources
    FROM proactive_suggestions ps
    WHERE ps.conversation_id = p_conversation_id
        AND ps.displayed = FALSE
        AND ps.confidence_score > 0.7
    ORDER BY ps.confidence_score DESC, ps.created_at DESC
    LIMIT p_limit;
END;
$$;

-- Create Row Level Security (RLS) policies for data protection
ALTER TABLE conversations ENABLE ROW LEVEL SECURITY;
ALTER TABLE messages ENABLE ROW LEVEL SECURITY;
ALTER TABLE query_logs ENABLE ROW LEVEL SECURITY;
ALTER TABLE response_feedback ENABLE ROW LEVEL SECURITY;

-- Note: RLS policies would be configured based on specific authentication requirements
-- For now, we'll set up the structure for future implementation 