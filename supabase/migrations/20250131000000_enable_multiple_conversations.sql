-- Migration: Enable Multiple Conversations Per User
-- Remove single conversation constraint and add optimizations

-- Add conversation status for better management
ALTER TABLE conversations 
ADD COLUMN IF NOT EXISTS status TEXT DEFAULT 'active' CHECK (status IN ('active', 'archived', 'deleted'));

-- Add index for active conversations per user (partial index for performance)
CREATE INDEX IF NOT EXISTS idx_conversations_user_status_updated 
ON conversations (user_id, status, updated_at DESC) 
WHERE status = 'active';

-- Add index for conversation title search (full-text search)
CREATE INDEX IF NOT EXISTS idx_conversations_title_search 
ON conversations USING gin (to_tsvector('english', title));

-- Update message count trigger function
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

-- Create trigger if it doesn't exist
DROP TRIGGER IF EXISTS trigger_update_message_count ON messages;
CREATE TRIGGER trigger_update_message_count
    AFTER INSERT OR DELETE ON messages
    FOR EACH ROW EXECUTE FUNCTION update_conversation_message_count();

-- Update existing conversations to have active status
UPDATE conversations SET status = 'active' WHERE status IS NULL;

-- Add comment for documentation
COMMENT ON COLUMN conversations.status IS 'Conversation status: active, archived, or deleted';
COMMENT ON INDEX idx_conversations_user_status_updated IS 'Optimized index for fetching active conversations per user';
COMMENT ON INDEX idx_conversations_title_search IS 'Full-text search index for conversation titles';
COMMENT ON FUNCTION update_conversation_message_count() IS 'Automatically maintains message_count and updated_at for conversations';
