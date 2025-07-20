-- Fix Summarization Counter Logic
-- Migration: 20250720000000_fix_summarization_counter.sql
-- Addresses issue where summarization happens at every step after threshold is met
-- because the counter isn't being properly reset

-- Add last_summarized_at column to conversations table
ALTER TABLE conversations 
ADD COLUMN IF NOT EXISTS last_summarized_at TIMESTAMP WITH TIME ZONE;

-- Create index for efficient querying
CREATE INDEX IF NOT EXISTS idx_conversations_last_summarized_at ON conversations(last_summarized_at);

-- Add comment to document the column
COMMENT ON COLUMN conversations.last_summarized_at IS 'Timestamp of last conversation summarization - used to count only new messages since last summary';

-- Log completion
DO $$ 
BEGIN
    RAISE NOTICE '✅ Summarization counter fix applied!';
    RAISE NOTICE '✅ Added last_summarized_at column to conversations table';
    RAISE NOTICE '✅ This will prevent repeated summarization after threshold is met';
END
$$; 