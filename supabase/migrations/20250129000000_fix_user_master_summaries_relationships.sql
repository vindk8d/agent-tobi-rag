-- Migration: Fix user_master_summaries foreign key relationships
-- Adds missing foreign key constraints for data integrity

-- 1. Add foreign key for user_id to users.user_id (text -> text)
-- Note: This maintains the existing TEXT-based user_id pattern used by the functions
ALTER TABLE user_master_summaries 
ADD CONSTRAINT user_master_summaries_user_id_fkey 
FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE;

-- 2. Add foreign key for last_conversation_id to conversations.id (uuid -> uuid)
ALTER TABLE user_master_summaries 
ADD CONSTRAINT user_master_summaries_last_conversation_id_fkey 
FOREIGN KEY (last_conversation_id) REFERENCES conversations(id) ON DELETE SET NULL;

-- 3. Add check constraint to ensure conversations_included elements are valid
-- Note: PostgreSQL doesn't support FK constraints on array elements directly,
-- but we can add a trigger-based validation function

-- Create function to validate conversations_included array
CREATE OR REPLACE FUNCTION validate_conversations_included() 
RETURNS TRIGGER AS $$
DECLARE
    conv_id UUID;
    invalid_ids UUID[] := '{}';
BEGIN
    -- Check each UUID in conversations_included array
    IF NEW.conversations_included IS NOT NULL THEN
        FOREACH conv_id IN ARRAY NEW.conversations_included
        LOOP
            IF NOT EXISTS (SELECT 1 FROM conversations WHERE id = conv_id) THEN
                invalid_ids := array_append(invalid_ids, conv_id);
            END IF;
        END LOOP;
        
        -- Raise error if invalid conversation IDs found
        IF array_length(invalid_ids, 1) > 0 THEN
            RAISE EXCEPTION 'Invalid conversation IDs in conversations_included: %', 
                array_to_string(invalid_ids, ', ');
        END IF;
    END IF;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- 4. Create trigger to validate conversations_included on INSERT/UPDATE
DROP TRIGGER IF EXISTS validate_conversations_included_trigger ON user_master_summaries;
CREATE TRIGGER validate_conversations_included_trigger
    BEFORE INSERT OR UPDATE ON user_master_summaries
    FOR EACH ROW
    EXECUTE FUNCTION validate_conversations_included();

-- 5. Add indexes for performance on the new foreign keys
CREATE INDEX IF NOT EXISTS idx_user_master_summaries_user_id 
    ON user_master_summaries(user_id);

CREATE INDEX IF NOT EXISTS idx_user_master_summaries_last_conversation_id 
    ON user_master_summaries(last_conversation_id);

-- 6. Add index for conversations_included array for performance
CREATE INDEX IF NOT EXISTS idx_user_master_summaries_conversations_included 
    ON user_master_summaries USING gin(conversations_included);

-- Log completion
DO $$ 
BEGIN
    RAISE NOTICE '✅ Added foreign key constraints to user_master_summaries';
    RAISE NOTICE '✅ user_id -> users.user_id (with CASCADE delete)';
    RAISE NOTICE '✅ last_conversation_id -> conversations.id (with SET NULL)';
    RAISE NOTICE '✅ Added validation trigger for conversations_included array';
    RAISE NOTICE '✅ Added performance indexes for foreign key columns';
    RAISE NOTICE '⚠️  Consider migrating to UUID-based user references in future';
END $$; 