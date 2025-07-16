-- Migration: Update messages table with user_id and new role options
-- This migration adds user_id column and updates role enum to support bot, human, HITL

-- Step 1: Add user_id column to messages table
ALTER TABLE messages ADD COLUMN IF NOT EXISTS user_id UUID;

-- Step 2: Update the role check constraint to support the new values
ALTER TABLE messages DROP CONSTRAINT IF EXISTS messages_role_check;
ALTER TABLE messages ADD CONSTRAINT messages_role_check 
    CHECK (role IN ('bot', 'human', 'HITL'));

-- Step 3: Add foreign key constraint for user_id
ALTER TABLE messages ADD CONSTRAINT messages_user_id_fkey 
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE;

-- Step 4: Create index for better performance on user_id lookups
CREATE INDEX IF NOT EXISTS idx_messages_user_id ON messages(user_id);

-- Step 5: Update existing messages to migrate old role values to new ones
-- This is a data migration step - update existing records
UPDATE messages SET role = 'bot' WHERE role = 'assistant';
UPDATE messages SET role = 'human' WHERE role = 'user';
UPDATE messages SET role = 'human' WHERE role = 'system'; -- system messages can be treated as human for now

-- Step 6: Add comments to document the new structure
COMMENT ON COLUMN messages.user_id IS 'Foreign key to users table - identifies which user is associated with this message';
COMMENT ON COLUMN messages.role IS 'Message role: bot (AI assistant), human (user), or HITL (human-in-the-loop)';

-- Step 7: Update RLS policies for messages table to include user_id
-- Drop existing policies first
DROP POLICY IF EXISTS "Allow users to read messages from their conversations" ON messages;
DROP POLICY IF EXISTS "Allow users to create messages in their conversations" ON messages;
DROP POLICY IF EXISTS "Allow users to update messages in their conversations" ON messages;
DROP POLICY IF EXISTS "Allow users to delete messages from their conversations" ON messages;

-- Create new policies that use user_id directly
CREATE POLICY "Users can read their own messages" ON messages 
    FOR SELECT USING (user_id = auth.uid()::uuid);

CREATE POLICY "Users can create their own messages" ON messages 
    FOR INSERT WITH CHECK (user_id = auth.uid()::uuid);

CREATE POLICY "Users can update their own messages" ON messages 
    FOR UPDATE USING (user_id = auth.uid()::uuid);

CREATE POLICY "Users can delete their own messages" ON messages 
    FOR DELETE USING (user_id = auth.uid()::uuid);

-- Step 8: Create a function to automatically set user_id for new messages
CREATE OR REPLACE FUNCTION set_message_user_id()
RETURNS TRIGGER AS $$
BEGIN
    -- If user_id is not provided, try to get it from the conversation
    IF NEW.user_id IS NULL THEN
        SELECT user_id INTO NEW.user_id 
        FROM conversations 
        WHERE id = NEW.conversation_id;
    END IF;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger to automatically set user_id
DROP TRIGGER IF EXISTS trigger_set_message_user_id ON messages;
CREATE TRIGGER trigger_set_message_user_id
    BEFORE INSERT ON messages
    FOR EACH ROW
    EXECUTE FUNCTION set_message_user_id();

-- Step 9: Update the schema documentation comment
COMMENT ON TABLE messages IS 'Stores individual messages within conversations. Each message has a role (bot, human, HITL) and is associated with a user.'; 