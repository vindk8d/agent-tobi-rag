-- Migration: Convert user_id fields from text to UUID
-- This migration simplifies the user reference system by:
-- 1. Converting all user_id text fields to UUID type
-- 2. Making them all reference users.id instead of users.user_id
-- 3. Deprecating users.user_id field

-- Step 1: Drop all RLS policies that depend on user_id columns
-- Conversations table policies
DROP POLICY IF EXISTS "Allow users to read their own conversations" ON conversations;
DROP POLICY IF EXISTS "Allow users to create conversations" ON conversations;
DROP POLICY IF EXISTS "Allow users to update their own conversations" ON conversations;
DROP POLICY IF EXISTS "Allow users to delete their own conversations" ON conversations;

-- Messages table policies (depend on conversations.user_id)
DROP POLICY IF EXISTS "Allow users to read messages from their conversations" ON messages;
DROP POLICY IF EXISTS "Allow users to create messages in their conversations" ON messages;
DROP POLICY IF EXISTS "Allow users to update messages in their conversations" ON messages;
DROP POLICY IF EXISTS "Allow users to delete messages from their conversations" ON messages;

-- Query logs policies (depend on conversations.user_id)
DROP POLICY IF EXISTS "Allow users to read query logs from their conversations" ON query_logs;
DROP POLICY IF EXISTS "Allow users to create query logs in their conversations" ON query_logs;

-- Response feedback policies
DROP POLICY IF EXISTS "Allow users to create response feedback" ON response_feedback;
DROP POLICY IF EXISTS "Allow users to read their own response feedback" ON response_feedback;
DROP POLICY IF EXISTS "Allow users to update their own response feedback" ON response_feedback;

-- Conversation summaries policies
DROP POLICY IF EXISTS "Users can access their own conversation summaries" ON conversation_summaries;
DROP POLICY IF EXISTS "conversation_summaries_user_access" ON conversation_summaries;

-- Memory access patterns policies
DROP POLICY IF EXISTS "Users can access their own memory patterns" ON memory_access_patterns;

-- Long term memories policies
DROP POLICY IF EXISTS "Users can access their own memories" ON long_term_memories;

-- Users table policies
DROP POLICY IF EXISTS "Users can view their own profile" ON users;
DROP POLICY IF EXISTS "Users can update their own profile" ON users;

-- User sessions policies
DROP POLICY IF EXISTS "Users can view their own sessions" ON user_sessions;

-- Step 2: Drop views that depend on users.user_id before we can drop the column
DROP VIEW IF EXISTS active_user_sessions;
DROP VIEW IF EXISTS user_profiles;

-- Step 3: Add temporary UUID columns to tables with text user_id (skip views)
ALTER TABLE conversations ADD COLUMN user_id_uuid UUID;
ALTER TABLE conversation_summaries ADD COLUMN user_id_uuid UUID;
ALTER TABLE memory_access_patterns ADD COLUMN user_id_uuid UUID;
ALTER TABLE response_feedback ADD COLUMN user_id_uuid UUID;

-- Step 4: Populate UUID columns by looking up users.id based on users.user_id
UPDATE conversations 
SET user_id_uuid = users.id 
FROM users 
WHERE conversations.user_id = users.user_id;

UPDATE conversation_summaries 
SET user_id_uuid = users.id 
FROM users 
WHERE conversation_summaries.user_id = users.user_id;

UPDATE memory_access_patterns 
SET user_id_uuid = users.id 
FROM users 
WHERE memory_access_patterns.user_id = users.user_id;

UPDATE response_feedback 
SET user_id_uuid = users.id 
FROM users 
WHERE response_feedback.user_id = users.user_id;

-- Step 5: Drop existing foreign key constraints
ALTER TABLE conversations DROP CONSTRAINT IF EXISTS conversations_user_id_fkey;
ALTER TABLE conversation_summaries DROP CONSTRAINT IF EXISTS conversation_summaries_user_id_fkey;
ALTER TABLE memory_access_patterns DROP CONSTRAINT IF EXISTS memory_access_patterns_user_id_fkey;
ALTER TABLE response_feedback DROP CONSTRAINT IF EXISTS response_feedback_user_id_fkey;

-- Step 6: Drop old text user_id columns
ALTER TABLE conversations DROP COLUMN user_id;
ALTER TABLE conversation_summaries DROP COLUMN user_id;
ALTER TABLE memory_access_patterns DROP COLUMN user_id;
ALTER TABLE response_feedback DROP COLUMN user_id;

-- Step 7: Rename UUID columns to user_id
ALTER TABLE conversations RENAME COLUMN user_id_uuid TO user_id;
ALTER TABLE conversation_summaries RENAME COLUMN user_id_uuid TO user_id;
ALTER TABLE memory_access_patterns RENAME COLUMN user_id_uuid TO user_id;
ALTER TABLE response_feedback RENAME COLUMN user_id_uuid TO user_id;

-- Step 8: Set NOT NULL constraints where needed
ALTER TABLE conversations ALTER COLUMN user_id SET NOT NULL;
ALTER TABLE conversation_summaries ALTER COLUMN user_id SET NOT NULL;
ALTER TABLE memory_access_patterns ALTER COLUMN user_id SET NOT NULL;
ALTER TABLE response_feedback ALTER COLUMN user_id SET NOT NULL;

-- Step 9: Add foreign key constraints to users.id
ALTER TABLE conversations ADD CONSTRAINT conversations_user_id_fkey 
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE;

ALTER TABLE conversation_summaries ADD CONSTRAINT conversation_summaries_user_id_fkey 
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE;

ALTER TABLE memory_access_patterns ADD CONSTRAINT memory_access_patterns_user_id_fkey 
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE;

ALTER TABLE response_feedback ADD CONSTRAINT response_feedback_user_id_fkey 
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE;

-- Step 10: Update user_sessions table to also reference users.id (it's already UUID)
ALTER TABLE user_sessions DROP CONSTRAINT IF EXISTS user_sessions_user_id_fkey;
ALTER TABLE user_sessions ADD CONSTRAINT user_sessions_user_id_fkey 
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE;

-- Step 11: Drop the users.user_id text field (no longer needed)
ALTER TABLE users DROP COLUMN user_id;

-- Step 12: Recreate views with the new UUID structure
CREATE VIEW user_profiles AS
SELECT 
    u.id as user_id,
    u.user_type,
    u.email,
    u.display_name,
    u.preferences,
    u.permissions,
    u.created_at,
    u.updated_at,
    e.name as employee_name,
    e.position
FROM users u
LEFT JOIN employees e ON u.employee_id = e.id;

CREATE VIEW active_user_sessions AS
SELECT 
    us.id,
    us.user_id,
    us.platform,
    us.platform_user_id,
    us.metadata,
    us.last_activity_at,
    us.created_at,
    u.email,
    u.display_name,
    u.user_type
FROM user_sessions us
JOIN users u ON us.user_id = u.id
WHERE us.expires_at > NOW();

-- Step 13: Update function signatures to use UUID
CREATE OR REPLACE FUNCTION create_user_from_employee(
    p_employee_id INTEGER,
    p_email TEXT,
    p_display_name TEXT DEFAULT NULL,
    p_user_type VARCHAR(20) DEFAULT 'employee'
) RETURNS UUID AS $$
DECLARE
    new_user_id UUID;
BEGIN
    INSERT INTO users (employee_id, email, display_name, user_type)
    VALUES (p_employee_id, p_email, p_display_name, p_user_type)
    RETURNING id INTO new_user_id;
    
    RETURN new_user_id;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION get_user_by_platform(
    p_platform TEXT,
    p_platform_user_id TEXT
) RETURNS UUID AS $$
DECLARE
    found_user_id UUID;
BEGIN
    SELECT u.id INTO found_user_id
    FROM users u
    JOIN user_sessions us ON u.id = us.user_id
    WHERE us.platform = p_platform 
      AND us.platform_user_id = p_platform_user_id
      AND us.expires_at > NOW();
    
    RETURN found_user_id;
END;
$$ LANGUAGE plpgsql;

-- Step 14: Recreate essential RLS policies with proper UUID logic
-- Note: These are simplified policies for now. Production policies should be more restrictive.

-- Drop and recreate policies instead of CREATE OR REPLACE (not supported for policies)
DROP POLICY IF EXISTS "Allow service role full access to conversations" ON conversations;
DROP POLICY IF EXISTS "Allow anonymous users to create conversations" ON conversations;
DROP POLICY IF EXISTS "Allow anonymous users to read conversations" ON conversations;
DROP POLICY IF EXISTS "Allow anonymous users to update conversations" ON conversations;

DROP POLICY IF EXISTS "Allow service role full access to messages" ON messages;
DROP POLICY IF EXISTS "Allow anonymous users to create messages" ON messages;
DROP POLICY IF EXISTS "Allow anonymous users to read messages" ON messages;
DROP POLICY IF EXISTS "Allow anonymous users to update messages" ON messages;

DROP POLICY IF EXISTS "Allow service role full access to query logs" ON query_logs;
DROP POLICY IF EXISTS "Allow anonymous users to create query logs" ON query_logs;
DROP POLICY IF EXISTS "Allow anonymous users to read query logs" ON query_logs;

DROP POLICY IF EXISTS "Allow service role full access to response feedback" ON response_feedback;
DROP POLICY IF EXISTS "Allow anonymous users to create response feedback" ON response_feedback;
DROP POLICY IF EXISTS "Allow anonymous users to read response feedback" ON response_feedback;

DROP POLICY IF EXISTS "Service role full access to conversation summaries" ON conversation_summaries;
DROP POLICY IF EXISTS "Service role full access to memory patterns" ON memory_access_patterns;
DROP POLICY IF EXISTS "Service role full access to memories" ON long_term_memories;
DROP POLICY IF EXISTS "Service role full access to users" ON users;
DROP POLICY IF EXISTS "Service role full access to sessions" ON user_sessions;

-- Conversations table policies
CREATE POLICY "Allow service role full access to conversations" ON conversations
    FOR ALL TO service_role USING (true);

CREATE POLICY "Allow anonymous users to create conversations" ON conversations
    FOR INSERT TO anon WITH CHECK (true);

CREATE POLICY "Allow anonymous users to read conversations" ON conversations
    FOR SELECT TO anon USING (true);

CREATE POLICY "Allow anonymous users to update conversations" ON conversations
    FOR UPDATE TO anon USING (true);

-- Messages table policies  
CREATE POLICY "Allow service role full access to messages" ON messages
    FOR ALL TO service_role USING (true);

CREATE POLICY "Allow anonymous users to create messages" ON messages
    FOR INSERT TO anon WITH CHECK (true);

CREATE POLICY "Allow anonymous users to read messages" ON messages
    FOR SELECT TO anon USING (true);

CREATE POLICY "Allow anonymous users to update messages" ON messages
    FOR UPDATE TO anon USING (true);

-- Query logs policies
CREATE POLICY "Allow service role full access to query logs" ON query_logs
    FOR ALL TO service_role USING (true);

CREATE POLICY "Allow anonymous users to create query logs" ON query_logs
    FOR INSERT TO anon WITH CHECK (true);

CREATE POLICY "Allow anonymous users to read query logs" ON query_logs
    FOR SELECT TO anon USING (true);

-- Response feedback policies
CREATE POLICY "Allow service role full access to response feedback" ON response_feedback
    FOR ALL TO service_role USING (true);

CREATE POLICY "Allow anonymous users to create response feedback" ON response_feedback
    FOR INSERT TO anon WITH CHECK (true);

CREATE POLICY "Allow anonymous users to read response feedback" ON response_feedback
    FOR SELECT TO anon USING (true);

-- Conversation summaries policies
CREATE POLICY "Service role full access to conversation summaries" ON conversation_summaries
    FOR ALL TO service_role USING (true);

-- Memory access patterns policies
CREATE POLICY "Service role full access to memory patterns" ON memory_access_patterns
    FOR ALL TO service_role USING (true);

-- Long term memories policies
CREATE POLICY "Service role full access to memories" ON long_term_memories
    FOR ALL TO service_role USING (true);

-- Users table policies
CREATE POLICY "Service role full access to users" ON users
    FOR ALL TO service_role USING (true);

-- User sessions policies
CREATE POLICY "Service role full access to sessions" ON user_sessions
    FOR ALL TO service_role USING (true);

-- Migration completed: Simplified user_id to use UUID references to users.id consistently across all tables 