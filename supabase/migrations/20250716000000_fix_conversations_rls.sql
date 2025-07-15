-- Migration: Fix RLS policies for conversations table
-- This migration adds proper RLS policies for the conversations table to allow
-- authenticated users and service role to perform CRUD operations

-- Drop existing policies if they exist to avoid conflicts
DROP POLICY IF EXISTS "Allow authenticated users to manage conversations" ON conversations;
DROP POLICY IF EXISTS "Allow service role full access to conversations" ON conversations;
DROP POLICY IF EXISTS "Allow anonymous users to create conversations" ON conversations;
DROP POLICY IF EXISTS "Allow users to read their own conversations" ON conversations;
DROP POLICY IF EXISTS "Allow users to update their own conversations" ON conversations;
DROP POLICY IF EXISTS "Allow users to delete their own conversations" ON conversations;

-- Create policies for authenticated users
CREATE POLICY "Allow users to read their own conversations"
ON conversations FOR SELECT
TO authenticated
USING (user_id = auth.uid()::text);

CREATE POLICY "Allow users to create conversations"
ON conversations FOR INSERT
TO authenticated
WITH CHECK (user_id = auth.uid()::text);

CREATE POLICY "Allow users to update their own conversations"
ON conversations FOR UPDATE
TO authenticated
USING (user_id = auth.uid()::text)
WITH CHECK (user_id = auth.uid()::text);

CREATE POLICY "Allow users to delete their own conversations"
ON conversations FOR DELETE
TO authenticated
USING (user_id = auth.uid()::text);

-- Create policies for service role (for backend operations)
CREATE POLICY "Allow service role full access to conversations"
ON conversations FOR ALL
TO service_role
USING (true)
WITH CHECK (true);

-- Create policies for anonymous users (for demo/testing)
CREATE POLICY "Allow anonymous users to create conversations"
ON conversations FOR INSERT
TO anon
WITH CHECK (true);

CREATE POLICY "Allow anonymous users to read conversations"
ON conversations FOR SELECT
TO anon
USING (true);

CREATE POLICY "Allow anonymous users to update conversations"
ON conversations FOR UPDATE
TO anon
USING (true)
WITH CHECK (true);

-- Apply similar policies to messages table
DROP POLICY IF EXISTS "Allow authenticated users to manage messages" ON messages;
DROP POLICY IF EXISTS "Allow service role full access to messages" ON messages;
DROP POLICY IF EXISTS "Allow anonymous users to create messages" ON messages;
DROP POLICY IF EXISTS "Allow users to read messages from their conversations" ON messages;
DROP POLICY IF EXISTS "Allow users to create messages in their conversations" ON messages;
DROP POLICY IF EXISTS "Allow users to update messages in their conversations" ON messages;
DROP POLICY IF EXISTS "Allow users to delete messages from their conversations" ON messages;

-- Messages policies for authenticated users
CREATE POLICY "Allow users to read messages from their conversations"
ON messages FOR SELECT
TO authenticated
USING (
  EXISTS (
    SELECT 1 FROM conversations 
    WHERE conversations.id = messages.conversation_id 
    AND conversations.user_id = auth.uid()::text
  )
);

CREATE POLICY "Allow users to create messages in their conversations"
ON messages FOR INSERT
TO authenticated
WITH CHECK (
  EXISTS (
    SELECT 1 FROM conversations 
    WHERE conversations.id = messages.conversation_id 
    AND conversations.user_id = auth.uid()::text
  )
);

CREATE POLICY "Allow users to update messages in their conversations"
ON messages FOR UPDATE
TO authenticated
USING (
  EXISTS (
    SELECT 1 FROM conversations 
    WHERE conversations.id = messages.conversation_id 
    AND conversations.user_id = auth.uid()::text
  )
)
WITH CHECK (
  EXISTS (
    SELECT 1 FROM conversations 
    WHERE conversations.id = messages.conversation_id 
    AND conversations.user_id = auth.uid()::text
  )
);

CREATE POLICY "Allow users to delete messages from their conversations"
ON messages FOR DELETE
TO authenticated
USING (
  EXISTS (
    SELECT 1 FROM conversations 
    WHERE conversations.id = messages.conversation_id 
    AND conversations.user_id = auth.uid()::text
  )
);

-- Messages policies for service role
CREATE POLICY "Allow service role full access to messages"
ON messages FOR ALL
TO service_role
USING (true)
WITH CHECK (true);

-- Messages policies for anonymous users
CREATE POLICY "Allow anonymous users to create messages"
ON messages FOR INSERT
TO anon
WITH CHECK (true);

CREATE POLICY "Allow anonymous users to read messages"
ON messages FOR SELECT
TO anon
USING (true);

CREATE POLICY "Allow anonymous users to update messages"
ON messages FOR UPDATE
TO anon
USING (true)
WITH CHECK (true);

-- Apply similar policies to query_logs table
DROP POLICY IF EXISTS "Allow authenticated users to manage query logs" ON query_logs;
DROP POLICY IF EXISTS "Allow service role full access to query logs" ON query_logs;
DROP POLICY IF EXISTS "Allow anonymous users to create query logs" ON query_logs;

-- Query logs policies for authenticated users
CREATE POLICY "Allow users to read query logs from their conversations"
ON query_logs FOR SELECT
TO authenticated
USING (
  EXISTS (
    SELECT 1 FROM conversations 
    WHERE conversations.id = query_logs.conversation_id 
    AND conversations.user_id = auth.uid()::text
  )
);

CREATE POLICY "Allow users to create query logs in their conversations"
ON query_logs FOR INSERT
TO authenticated
WITH CHECK (
  EXISTS (
    SELECT 1 FROM conversations 
    WHERE conversations.id = query_logs.conversation_id 
    AND conversations.user_id = auth.uid()::text
  )
);

-- Query logs policies for service role
CREATE POLICY "Allow service role full access to query logs"
ON query_logs FOR ALL
TO service_role
USING (true)
WITH CHECK (true);

-- Query logs policies for anonymous users
CREATE POLICY "Allow anonymous users to create query logs"
ON query_logs FOR INSERT
TO anon
WITH CHECK (true);

CREATE POLICY "Allow anonymous users to read query logs"
ON query_logs FOR SELECT
TO anon
USING (true);

-- Apply similar policies to response_feedback table
DROP POLICY IF EXISTS "Allow authenticated users to manage response feedback" ON response_feedback;
DROP POLICY IF EXISTS "Allow service role full access to response feedback" ON response_feedback;
DROP POLICY IF EXISTS "Allow anonymous users to create response feedback" ON response_feedback;

-- Response feedback policies for authenticated users
CREATE POLICY "Allow users to read their own response feedback"
ON response_feedback FOR SELECT
TO authenticated
USING (user_id = auth.uid()::text);

CREATE POLICY "Allow users to create response feedback"
ON response_feedback FOR INSERT
TO authenticated
WITH CHECK (user_id = auth.uid()::text);

CREATE POLICY "Allow users to update their own response feedback"
ON response_feedback FOR UPDATE
TO authenticated
USING (user_id = auth.uid()::text)
WITH CHECK (user_id = auth.uid()::text);

-- Response feedback policies for service role
CREATE POLICY "Allow service role full access to response feedback"
ON response_feedback FOR ALL
TO service_role
USING (true)
WITH CHECK (true);

-- Response feedback policies for anonymous users
CREATE POLICY "Allow anonymous users to create response feedback"
ON response_feedback FOR INSERT
TO anon
WITH CHECK (true);

CREATE POLICY "Allow anonymous users to read response feedback"
ON response_feedback FOR SELECT
TO anon
USING (true);

-- Log completion
DO $$ 
BEGIN
    RAISE NOTICE '✅ RLS policies created for conversations and related tables';
    RAISE NOTICE '✅ Service role has full access for backend operations';
    RAISE NOTICE '✅ Anonymous users can create/read for demo purposes';
    RAISE NOTICE '✅ Authenticated users can only access their own data';
END
$$; 