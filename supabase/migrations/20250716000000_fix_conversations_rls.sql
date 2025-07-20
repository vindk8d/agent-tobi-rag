-- Migration: Fix RLS policies for conversations table
-- This migration adds proper RLS policies for the conversations table to allow
-- authenticated users and service role to perform CRUD operations

-- Drop existing policies if they exist to avoid conflicts
DROP POLICY IF EXISTS "Allow authenticated users to manage conversations" ON conversations;
DROP POLICY IF EXISTS "Allow service role full access to conversations" ON conversations;
DROP POLICY IF EXISTS "Allow anonymous users to create conversations" ON conversations;
DROP POLICY IF EXISTS "Allow anonymous users to read conversations" ON conversations;
DROP POLICY IF EXISTS "Allow anonymous users to update conversations" ON conversations;
DROP POLICY IF EXISTS "Allow users to read their own conversations" ON conversations;
DROP POLICY IF EXISTS "Allow users to create conversations" ON conversations;
DROP POLICY IF EXISTS "Allow users to update their own conversations" ON conversations;
DROP POLICY IF EXISTS "Allow users to delete their own conversations" ON conversations;

-- Create policies for authenticated users (conversations.user_id is UUID, so compare with auth.uid())
CREATE POLICY "Allow users to read their own conversations"
ON conversations FOR SELECT
TO authenticated
USING (user_id = auth.uid());

CREATE POLICY "Allow users to create conversations"
ON conversations FOR INSERT
TO authenticated
WITH CHECK (user_id = auth.uid());

CREATE POLICY "Allow users to update their own conversations"
ON conversations FOR UPDATE
TO authenticated
USING (user_id = auth.uid())
WITH CHECK (user_id = auth.uid());

CREATE POLICY "Allow users to delete their own conversations"
ON conversations FOR DELETE
TO authenticated
USING (user_id = auth.uid());

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
DROP POLICY IF EXISTS "Allow anonymous users to read messages" ON messages;
DROP POLICY IF EXISTS "Allow anonymous users to update messages" ON messages;
DROP POLICY IF EXISTS "Allow users to read messages from their conversations" ON messages;
DROP POLICY IF EXISTS "Allow users to create messages in their conversations" ON messages;
DROP POLICY IF EXISTS "Allow users to update messages in their conversations" ON messages;
DROP POLICY IF EXISTS "Allow users to delete messages from their conversations" ON messages;

-- Messages policies for authenticated users (fix UUID comparison)
CREATE POLICY "Allow users to read messages from their conversations"
ON messages FOR SELECT
TO authenticated
USING (
  EXISTS (
    SELECT 1 FROM conversations 
    WHERE conversations.id = messages.conversation_id 
    AND conversations.user_id = auth.uid()
  )
);

CREATE POLICY "Allow users to create messages in their conversations"
ON messages FOR INSERT
TO authenticated
WITH CHECK (
  EXISTS (
    SELECT 1 FROM conversations 
    WHERE conversations.id = messages.conversation_id 
    AND conversations.user_id = auth.uid()
  )
);

CREATE POLICY "Allow users to update messages in their conversations"
ON messages FOR UPDATE
TO authenticated
USING (
  EXISTS (
    SELECT 1 FROM conversations 
    WHERE conversations.id = messages.conversation_id 
    AND conversations.user_id = auth.uid()
  )
)
WITH CHECK (
  EXISTS (
    SELECT 1 FROM conversations 
    WHERE conversations.id = messages.conversation_id 
    AND conversations.user_id = auth.uid()
  )
);

CREATE POLICY "Allow users to delete messages from their conversations"
ON messages FOR DELETE
TO authenticated
USING (
  EXISTS (
    SELECT 1 FROM conversations 
    WHERE conversations.id = messages.conversation_id 
    AND conversations.user_id = auth.uid()
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

-- Skip query_logs policies - table doesn't exist in current schema

-- Skip response_feedback policies - table doesn't exist in current schema

-- Log completion
DO $$ 
BEGIN
    RAISE NOTICE '✅ RLS policies created for conversations and related tables';
    RAISE NOTICE '✅ Service role has full access for backend operations';
    RAISE NOTICE '✅ Anonymous users can create/read for demo purposes';
    RAISE NOTICE '✅ Authenticated users can only access their own data';
END
$$; 