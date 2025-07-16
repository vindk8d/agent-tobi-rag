-- Migration: Fix User ID Consistency
-- This migration removes redundant user_uuid columns and implements consistent user_id usage
-- throughout the system, making it simpler and more aligned with existing patterns.

-- 1. First, drop RLS policies that depend on user_uuid columns
DROP POLICY IF EXISTS "Users can access their own conversation summaries" ON conversation_summaries;
DROP POLICY IF EXISTS "Users can access their own memory patterns" ON memory_access_patterns;
DROP POLICY IF EXISTS "Users can access their own memories" ON long_term_memories;

-- 2. Remove user_uuid columns that create confusion
ALTER TABLE conversations DROP COLUMN IF EXISTS user_uuid;
ALTER TABLE long_term_memories DROP COLUMN IF EXISTS user_uuid;
ALTER TABLE conversation_summaries DROP COLUMN IF EXISTS user_uuid;
ALTER TABLE memory_access_patterns DROP COLUMN IF EXISTS user_uuid;

-- 3. Drop the existing foreign key constraints from user_uuid columns if they exist
ALTER TABLE conversations DROP CONSTRAINT IF EXISTS conversations_user_uuid_fkey;
ALTER TABLE long_term_memories DROP CONSTRAINT IF EXISTS long_term_memories_user_uuid_fkey;
ALTER TABLE conversation_summaries DROP CONSTRAINT IF EXISTS conversation_summaries_user_uuid_fkey;
ALTER TABLE memory_access_patterns DROP CONSTRAINT IF EXISTS memory_access_patterns_user_uuid_fkey;

-- 4. Create function to ensure users exist for all user_id references
CREATE OR REPLACE FUNCTION ensure_user_exists(input_user_id TEXT) RETURNS UUID AS $$
DECLARE
    found_user_id UUID;
    employee_data RECORD;
BEGIN
    -- Check if user already exists
    SELECT id INTO found_user_id FROM users WHERE user_id = input_user_id;
    
    IF found_user_id IS NOT NULL THEN
        RETURN found_user_id;
    END IF;
    
    -- Try to find matching employee by email
    SELECT id, name, email INTO employee_data 
    FROM employees 
    WHERE email = input_user_id OR email = input_user_id || '@system.local';
    
    -- Create user with employee link if found
    IF employee_data.id IS NOT NULL THEN
        INSERT INTO users (user_id, employee_id, email, display_name, user_type)
        VALUES (
            input_user_id,
            employee_data.id,
            COALESCE(employee_data.email, input_user_id),
            employee_data.name,
            'employee'
        )
        RETURNING id INTO found_user_id;
    ELSE
        -- Create generic user
        INSERT INTO users (user_id, email, display_name, user_type)
        VALUES (
            input_user_id,
            CASE 
                WHEN input_user_id LIKE '%@%' THEN input_user_id 
                ELSE input_user_id || '@system.local' 
            END,
            input_user_id,
            'employee'
        )
        RETURNING id INTO found_user_id;
    END IF;
    
    RETURN found_user_id;
END;
$$ LANGUAGE plpgsql;

-- 5. Create users for all existing user_id references
DO $$
DECLARE
    user_id_record RECORD;
BEGIN
    -- Create users for all existing conversations
    FOR user_id_record IN 
        SELECT DISTINCT user_id 
        FROM conversations 
        WHERE user_id IS NOT NULL
    LOOP
        PERFORM ensure_user_exists(user_id_record.user_id);
    END LOOP;
    
    -- Create users for all existing conversation_summaries  
    FOR user_id_record IN 
        SELECT DISTINCT user_id 
        FROM conversation_summaries 
        WHERE user_id IS NOT NULL
    LOOP
        PERFORM ensure_user_exists(user_id_record.user_id);
    END LOOP;
    
    -- Create users for all existing memory_access_patterns
    FOR user_id_record IN 
        SELECT DISTINCT user_id 
        FROM memory_access_patterns 
        WHERE user_id IS NOT NULL
    LOOP
        PERFORM ensure_user_exists(user_id_record.user_id);
    END LOOP;
    
    -- Create users for all existing response_feedback
    FOR user_id_record IN 
        SELECT DISTINCT user_id 
        FROM response_feedback 
        WHERE user_id IS NOT NULL
    LOOP
        PERFORM ensure_user_exists(user_id_record.user_id);
    END LOOP;
    
    -- Create users for long_term_memories namespace patterns
    FOR user_id_record IN 
        SELECT DISTINCT namespace[2] as user_id
        FROM long_term_memories 
        WHERE array_length(namespace, 1) >= 2 
        AND namespace[1] = 'user'
        AND namespace[2] IS NOT NULL
    LOOP
        PERFORM ensure_user_exists(user_id_record.user_id);
    END LOOP;
END $$;

-- 6. Add foreign key constraints to existing user_id columns
-- Note: Adding constraints to existing columns, not creating new ones

-- Add foreign key to conversations.user_id (drop first if exists)
DO $$
BEGIN
    ALTER TABLE conversations DROP CONSTRAINT IF EXISTS fk_conversations_user_id;
    ALTER TABLE conversations ADD CONSTRAINT fk_conversations_user_id 
        FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE;
EXCEPTION
    WHEN duplicate_object THEN
        NULL; -- Ignore if constraint already exists
END $$;

-- Add foreign key to conversation_summaries.user_id
DO $$
BEGIN
    ALTER TABLE conversation_summaries DROP CONSTRAINT IF EXISTS fk_conversation_summaries_user_id;
    ALTER TABLE conversation_summaries ADD CONSTRAINT fk_conversation_summaries_user_id 
        FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE;
EXCEPTION
    WHEN duplicate_object THEN
        NULL; -- Ignore if constraint already exists
END $$;

-- Add foreign key to memory_access_patterns.user_id
DO $$
BEGIN
    ALTER TABLE memory_access_patterns DROP CONSTRAINT IF EXISTS fk_memory_access_patterns_user_id;
    ALTER TABLE memory_access_patterns ADD CONSTRAINT fk_memory_access_patterns_user_id 
        FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE;
EXCEPTION
    WHEN duplicate_object THEN
        NULL; -- Ignore if constraint already exists
END $$;

-- Add foreign key to response_feedback.user_id
DO $$
BEGIN
    ALTER TABLE response_feedback DROP CONSTRAINT IF EXISTS fk_response_feedback_user_id;
    ALTER TABLE response_feedback ADD CONSTRAINT fk_response_feedback_user_id 
        FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE;
EXCEPTION
    WHEN duplicate_object THEN
        NULL; -- Ignore if constraint already exists
END $$;

-- 7. Update the user_profiles view to be more useful
DROP VIEW IF EXISTS user_profiles;
CREATE VIEW user_profiles AS
SELECT 
    u.id as user_uuid,
    u.user_id,
    u.email,
    u.display_name,
    u.user_type,
    u.is_active,
    u.is_verified,
    u.last_login_at,
    u.created_at,
    e.name as employee_name,
    e.position as employee_position,
    e.phone as employee_phone,
    c.name as customer_name,
    c.company as customer_company
FROM users u
LEFT JOIN employees e ON u.employee_id = e.id
LEFT JOIN customers c ON u.customer_id = c.id;

-- 8. Update the active_user_sessions view
DROP VIEW IF EXISTS active_user_sessions;
CREATE VIEW active_user_sessions AS
SELECT 
    us.id as session_id,
    us.session_token,
    us.platform,
    us.platform_user_id,
    us.is_active,
    us.expires_at,
    us.last_activity_at,
    up.user_id,
    up.display_name,
    up.email,
    up.user_type,
    up.employee_name
FROM user_sessions us
JOIN user_profiles up ON us.user_id = up.user_uuid
WHERE us.is_active = true 
AND (us.expires_at IS NULL OR us.expires_at > NOW());

-- 9. Create new RLS policies using user_id consistently
CREATE POLICY "Users can access their own conversation summaries"
ON conversation_summaries FOR ALL
USING (user_id = current_setting('app.current_user_id', true));

CREATE POLICY "Users can access their own memory patterns"
ON memory_access_patterns FOR ALL
USING (user_id = current_setting('app.current_user_id', true));

CREATE POLICY "Users can access their own memories"
ON long_term_memories FOR ALL
USING (
    array_length(namespace, 1) >= 2 
    AND namespace[1] = 'user'
    AND namespace[2] = current_setting('app.current_user_id', true)
);

-- 10. Create indexes for the new foreign key constraints
CREATE INDEX IF NOT EXISTS idx_conversations_user_id ON conversations(user_id);
CREATE INDEX IF NOT EXISTS idx_conversation_summaries_user_id ON conversation_summaries(user_id);
CREATE INDEX IF NOT EXISTS idx_memory_access_patterns_user_id ON memory_access_patterns(user_id);
CREATE INDEX IF NOT EXISTS idx_response_feedback_user_id ON response_feedback(user_id);

-- 11. Log completion
DO $$ 
BEGIN
    RAISE NOTICE '✅ Fixed user ID consistency - removed user_uuid columns';
    RAISE NOTICE '✅ Added foreign key constraints to existing user_id columns';
    RAISE NOTICE '✅ Created users for all existing user_id references';
    RAISE NOTICE '✅ Updated views and policies to use user_id consistently';
    RAISE NOTICE '✅ User management system now uses user_id consistently throughout';
END $$; 