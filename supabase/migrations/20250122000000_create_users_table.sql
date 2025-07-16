-- Migration: Create Users Table and Establish User-Employee Relationship
-- This migration creates a proper user management system that separates business entities (employees)
-- from system users, enabling proper user identification in conversations and messages.
-- Uses user_id consistently throughout the system instead of introducing user_uuid.

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- 1. Create users table for system user management
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id TEXT UNIQUE NOT NULL,                    -- System user identifier (email, username, etc.)
    user_type VARCHAR(20) NOT NULL DEFAULT 'employee' CHECK (user_type IN ('employee', 'customer', 'admin', 'system')),
    employee_id UUID REFERENCES employees(id) ON DELETE SET NULL,  -- Link to employees table
    customer_id UUID,                                -- Future: link to customers table
    email TEXT UNIQUE NOT NULL,
    display_name TEXT NOT NULL,
    avatar_url TEXT,
    is_active BOOLEAN DEFAULT TRUE,
    is_verified BOOLEAN DEFAULT FALSE,
    last_login_at TIMESTAMP WITH TIME ZONE,
    preferences JSONB DEFAULT '{}'::jsonb,          -- User preferences (notifications, settings, etc.)
    permissions JSONB DEFAULT '{}'::jsonb,          -- User permissions and roles
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'::jsonb
);

-- 2. Create user_sessions table for session management
CREATE TABLE IF NOT EXISTS user_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    session_token TEXT UNIQUE NOT NULL,
    platform VARCHAR(50) NOT NULL,                 -- 'telegram', 'web', 'api', etc.
    platform_user_id TEXT,                         -- Platform-specific user ID (telegram user ID, etc.)
    ip_address INET,
    user_agent TEXT,
    is_active BOOLEAN DEFAULT TRUE,
    expires_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_activity_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'::jsonb
);

-- 3. Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_users_user_id ON users(user_id);
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_users_employee_id ON users(employee_id);
CREATE INDEX IF NOT EXISTS idx_users_user_type ON users(user_type);
CREATE INDEX IF NOT EXISTS idx_users_is_active ON users(is_active);
CREATE INDEX IF NOT EXISTS idx_users_created_at ON users(created_at);

CREATE INDEX IF NOT EXISTS idx_user_sessions_user_id ON user_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_user_sessions_session_token ON user_sessions(session_token);
CREATE INDEX IF NOT EXISTS idx_user_sessions_platform ON user_sessions(platform);
CREATE INDEX IF NOT EXISTS idx_user_sessions_platform_user_id ON user_sessions(platform_user_id);
CREATE INDEX IF NOT EXISTS idx_user_sessions_is_active ON user_sessions(is_active);
CREATE INDEX IF NOT EXISTS idx_user_sessions_expires_at ON user_sessions(expires_at);

-- 4. Create function to automatically create users from existing user_id references
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

-- 6. Add foreign key constraints to existing tables (keeping existing user_id columns)
-- Note: We're adding constraints to existing user_id columns, not creating new ones

-- Add foreign key to conversations.user_id
ALTER TABLE conversations 
ADD CONSTRAINT fk_conversations_user_id 
FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE;

-- Add foreign key to conversation_summaries.user_id
ALTER TABLE conversation_summaries 
ADD CONSTRAINT fk_conversation_summaries_user_id 
FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE;

-- Add foreign key to memory_access_patterns.user_id
ALTER TABLE memory_access_patterns 
ADD CONSTRAINT fk_memory_access_patterns_user_id 
FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE;

-- Add foreign key to response_feedback.user_id
ALTER TABLE response_feedback 
ADD CONSTRAINT fk_response_feedback_user_id 
FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE;

-- 7. Create helpful views for common queries
CREATE OR REPLACE VIEW user_profiles AS
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

CREATE OR REPLACE VIEW active_user_sessions AS
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

-- 8. Create management functions
CREATE OR REPLACE FUNCTION create_user_from_employee(emp_id UUID) RETURNS UUID AS $$
DECLARE
    employee_data RECORD;
    new_user_id UUID;
BEGIN
    -- Get employee data
    SELECT id, name, email INTO employee_data FROM employees WHERE id = emp_id;
    
    IF employee_data.id IS NULL THEN
        RAISE EXCEPTION 'Employee not found with id: %', emp_id;
    END IF;
    
    -- Create user
    INSERT INTO users (user_id, employee_id, email, display_name, user_type)
    VALUES (
        employee_data.email,
        employee_data.id,
        employee_data.email,
        employee_data.name,
        'employee'
    )
    RETURNING id INTO new_user_id;
    
    RETURN new_user_id;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION get_user_by_platform(platform_name TEXT, platform_user_id TEXT) RETURNS TABLE(
    user_uuid UUID,
    user_id TEXT,
    display_name TEXT,
    user_type TEXT,
    is_active BOOLEAN
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        up.user_uuid,
        up.user_id,
        up.display_name,
        up.user_type,
        up.is_active
    FROM user_profiles up
    JOIN user_sessions us ON up.user_uuid = us.user_id
    WHERE us.platform = platform_name 
    AND us.platform_user_id = platform_user_id
    AND us.is_active = true
    AND up.is_active = true;
END;
$$ LANGUAGE plpgsql;

-- 9. Set up Row Level Security (RLS)
ALTER TABLE users ENABLE ROW LEVEL SECURITY;
ALTER TABLE user_sessions ENABLE ROW LEVEL SECURITY;

-- Basic RLS policies - users can only see their own data
CREATE POLICY user_own_data ON users 
    FOR ALL USING (user_id = current_setting('app.current_user_id', true));

CREATE POLICY user_own_sessions ON user_sessions 
    FOR ALL USING (
        user_id = (
            SELECT id FROM users 
            WHERE user_id = current_setting('app.current_user_id', true)
        )
    );

-- 10. Create update triggers for updated_at
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_users_updated_at 
    BEFORE UPDATE ON users 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- 11. Add helpful comments
COMMENT ON TABLE users IS 'System users table that maps to business entities (employees, customers)';
COMMENT ON TABLE user_sessions IS 'Active user sessions across different platforms (Telegram, Web, API)';
COMMENT ON COLUMN users.user_id IS 'System user identifier (email, username, or platform-specific ID)';
COMMENT ON COLUMN users.user_type IS 'Type of user: employee, customer, admin, or system';
COMMENT ON COLUMN users.employee_id IS 'Foreign key to employees table for employee users';
COMMENT ON COLUMN user_sessions.platform IS 'Platform where session is active: telegram, web, api, etc.';
COMMENT ON COLUMN user_sessions.platform_user_id IS 'Platform-specific user identifier (e.g., Telegram user ID)'; 