-- Migration: Create Users Table and Establish User-Employee Relationship
-- This migration creates a proper user management system that separates business entities (employees)
-- from system users, enabling proper user identification in conversations and messages.
-- Uses user_id consistently throughout the system instead of introducing user_uuid.

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- 1. Update existing users table with additional columns (base schema already exists)
ALTER TABLE users 
ADD COLUMN IF NOT EXISTS user_id TEXT UNIQUE,                    -- System user identifier (email, username, etc.)
ADD COLUMN IF NOT EXISTS employee_id UUID,  -- Link to employees table (will add FK later when employees exists)
ADD COLUMN IF NOT EXISTS customer_id UUID,                      -- Future: link to customers table
ADD COLUMN IF NOT EXISTS avatar_url TEXT,
ADD COLUMN IF NOT EXISTS is_verified BOOLEAN DEFAULT FALSE,
ADD COLUMN IF NOT EXISTS last_login_at TIMESTAMP WITH TIME ZONE,
ADD COLUMN IF NOT EXISTS preferences JSONB DEFAULT '{}'::jsonb,  -- User preferences (notifications, settings, etc.)
ADD COLUMN IF NOT EXISTS permissions JSONB DEFAULT '{}'::jsonb;  -- User permissions and roles

-- Update user_type to have proper constraints
ALTER TABLE users 
DROP CONSTRAINT IF EXISTS users_user_type_check,
ADD CONSTRAINT users_user_type_check CHECK (user_type IN ('employee', 'customer', 'admin', 'system'));

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

-- 3. Create indexes for performance (only create if columns exist)
CREATE INDEX IF NOT EXISTS idx_users_user_id ON users(user_id) WHERE user_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_users_employee_id ON users(employee_id) WHERE employee_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_users_user_type ON users(user_type);
CREATE INDEX IF NOT EXISTS idx_users_is_active ON users(is_active);
CREATE INDEX IF NOT EXISTS idx_users_created_at ON users(created_at);

-- 4. Create indexes for user_sessions
CREATE INDEX IF NOT EXISTS idx_user_sessions_user_id ON user_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_user_sessions_session_token ON user_sessions(session_token);
CREATE INDEX IF NOT EXISTS idx_user_sessions_platform ON user_sessions(platform);
CREATE INDEX IF NOT EXISTS idx_user_sessions_platform_user_id ON user_sessions(platform, platform_user_id);
CREATE INDEX IF NOT EXISTS idx_user_sessions_is_active ON user_sessions(is_active);
CREATE INDEX IF NOT EXISTS idx_user_sessions_expires_at ON user_sessions(expires_at);

-- 5. Create updated_at trigger for user_sessions
CREATE OR REPLACE FUNCTION update_user_sessions_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.last_activity_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS update_user_sessions_updated_at ON user_sessions;
CREATE TRIGGER update_user_sessions_updated_at
    BEFORE UPDATE ON user_sessions
    FOR EACH ROW
    EXECUTE FUNCTION update_user_sessions_updated_at(); 