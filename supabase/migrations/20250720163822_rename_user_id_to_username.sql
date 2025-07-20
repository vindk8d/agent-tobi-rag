-- Rename user_id column to username in users table for clarity
-- This resolves confusion between user_id (containing usernames like "alex.thompson") 
-- and the expected UUID format for user identification

-- Step 1: Rename the column from user_id to username
ALTER TABLE users RENAME COLUMN user_id TO username;

-- Step 2: Add a comment to clarify the purpose of this column
COMMENT ON COLUMN users.username IS 'Human-readable username/login identifier (e.g., alex.thompson, not a UUID)';

-- Step 3: Create an index on username for faster lookups (if not already exists)
CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);

-- Step 4: Add a unique constraint on username to prevent duplicates
ALTER TABLE users ADD CONSTRAINT unique_users_username UNIQUE (username);
