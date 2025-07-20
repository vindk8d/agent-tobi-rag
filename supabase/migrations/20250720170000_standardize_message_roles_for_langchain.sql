-- Standardize message roles to be directly compatible with LangChain message types
-- This eliminates the need for conversion logic and improves performance

-- Step 1: First remove the old constraint that blocks 'ai' role
ALTER TABLE messages DROP CONSTRAINT IF EXISTS messages_role_check;

-- Step 2: Update existing data to use LangChain-compatible role names
UPDATE messages 
SET role = CASE 
    WHEN role = 'bot' THEN 'ai'
    WHEN role = 'HITL' THEN 'assistant'
    ELSE role  -- 'human' and 'ai' stay as-is
END;

-- Step 3: Add the new constraint with LangChain-compatible types
ALTER TABLE messages ADD CONSTRAINT messages_role_check 
    CHECK (role IN ('ai', 'human', 'assistant', 'system', 'user', 'tool', 'function'));

-- Step 3: Update any existing functions or triggers that might reference old role values
-- (Currently none exist, but this is for future-proofing)

-- Step 4: Add a helpful comment documenting the role types
COMMENT ON COLUMN messages.role IS 'Message role compatible with LangChain message types: ai (assistant messages), human (user messages), assistant (human-in-the-loop), system (system messages), user (alternative for human), tool/function (tool responses)';

-- Verification: Check the updated data
-- (Uncomment to verify the migration worked)
-- SELECT role, COUNT(*) FROM messages GROUP BY role ORDER BY COUNT(*) DESC; 