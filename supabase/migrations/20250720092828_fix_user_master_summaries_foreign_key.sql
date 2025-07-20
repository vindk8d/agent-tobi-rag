-- Fix foreign key constraint for user_master_summaries table
-- Change from referencing users(user_id) to users(id) and update data type

-- First, drop the existing foreign key constraint
ALTER TABLE user_master_summaries 
DROP CONSTRAINT IF EXISTS user_master_summaries_user_id_fkey;

-- Delete existing data since we can't easily convert text user_ids to uuids
DELETE FROM user_master_summaries;

-- Change user_id column from text to uuid to match users.id
ALTER TABLE user_master_summaries 
ALTER COLUMN user_id TYPE uuid USING user_id::uuid;

-- Add new foreign key constraint referencing users.id
ALTER TABLE user_master_summaries 
ADD CONSTRAINT user_master_summaries_user_id_fkey 
FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE;
