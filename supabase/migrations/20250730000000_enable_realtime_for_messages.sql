-- Enable realtime for messages table so hot reload works in the watchtower
ALTER PUBLICATION supabase_realtime ADD TABLE messages;

-- Also enable realtime for conversations table for conversation list updates
ALTER PUBLICATION supabase_realtime ADD TABLE conversations;

-- Verify the tables are now included in the publication
-- (This is just for verification, the SELECT won't affect the migration)
-- SELECT tablename FROM pg_publication_tables WHERE pubname = 'supabase_realtime'; 