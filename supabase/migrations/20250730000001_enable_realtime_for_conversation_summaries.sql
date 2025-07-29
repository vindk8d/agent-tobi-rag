-- Enable realtime for conversation_summaries table so hot reload works in ConversationSummary component
ALTER PUBLICATION supabase_realtime ADD TABLE conversation_summaries;

-- Verify the table is now included in the publication
-- (This is just for verification, the SELECT won't affect the migration)
-- SELECT tablename FROM pg_publication_tables WHERE pubname = 'supabase_realtime' AND tablename = 'conversation_summaries'; 