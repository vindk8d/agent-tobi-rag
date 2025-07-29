-- Add RLS policy to allow anonymous users to read conversation summaries
-- This allows the watchtower interface (which runs as anonymous) to display summaries

CREATE POLICY "Allow anonymous users to read conversation summaries"
ON conversation_summaries FOR SELECT
TO anon
USING (true);

-- Also add policy for authenticated users for consistency
CREATE POLICY "Allow authenticated users to read conversation summaries"
ON conversation_summaries FOR SELECT
TO authenticated
USING (true); 