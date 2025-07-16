-- Add test users for long-term memory tests

-- Insert test users to satisfy foreign key constraints
INSERT INTO users (id, email, display_name, created_at, updated_at)
VALUES 
    ('123e4567-e89b-12d3-a456-426614174000', 'test1@example.com', 'Test User 1', NOW(), NOW()),
    ('123e4567-e89b-12d3-a456-426614174001', 'test2@example.com', 'Test User 2', NOW(), NOW()),
    ('123e4567-e89b-12d3-a456-426614174002', 'test3@example.com', 'Test User 3', NOW(), NOW()),
    ('123e4567-e89b-12d3-a456-426614174003', 'test4@example.com', 'Test User 4', NOW(), NOW())
ON CONFLICT (id) DO NOTHING; 