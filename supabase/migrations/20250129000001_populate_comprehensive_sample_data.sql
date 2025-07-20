-- Comprehensive Sample Data Population
-- Populates users, conversations, messages, and links all CRM relationships for complete testing

-- First, let's update existing users with proper user_id values and link them to employees/customers
UPDATE users SET 
    user_id = 'test-user-1',
    display_name = 'Alex Thompson (Employee)',
    user_type = 'employee',
    employee_id = (SELECT id FROM employees WHERE email = 'alex.thompson@company.com'),
    is_verified = true
WHERE email = 'test1@example.com';

UPDATE users SET 
    user_id = 'test-user-2', 
    display_name = 'Jennifer Lee (Customer)',
    user_type = 'customer',
    customer_id = (SELECT id FROM customers WHERE email = 'jennifer.lee@email.com'),
    is_verified = true
WHERE email = 'test2@example.com';

UPDATE users SET 
    user_id = 'test-user-3',
    display_name = 'TechCorp Solutions (Customer)',
    user_type = 'customer', 
    customer_id = (SELECT id FROM customers WHERE email = 'procurement@techcorp.com'),
    is_verified = true
WHERE email = 'test3@example.com';

UPDATE users SET 
    user_id = 'test-user-4',
    display_name = 'Sarah Johnson (Manager)',
    user_type = 'employee',
    employee_id = (SELECT id FROM employees WHERE email = 'sarah.johnson@company.com'),
    is_verified = true  
WHERE email = 'test4@example.com';

-- Insert comprehensive additional users covering all roles and scenarios
INSERT INTO users (id, user_id, email, display_name, user_type, employee_id, customer_id, is_verified, preferences, permissions) VALUES

-- More Employee Users
('550e8400-e29b-41d4-a716-446655440001', 'sales-mike-001', 'mike.davis@company.com', 'Mike Davis (Manager)', 'employee', 
 (SELECT id FROM employees WHERE email = 'mike.davis@company.com'), null, true, 
 '{"language": "en", "notifications": true, "theme": "light"}', 
 '{"crm_read": true, "crm_write": true, "reports": true}'),

('550e8400-e29b-41d4-a716-446655440002', 'sales-lisa-002', 'lisa.wang@company.com', 'Lisa Wang (Manager)', 'employee',
 (SELECT id FROM employees WHERE email = 'lisa.wang@company.com'), null, true,
 '{"language": "en", "notifications": true, "theme": "dark"}',
 '{"crm_read": true, "crm_write": true, "reports": true, "admin": true}'),

-- More Customer Users  
('550e8400-e29b-41d4-a716-446655440003', 'customer-mark-003', 'mark.johnson@email.com', 'Mark Johnson', 'customer',
 null, (SELECT id FROM customers WHERE email = 'mark.johnson@email.com'), true,
 '{"language": "en", "notifications": true, "preferred_contact": "email"}', '{}'),

('550e8400-e29b-41d4-a716-446655440004', 'customer-robert-004', 'robert.brown@email.com', 'Robert Brown', 'customer', 
 null, (SELECT id FROM customers WHERE email = 'robert.brown@email.com'), true,
 '{"language": "en", "notifications": false, "preferred_contact": "phone"}', '{}'),

-- Admin and System Users
('550e8400-e29b-41d4-a716-446655440005', 'admin-tobi-001', 'admin@tobi.ai', 'Tobi System Admin', 'admin', 
 null, null, true,
 '{"language": "en", "notifications": true, "theme": "dark", "admin_view": "advanced"}',
 '{"super_admin": true, "user_management": true, "system_config": true}'),

('550e8400-e29b-41d4-a716-446655440006', 'system-agent-001', 'system@tobi.ai', 'Tobi Sales Agent', 'system',
 null, null, true,
 '{"auto_response": true, "learning_mode": true, "memory_consolidation": true}',
 '{"memory_access": true, "conversation_create": true, "crm_read": true}'),

-- Test users for memory persistence scenarios
('550e8400-e29b-41d4-a716-446655440007', 'memory-test-001', 'memtest1@example.com', 'Memory Test User 1', 'customer',
 null, null, true, '{"test_scenario": "short_term_memory"}', '{}'),

('550e8400-e29b-41d4-a716-446655440008', 'memory-test-002', 'memtest2@example.com', 'Memory Test User 2', 'customer', 
 null, null, true, '{"test_scenario": "long_term_memory"}', '{}'),

('550e8400-e29b-41d4-a716-446655440009', 'memory-test-003', 'memtest3@example.com', 'Memory Test User 3', 'customer',
 null, null, true, '{"test_scenario": "cross_conversation_memory"}', '{}');

-- Create sample conversations for memory testing
INSERT INTO conversations (id, user_id, title, message_count, created_at, updated_at) VALUES

-- Conversations for short-term memory testing
('c0000001-0000-0000-0000-000000000001', (SELECT id FROM users WHERE user_id = 'test-user-1'), 'Vehicle Inquiry - Honda Accord', 5, NOW() - INTERVAL '2 hours', NOW() - INTERVAL '1 hour'),
('c0000001-0000-0000-0000-000000000002', (SELECT id FROM users WHERE user_id = 'test-user-2'), 'Price Quote Request', 8, NOW() - INTERVAL '1 day', NOW() - INTERVAL '12 hours'),
('c0000001-0000-0000-0000-000000000003', (SELECT id FROM users WHERE user_id = 'memory-test-001'), 'Multi-turn Vehicle Search', 15, NOW() - INTERVAL '3 days', NOW() - INTERVAL '1 day'),

-- Conversations for long-term memory testing  
('c0000001-0000-0000-0000-000000000004', (SELECT id FROM users WHERE user_id = 'memory-test-002'), 'Extended Purchase Journey', 25, NOW() - INTERVAL '2 weeks', NOW() - INTERVAL '1 week'),
('c0000001-0000-0000-0000-000000000005', (SELECT id FROM users WHERE user_id = 'memory-test-002'), 'Follow-up After Test Drive', 12, NOW() - INTERVAL '1 week', NOW() - INTERVAL '3 days'),
('c0000001-0000-0000-0000-000000000006', (SELECT id FROM users WHERE user_id = 'memory-test-003'), 'Cross-session Preferences', 18, NOW() - INTERVAL '1 month', NOW() - INTERVAL '2 weeks'),

-- Business customer conversations
('c0000001-0000-0000-0000-000000000007', (SELECT id FROM users WHERE user_id = 'test-user-3'), 'Fleet Purchase Discussion', 22, NOW() - INTERVAL '1 week', NOW() - INTERVAL '2 days'),
('c0000001-0000-0000-0000-000000000008', (SELECT id FROM users WHERE user_id = 'customer-mark-003'), 'Family Car Shopping', 14, NOW() - INTERVAL '5 days', NOW() - INTERVAL '1 day');

-- Insert sample messages for memory testing scenarios
INSERT INTO messages (id, conversation_id, user_id, role, content, created_at) VALUES

-- Short-term memory conversation (Vehicle Inquiry)
('10000001-0000-0000-0000-000000000001', 'c0000001-0000-0000-0000-000000000001', 
 (SELECT id FROM users WHERE user_id = 'test-user-1'), 'human', 
 'Hi, I''m interested in learning about Honda Accord models. What options do you have available?', 
 NOW() - INTERVAL '2 hours'),
 
('10000001-0000-0000-0000-000000000002', 'c0000001-0000-0000-0000-000000000001', 
 (SELECT id FROM users WHERE user_id = 'system-agent-001'), 'bot',
 'Great choice! We have several Honda Accord models available. Are you looking for a new or used vehicle? And what''s your budget range?',
 NOW() - INTERVAL '2 hours' + INTERVAL '2 minutes'),

('10000001-0000-0000-0000-000000000003', 'c0000001-0000-0000-0000-000000000001',
 (SELECT id FROM users WHERE user_id = 'test-user-1'), 'human',
 'I''m looking for a new 2024 model, budget around $35,000. I need something reliable for daily commuting.',
 NOW() - INTERVAL '2 hours' + INTERVAL '5 minutes'),

('10000001-0000-0000-0000-000000000004', 'c0000001-0000-0000-0000-000000000001',
 (SELECT id FROM users WHERE user_id = 'system-agent-001'), 'bot', 
 'Perfect! The 2024 Honda Accord LX starts at $28,000 and the Sport is around $32,000. Both are excellent for commuting with great fuel economy. Would you like to know more about specific features or schedule a test drive?',
 NOW() - INTERVAL '2 hours' + INTERVAL '7 minutes'),

('10000001-0000-0000-0000-000000000005', 'c0000001-0000-0000-0000-000000000001',
 (SELECT id FROM users WHERE user_id = 'test-user-1'), 'human',
 'The Sport sounds interesting. Can you tell me about the warranty and financing options?',
 NOW() - INTERVAL '1 hour' + INTERVAL '30 minutes'),

-- Extended conversation for long-term memory testing
('20000002-0000-0000-0000-000000000001', 'c0000001-0000-0000-0000-000000000004',
 (SELECT id FROM users WHERE user_id = 'memory-test-002'), 'human',
 'Hi, I''ve been thinking about buying a car for about 6 months now. I have a family of 4 and need something spacious and safe.',
 NOW() - INTERVAL '2 weeks'),

('20000002-0000-0000-0000-000000000002', 'c0000001-0000-0000-0000-000000000004',
 (SELECT id FROM users WHERE user_id = 'system-agent-001'), 'bot',
 'I understand you need a family-friendly vehicle with space and safety as priorities. Have you considered SUVs or would you prefer a sedan? What''s your budget range?',
 NOW() - INTERVAL '2 weeks' + INTERVAL '3 minutes'),

-- Business conversation
('30000003-0000-0000-0000-000000000001', 'c0000001-0000-0000-0000-000000000007',
 (SELECT id FROM users WHERE user_id = 'test-user-3'), 'human',
 'We''re TechCorp Solutions and looking to purchase a fleet of 10 vehicles for our sales team. We need reliable, professional-looking cars with good fuel economy.',
 NOW() - INTERVAL '1 week'),

('30000003-0000-0000-0000-000000000002', 'c0000001-0000-0000-0000-000000000007',
 (SELECT id FROM users WHERE user_id = 'system-agent-001'), 'bot',
 'Excellent! Fleet purchases are one of our specialties. For 10 vehicles, we can offer volume discounts. Are you looking at sedans like the Accord or Civic, or would you consider hybrid options for better fuel economy?',
 NOW() - INTERVAL '1 week' + INTERVAL '5 minutes');

-- Create sample conversation summaries for memory testing
INSERT INTO conversation_summaries (id, conversation_id, user_id, summary_text, summary_type, message_count, created_at) VALUES

('50000001-0000-0000-0000-000000000001', 'c0000001-0000-0000-0000-000000000001', 
 (SELECT id FROM users WHERE user_id = 'test-user-1'),
 'Customer Alex Thompson inquired about Honda Accord models, specifically interested in 2024 new vehicles with $35,000 budget for daily commuting. Showed interest in Sport model and asked about warranty/financing.',
 'periodic', 5, NOW() - INTERVAL '1 hour'),

('50000001-0000-0000-0000-000000000002', 'c0000001-0000-0000-0000-000000000004',
 (SELECT id FROM users WHERE user_id = 'memory-test-002'),
 'Long-term customer considering vehicle purchase for 6 months. Family of 4 needs spacious, safe vehicle. Discussing SUVs vs sedans and budget considerations.',
 'periodic', 15, NOW() - INTERVAL '1 week'),

('50000001-0000-0000-0000-000000000003', 'c0000001-0000-0000-0000-000000000007',
 (SELECT id FROM users WHERE user_id = 'test-user-3'),  
 'TechCorp Solutions fleet purchase inquiry for 10 vehicles for sales team. Requirements: reliable, professional appearance, good fuel economy. Discussing volume discounts and vehicle options.',
 'periodic', 12, NOW() - INTERVAL '2 days');

-- Create user master summaries for testing consolidation
INSERT INTO user_master_summaries (id, user_id, master_summary, total_conversations, total_messages, conversations_included, preferences) VALUES

('60000001-0000-0000-0000-000000000001', 'test-user-1',
 'Alex Thompson is an Account Executive employee interested in Honda Accord vehicles. Prefers 2024 new models, budget around $35,000, focused on reliability for daily commuting. Interested in Sport model features and financing options.',
 1, 5, ARRAY['c0000001-0000-0000-0000-000000000001'::uuid],
 '{"preferred_brands": ["Honda"], "budget_range": "$30,000-$35,000", "usage": "commuting", "body_type": "sedan"}'),

('60000001-0000-0000-0000-000000000002', 'memory-test-002', 
 'Long-term prospective customer with family of 4. Has been considering vehicle purchase for 6+ months. Prioritizes space and safety. Evaluating SUVs vs sedans. Methodical decision-making approach.',
 2, 37, ARRAY['c0000001-0000-0000-0000-000000000004'::uuid, 'c0000001-0000-0000-0000-000000000005'::uuid],
 '{"family_size": 4, "priorities": ["space", "safety"], "decision_timeline": "extended", "vehicle_types": ["SUV", "sedan"]}'),

('60000001-0000-0000-0000-000000000003', 'test-user-3',
 'TechCorp Solutions business customer interested in fleet purchases. Professional requirements, fuel economy focused. Decision maker for company vehicle procurement. Volume purchase potential.',
 1, 22, ARRAY['c0000001-0000-0000-0000-000000000007'::uuid], 
 '{"business_customer": true, "fleet_size": 10, "requirements": ["professional", "fuel_economy"], "purchase_type": "volume"}');

-- Insert sample long-term memories for testing
INSERT INTO long_term_memories (namespace, key, value, memory_type, source_thread_id, created_at) VALUES

-- User preferences and behavior patterns
(ARRAY['user', 'test-user-1', 'preferences'], 'vehicle_preferences', 
 '{"preferred_brands": ["Honda", "Toyota"], "budget_max": 35000, "usage_type": "commuting", "body_style": "sedan"}',
 'semantic', 'c0000001-0000-0000-0000-000000000001', NOW() - INTERVAL '1 hour'),

(ARRAY['user', 'test-user-1', 'history'], 'interaction_patterns',
 '{"response_time": "quick", "detail_level": "moderate", "decision_style": "analytical", "contact_preference": "email"}', 
 'behavioral', 'c0000001-0000-0000-0000-000000000001', NOW() - INTERVAL '1 hour'),

-- Business customer patterns
(ARRAY['business', 'test-user-3', 'requirements'], 'fleet_specifications',
 '{"quantity": 10, "vehicle_type": "professional", "features": ["fuel_efficient", "reliable", "warranty"], "budget_per_unit": 30000}',
 'semantic', 'c0000001-0000-0000-0000-000000000007', NOW() - INTERVAL '2 days'),

-- Cross-conversation insights  
(ARRAY['insights', 'customer_segments'], 'family_buyers',
 '{"characteristics": ["safety_focused", "space_requirements", "extended_research"], "typical_timeline": "3-6 months", "price_sensitivity": "moderate"}',
 'procedural', null, NOW() - INTERVAL '1 week'),

(ARRAY['insights', 'business_customers'], 'fleet_buyers',
 '{"decision_factors": ["volume_discounts", "fuel_economy", "maintenance_costs"], "typical_quantity": "5-20 units", "decision_timeline": "1-3 months"}',
 'procedural', null, NOW() - INTERVAL '3 days');

-- Add sample data sources for RAG testing
INSERT INTO data_sources (id, name, type, status, url, document_count, description, document_type, configuration) VALUES

('80000001-0000-0000-0000-000000000001', 'Honda Vehicle Specifications', 'web', 'active', 
 'https://honda.com/vehicles', 15, 'Official Honda vehicle specifications and features', 'html',
 '{"crawl_frequency": "weekly", "sections": ["specifications", "pricing", "features"]}'),

('80000001-0000-0000-0000-000000000002', 'Toyota Fleet Information', 'web', 'active',
 'https://toyota.com/fleet', 8, 'Toyota fleet and business vehicle information', 'html', 
 '{"crawl_frequency": "monthly", "sections": ["fleet", "business", "financing"]}'),

('80000001-0000-0000-0000-000000000003', 'Company Policy Manual', 'text', 'active',
 null, 5, 'Internal company policies and procedures', 'text',
 '{"source_type": "internal", "last_updated": "2024-01-15"}');

-- Add sample documents for RAG
INSERT INTO documents (id, data_source_id, content, metadata, document_type, processed_at, embedding_count) VALUES

('90000001-0000-0000-0000-000000000001', '80000001-0000-0000-0000-000000000001',
 '2024 Honda Accord Sport features include turbocharged engine, 252 HP, CVT transmission, Honda Sensing suite, 32 city/42 highway MPG. Starting MSRP $32,000. Available in multiple colors including Sport-exclusive wheels.',
 '{"title": "Honda Accord Sport 2024", "category": "specifications", "brand": "Honda", "model": "Accord", "trim": "Sport"}',
 'html', NOW() - INTERVAL '1 day', 3),

('90000001-0000-0000-0000-000000000002', '80000001-0000-0000-0000-000000000002', 
 'Toyota Camry fleet program offers volume discounts for purchases of 5+ vehicles. Fleet customers receive extended warranty, priority service, and dedicated fleet representative. Hybrid options available with 51 MPG combined.',
 '{"title": "Toyota Fleet Program", "category": "fleet", "brand": "Toyota", "program": "business"}',
 'html', NOW() - INTERVAL '2 days', 4),

('90000001-0000-0000-0000-000000000003', '80000001-0000-0000-0000-000000000003',
 'Company policy for vehicle purchases: All fleet vehicles must meet minimum safety ratings of 5-star NHTSA or Top Safety Pick IIHS. Fuel economy minimum 30 MPG combined. Maximum per-unit cost $35,000 unless approved by director.',
 '{"title": "Fleet Purchase Policy", "category": "policy", "department": "procurement", "effective_date": "2024-01-01"}',
 'text', NOW() - INTERVAL '3 days', 2);

-- Create memory access patterns for optimization testing
INSERT INTO memory_access_patterns (user_id, memory_namespace, memory_key, access_frequency, context_relevance, access_context, retrieval_method, metadata) VALUES

((SELECT id FROM users WHERE user_id = 'test-user-1'), ARRAY['user', 'test-user-1', 'preferences'], 'vehicle_preferences', 5, 0.95, 'vehicle_search', 'semantic_search',
 '{"last_contexts": ["honda_inquiry", "price_discussion", "feature_comparison"], "relevance_scores": [0.95, 0.87, 0.92]}'),

((SELECT id FROM users WHERE user_id = 'test-user-3'), ARRAY['business', 'test-user-3', 'requirements'], 'fleet_specifications', 3, 0.88, 'fleet_inquiry', 'exact_match',
 '{"last_contexts": ["fleet_discussion", "volume_pricing", "specifications"], "relevance_scores": [0.88, 0.91, 0.85]}'),

((SELECT id FROM users WHERE user_id = 'system-agent-001'), ARRAY['insights', 'customer_segments'], 'family_buyers', 12, 0.78, 'customer_categorization', 'similarity_search',
 '{"usage_patterns": ["customer_profiling", "recommendation_engine"], "avg_relevance": 0.78}');

COMMENT ON TABLE users IS 'Enhanced with comprehensive test users covering all roles and memory test scenarios';
COMMENT ON TABLE conversations IS 'Sample conversations for testing short-term, long-term, and cross-session memory persistence';
COMMENT ON TABLE messages IS 'Realistic conversation flows for testing memory consolidation and retrieval';
COMMENT ON TABLE user_master_summaries IS 'Pre-populated summaries for testing memory consolidation functions';
COMMENT ON TABLE long_term_memories IS 'Sample semantic, episodic, and procedural memories for comprehensive testing';
COMMENT ON TABLE data_sources IS 'Sample data sources with documents for RAG functionality testing';
COMMENT ON TABLE memory_access_patterns IS 'Access patterns for testing memory optimization and retrieval analytics';

-- Update conversation message counts to match inserted messages
UPDATE conversations SET message_count = (
    SELECT COUNT(*) FROM messages WHERE messages.conversation_id = conversations.id
) WHERE id IN (
    'c0000001-0000-0000-0000-000000000001',
    'c0000001-0000-0000-0000-000000000004', 
    'c0000001-0000-0000-0000-000000000007'
); 