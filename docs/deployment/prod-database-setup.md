# Production Database Setup Guide

## Overview

This guide covers setting up a production Supabase database for Agent Tobi RAG. Based on analysis of the current development database, we recommend a **migration-based approach** with selective data copying and fresh production data generation.

## Current Database Analysis

### Schema Structure
The database contains the following main components:

**Core Tables:**
- `users` - User management (employees, customers, admin)
- `conversations` - Chat conversations 
- `messages` - Individual messages within conversations
- `conversation_summaries` - AI-generated conversation summaries

**RAG System Tables:**
- `data_sources` - Content sources for RAG (websites, documents)
- `documents` (renamed to `document_chunks`) - Text chunks for vector search
- `embeddings` - Vector embeddings for semantic search
- `query_logs` - Search query logging
- `response_feedback` - User feedback on AI responses

**CRM/Sales Tables:**
- `branches` - Sales branch locations
- `employees` - Sales staff and managers
- `customers` - Customer records
- `vehicles` - Vehicle inventory
- `opportunities` - Sales opportunities/leads
- `transactions` - Sales transactions
- `pricing` - Vehicle pricing and promotions
- `activities` - Sales activities (calls, meetings, demos)

**Memory & Context Tables:**
- `user_master_summaries` - Long-term user context summaries
- `memory_access_patterns` - User behavior patterns
- `customer_warmth` - Customer engagement scoring
- `customer_warmth_history` - Historical warmth tracking

**System Tables:**
- `system_metrics` - Performance monitoring
- `conflict_logs` - Data conflict resolution
- `proactive_suggestions` - AI-generated suggestions

## Recommended Approach: Clean Migration Strategy

### Phase 1: Create Production Supabase Project

1. **Create New Supabase Project**
   ```bash
   # Navigate to https://supabase.com/dashboard
   # Click "New Project"
   # Choose appropriate organization
   # Select region closest to your users
   # Use strong database password
   ```

2. **Enable Required Extensions**
   ```sql
   -- Run in Supabase SQL Editor
   CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
   CREATE EXTENSION IF NOT EXISTS "vector";
   ```

3. **Configure Project Settings**
   - Enable Row Level Security (RLS)
   - Configure Auth settings
   - Set up proper database password
   - Configure API settings

### Phase 2: Generate Clean Migration Files

#### Option A: Generate from Current State (Recommended)
```bash
# In your local development environment
cd /Users/markbanaria/agent-tobi-rag

# Generate a clean schema dump from current database
supabase db dump --data-only=false --schema-only=true > clean_schema.sql

# Create a new migration file
supabase migration new create_production_schema

# Copy the cleaned schema to the new migration file
# Remove any development-specific data or test entries
```

#### Option B: Consolidate Existing Migrations
```bash
# Create a consolidated migration that combines all essential schema changes
supabase migration new consolidated_production_schema

# Manually combine essential parts from:
# - 20250108000000_create_base_schema.sql
# - 20250115000000_create_crm_sales_tables.sql
# - 20250120000000_add_long_term_memory_tables.sql
# - 20250128000000_create_customer_warmth_table.sql
# - Recent RLS and function updates
```

### Phase 3: Apply Migrations to Production

1. **Link to Production Database**
   ```bash
   # Update supabase/config.toml with production project details
   supabase link --project-ref YOUR_PROD_PROJECT_ID
   ```

2. **Run Migrations**
   ```bash
   # Apply all migrations to production
   supabase db push

   # Alternative: Apply specific migration files
   supabase migration up
   ```

3. **Verify Schema**
   ```bash
   # Check if all tables were created correctly
   supabase db diff
   ```

## Data Population Strategy

### Tables to Keep Empty (Start Fresh)

**User-Generated Content:**
- ✅ `messages` - Start with clean message history
- ✅ `conversations` - Start with clean conversation history  
- ✅ `conversation_summaries` - Will be generated as users interact
- ✅ `query_logs` - Start fresh for production analytics
- ✅ `response_feedback` - Collect fresh production feedback
- ✅ `activities` - Sales activities should be entered fresh
- ✅ `memory_access_patterns` - Will build up from real usage
- ✅ `user_master_summaries` - Will be generated from real conversations

**System Logs:**
- ✅ `conflict_logs` - Start fresh
- ✅ `system_metrics` - Start fresh for production monitoring
- ✅ `proactive_suggestions` - Will be generated based on real usage

### Tables to Copy/Generate Mock Data

**Essential Business Data:**
- 🔄 `branches` - Copy real branch information
- 🔄 `employees` - Copy real employee data (sanitized)
- 🔄 `vehicles` - Copy current inventory
- 🔄 `pricing` - Copy current pricing structures

**Content for RAG System:**
- 🔄 `data_sources` - Copy essential content sources
- 🔄 `documents`/`document_chunks` - Copy processed content
- 🔄 `embeddings` - Copy vector embeddings (or regenerate)

**Reference Data:**
- 🔄 `customers` - Generate realistic mock customers OR import sanitized real data
- 🔄 `opportunities` - Generate mock sales opportunities for testing
- 🔄 `transactions` - Generate sample transaction history
- 🔄 `customer_warmth` - Generate initial warmth scores

**User Management:**
- 🔄 `users` - Create production user accounts (employees only initially)

### Phase 4: Data Population Scripts

#### Script 1: Copy Essential Business Data
```sql
-- Copy branch data (update with real production branches)
INSERT INTO branches (name, region, address, brand) VALUES
('Main Showroom', 'central', '123 Main St, Metro Manila', 'Toyota'),
('North Branch', 'north', '456 North Ave, Quezon City', 'Toyota'),
('South Branch', 'south', '789 South Blvd, Makati', 'Toyota');

-- Copy real employee data (sanitized)
-- NOTE: Update with actual employee information
INSERT INTO employees (name, position, email, phone, branch_id) VALUES
('Production Manager', 'manager', 'manager@company.com', '+63901234567', 
 (SELECT id FROM branches WHERE name = 'Main Showroom'));
```

#### Script 2: Generate Mock Customer Data
```sql
-- Generate realistic mock customers for testing
INSERT INTO customers (name, email, phone, is_for_business) VALUES
('Sample Customer 1', 'customer1@example.com', '+63901111111', false),
('Sample Business Corp', 'contact@samplebiz.com', '+63902222222', true),
('Test Individual', 'individual@test.com', '+63903333333', false);
```

#### Script 3: Copy Content Data
```bash
# Script to copy RAG content from development to production
# This preserves processed embeddings and avoids reprocessing
python scripts/copy_rag_content.py --source dev --target prod
```

## Environment Configuration

### Production Environment Variables
```bash
# Update these in your production deployment
SUPABASE_URL=https://your-prod-project.supabase.co
SUPABASE_ANON_KEY=your_prod_anon_key
SUPABASE_SERVICE_KEY=your_prod_service_key
SUPABASE_DB_PASSWORD=your_prod_db_password

# Environment marker
ENVIRONMENT=production
```

### Railway Deployment Updates
```bash
# Update railway.json if needed
{
  "environments": {
    "production": {
      "variables": {
        "SUPABASE_URL": "https://your-prod-project.supabase.co",
        "SUPABASE_ANON_KEY": "your_prod_anon_key",
        "SUPABASE_SERVICE_KEY": "your_prod_service_key"
      }
    }
  }
}
```

## Security Considerations

### Row Level Security (RLS)
```sql
-- Ensure RLS is enabled on all sensitive tables
ALTER TABLE users ENABLE ROW LEVEL SECURITY;
ALTER TABLE conversations ENABLE ROW LEVEL SECURITY;
ALTER TABLE messages ENABLE ROW LEVEL SECURITY;
ALTER TABLE customers ENABLE ROW LEVEL SECURITY;
ALTER TABLE opportunities ENABLE ROW LEVEL SECURITY;
ALTER TABLE activities ENABLE ROW LEVEL SECURITY;

-- Update RLS policies for production use
-- (policies should already be defined in migrations)
```

### Data Sanitization
- Remove any development test data
- Sanitize employee personal information
- Use mock customer data instead of real customer data initially
- Remove any API keys or secrets from database entries

## Validation & Testing

### Post-Migration Checklist
- [ ] All tables created successfully
- [ ] All indexes created
- [ ] All functions and triggers working
- [ ] RLS policies active and working
- [ ] Extensions enabled (uuid-ossp, vector)
- [ ] Essential data populated
- [ ] API connections working
- [ ] Authentication working
- [ ] Vector search working
- [ ] Real-time subscriptions working

### Testing Strategy
1. **Schema Validation**: Run automated tests against production schema
2. **API Testing**: Test all CRUD operations
3. **RAG Testing**: Verify vector search and embedding generation
4. **Auth Testing**: Test user authentication and authorization
5. **Performance Testing**: Test with expected load

## Maintenance & Monitoring

### Backup Strategy
```bash
# Set up automated backups in Supabase dashboard
# Configure point-in-time recovery
# Test restore procedures
```

### Monitoring Setup
```sql
-- Enable performance insights
-- Set up log retention
-- Configure alerts for critical metrics
```

## Migration Execution Timeline

1. **Day 1**: Create production Supabase project and configure settings
2. **Day 2**: Generate and test clean migration files locally
3. **Day 3**: Apply migrations to production and populate essential data
4. **Day 4**: Test all functionality and fix any issues
5. **Day 5**: Update application configuration and deploy
6. **Day 6**: Final testing and go-live
7. **Day 7**: Monitor and address any production issues

## Rollback Plan

In case of issues:
1. Keep development database as fallback
2. Document all configuration changes
3. Have environment variable rollback ready
4. Prepare application rollback deployment

## Questions for Decision

Please confirm your preferences for the following:

### Data Population Decisions:
1. **Customer Data**: Use mock data or import sanitized real customers?
2. **Sales Data**: Generate sample opportunities/transactions or import historical data?
3. **Content Data**: Regenerate embeddings or copy from development?
4. **User Accounts**: Start with employees only or include test customer accounts?

### Migration Strategy:
1. **Approach**: Generate clean consolidated migration or use existing migration chain?
2. **Timing**: Immediate migration or staged rollout?
3. **Validation**: Full testing cycle before production or parallel testing?

Let me know your preferences and I can provide more specific scripts and guidance for your chosen approach.
