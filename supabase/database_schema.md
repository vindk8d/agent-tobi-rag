# RAG-Tobi Database Schema Overview

## Architecture Overview

The RAG-Tobi salesperson copilot system uses a Supabase PostgreSQL database with pgvector extension for vector similarity search. The database is organized into 11 tables across 3 migration files, providing comprehensive functionality for document processing, conversation management, analytics, and intelligent assistance.

## Migration Files Structure

### 20250707195741_add_document_type_column.sql
- Complete RAG database schema setup
- All core tables with proper relationships
- Vector search capabilities with pgvector
- Analytics and intelligence features
- Security policies (RLS)

### 20250708030625_add_document_metadata_columns.sql
- Added structured columns to documents table
- Migrated metadata to proper typed columns
- Performance optimization with indexes
- Backward compatibility maintained

### 20250708031942_add_document_type_column_fix.sql
- Added missing document_type column with enum
- Graceful handling of existing data
- Index optimization for document type queries

## Database Tables

### 1. Content Management Tables

#### `data_sources`
**Purpose**: Manages different sources of information (websites, documents, APIs)
```sql
- id: UUID (Primary Key)
- name: VARCHAR(255) - Human-readable name
- source_type: ENUM - Type of source (website, document)
- url: TEXT - Source URL or identifier
- file_path: TEXT - File path for uploaded documents
- status: ENUM - active, inactive, error, processing, failed
- document_type: ENUM - Document type (pdf, word, text, html, markdown, web_page)
- chunk_count: INTEGER - Number of chunks generated from this source
- scraping_frequency: ENUM - How often to scrape (daily, weekly, monthly, manual)
- last_scraped_at: TIMESTAMP - Last successful scraping
- last_success: TIMESTAMP - Last successful processing
- error_count: INTEGER - Number of errors encountered
- last_error: TEXT - Last error message
- description: TEXT - Optional description
- configuration: JSONB - Source-specific configuration
- created_at: TIMESTAMP
- updated_at: TIMESTAMP
- metadata: JSONB - Additional metadata
```

#### `document_chunks`
**Purpose**: Stores processed document chunks ready for embedding
```sql
- id: UUID (Primary Key)
- data_source_id: UUID - Foreign key to data_sources
- title: TEXT - Document chunk title
- content: TEXT - Document chunk content
- document_type: ENUM - Document type (pdf, word, text, html, markdown, web_page)
- chunk_index: INTEGER - Order within source (default: 0)
- status: ENUM - Processing status (pending, processing, completed, failed)
- word_count: INTEGER - Number of words in content
- character_count: INTEGER - Number of characters in content
- processed_at: TIMESTAMP - When document processing completed
- storage_path: TEXT - File system path or storage identifier
- original_filename: TEXT - Original filename as uploaded by user
- file_size: BIGINT - File size in bytes
- created_at: TIMESTAMP
- updated_at: TIMESTAMP
- metadata: JSONB - Additional metadata (legacy, prefer specific columns)
```

### 2. Vector Search Tables

#### `embeddings`
**Purpose**: Stores vector embeddings for similarity search
```sql
- id: UUID (Primary Key)
- document_chunk_id: UUID - Foreign key to document_chunks
- embedding: VECTOR(1536) - OpenAI text-embedding-3-small vector
- model_name: VARCHAR(100) - Embedding model used (default: text-embedding-3-small)
- created_at: TIMESTAMP
- metadata: JSONB - Additional embedding metadata
```

### 3. Conversation Management Tables

#### `conversations`
**Purpose**: Manages user conversation sessions
```sql
- id: UUID (Primary Key)
- user_id: UUID - User identifier
- title: TEXT - Conversation title
- metadata: JSONB - Additional conversation data
- created_at: TIMESTAMP
- updated_at: TIMESTAMP
```

#### `messages`
**Purpose**: Stores individual messages within conversations
```sql
- id: UUID (Primary Key)
- conversation_id: UUID - Foreign key to conversations
- role: VARCHAR(20) - user, assistant, system
- content: TEXT - Message content
- metadata: JSONB - Additional message data
- created_at: TIMESTAMP
```

### 4. Analytics Tables

#### `query_logs`
**Purpose**: Tracks all user queries for analytics
```sql
- id: UUID (Primary Key)
- user_id: UUID - User identifier
- query: TEXT - User query
- response: TEXT - Generated response
- sources_used: JSONB - Array of source IDs used
- response_time: INTERVAL - Time to generate response
- created_at: TIMESTAMP
```

#### `conflict_logs`
**Purpose**: Logs when conflicting information is detected
```sql
- id: UUID (Primary Key)
- query_id: UUID - Foreign key to query_logs
- source_ids: JSONB - Array of conflicting source IDs
- conflict_type: VARCHAR(50) - Type of conflict
- details: JSONB - Conflict details
- created_at: TIMESTAMP
```

#### `source_conflicts`
**Purpose**: Tracks ongoing conflicts between sources
```sql
- id: UUID (Primary Key)
- source_a_id: UUID - First source
- source_b_id: UUID - Second source
- conflict_type: VARCHAR(50) - Type of conflict
- description: TEXT - Conflict description
- status: VARCHAR(20) - active, resolved, ignored
- created_at: TIMESTAMP
- resolved_at: TIMESTAMP
```

### 5. Intelligence Tables

#### `proactive_suggestions`
**Purpose**: Stores AI-generated proactive suggestions
```sql
- id: UUID (Primary Key)
- user_id: UUID - Target user
- type: VARCHAR(50) - Suggestion type
- content: TEXT - Suggestion content
- context: JSONB - Context information
- priority: INTEGER - Suggestion priority
- status: VARCHAR(20) - pending, shown, dismissed
- created_at: TIMESTAMP
- expires_at: TIMESTAMP
```

#### `response_feedback`
**Purpose**: Collects user feedback on AI responses
```sql
- id: UUID (Primary Key)
- query_id: UUID - Foreign key to query_logs
- user_id: UUID - User providing feedback
- rating: INTEGER - Numeric rating (1-5)
- feedback_text: TEXT - Optional feedback text
- created_at: TIMESTAMP
```

#### `system_metrics`
**Purpose**: Stores system performance metrics
```sql
- id: UUID (Primary Key)
- metric_name: VARCHAR(100) - Metric identifier
- value: NUMERIC - Metric value
- metadata: JSONB - Additional metric data
- timestamp: TIMESTAMP - When metric was recorded
```

## Database Functions

### Vector Search Functions

#### `similarity_search(query_embedding, match_threshold, match_count)`
- Performs vector similarity search using cosine similarity
- Returns matching documents with similarity scores
- Configurable similarity threshold and result count

#### `hybrid_search(query_text, query_embedding, match_threshold, match_count)`
- Combines text search with vector similarity
- Uses PostgreSQL full-text search + vector search
- Returns ranked results from both methods

### Intelligence Functions

#### `detect_conflicts(source_ids)`
- Analyzes content from multiple sources
- Identifies potential conflicts or contradictions
- Returns conflict analysis and recommendations

#### `get_proactive_suggestions(user_id, limit)`
- Retrieves active suggestions for a user
- Orders by priority and relevance
- Filters out expired suggestions

## Database Views

### `system_dashboard`
**Purpose**: Real-time system metrics and health overview
```sql
- total_documents: COUNT of all documents
- total_embeddings: COUNT of all embeddings
- avg_response_time: Average query response time
- daily_queries: Queries in last 24 hours
- active_conflicts: Unresolved conflicts
- system_health: Overall system status
```

## Security Features

### Row Level Security (RLS)
- **Enabled Tables**: `conversations`, `messages`, `query_logs`, `response_feedback`
- **User Isolation**: Users can only access their own data
- **Admin Override**: System administrators have full access
- **Audit Trail**: All access is logged for security monitoring

### Data Protection
- **Encryption**: All sensitive data encrypted at rest
- **Access Control**: Role-based permissions
- **API Keys**: Secure storage of external API credentials
- **Backup**: Automated daily backups with point-in-time recovery

## Vector Configuration

### OpenAI Integration
- **Model**: text-embedding-3-small
- **Dimensions**: 1536 (optimized for database compatibility)
- **Similarity**: Cosine similarity for search
- **Batch Processing**: Efficient bulk embedding operations

### Performance Optimization
- **Indexing**: HNSW index on embedding vectors
- **Caching**: Frequently accessed embeddings cached
- **Rate Limiting**: API request throttling
- **Monitoring**: Real-time performance metrics

## Usage Patterns

### Document Processing Flow
1. Data source configuration in `data_sources`
2. Document ingestion and chunking in `documents`
3. Embedding generation in `embeddings`
4. Vector indexing for fast search

### Query Processing Flow
1. User query received and logged in `query_logs`
2. Vector similarity search via `similarity_search()`
3. Conflict detection via `detect_conflicts()`
4. Response generation and feedback collection

### Analytics Pipeline
1. Query patterns analyzed from `query_logs`
2. Source effectiveness measured via `response_feedback`
3. Conflicts tracked in `conflict_logs` and `source_conflicts`
4. Proactive suggestions generated in `proactive_suggestions`

## Maintenance

### Regular Tasks
- **Reindexing**: Weekly vector index optimization
- **Cleanup**: Monthly removal of old logs and expired suggestions
- **Backup Verification**: Daily backup integrity checks
- **Performance Monitoring**: Continuous query performance analysis

### Scaling Considerations
- **Horizontal Scaling**: Read replicas for query distribution
- **Vertical Scaling**: Memory optimization for vector operations
- **Archiving**: Historical data archiving strategy
- **Partitioning**: Table partitioning for large datasets

## Configuration

### Environment Variables
```bash
# Database Configuration
SUPABASE_URL=your_supabase_url
SUPABASE_ANON_KEY=your_anon_key
SUPABASE_SERVICE_KEY=your_service_key

# Vector Search Configuration
OPENAI_API_KEY=your_openai_key
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIMENSIONS=1536
SIMILARITY_THRESHOLD=0.7

# Performance Tuning
MAX_BATCH_SIZE=100
QUERY_TIMEOUT=30
CACHE_TTL=3600
```

## Migration Commands

```bash
# Apply all migrations
supabase db push

# Reset database (development only)
supabase db reset

# Create new migration
supabase migration new migration_name

# View migration status
supabase migration list
```

---

*Last Updated: December 2024*
*Version: 1.0*
*Project: RAG-Tobi Salesperson Copilot* 