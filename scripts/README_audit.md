# RAG Content Auditing Tools

This directory contains tools to help you audit and inspect the text content stored in your RAG system embeddings.

## Overview

Your RAG system stores data in several related tables:
- **`documents`**: Contains the actual text content from scraped/uploaded sources
- **`embeddings`**: Contains vector embeddings linked to documents
- **`data_sources`**: Contains information about where the content came from

## Tools Available

### 1. SQL Queries (`audit_embeddings.sql`)

Direct SQL queries you can run in your Supabase dashboard or SQL client:

```sql
-- See all documents with their embeddings
SELECT d.title, d.content, ds.name as source_name 
FROM documents d 
LEFT JOIN embeddings e ON d.id = e.document_id 
LEFT JOIN data_sources ds ON d.data_source_id = ds.id;
```

**Key queries included:**
- Basic overview of all documents and embeddings
- Document count and statistics per data source
- Recent scraping activity
- Documents missing embeddings
- Full-text search across all content
- Detailed view of specific documents

### 2. Python Audit Script (`audit_embeddings.py`)

A comprehensive Python script for auditing your RAG content:

```bash
# Show system summary
python scripts/audit_embeddings.py --summary

# Show recent documents (last 7 days)
python scripts/audit_embeddings.py --recent 7

# Search for specific text
python scripts/audit_embeddings.py --search "sales training"

# View full content of a specific document
python scripts/audit_embeddings.py --document "uuid-here"

# Export all content to JSON file
python scripts/audit_embeddings.py --export "my_content.json"
```

**Features:**
- System overview and statistics
- Content breakdown by data source
- Recent activity monitoring
- Full-text search capabilities
- Export functionality for external analysis
- Identify orphaned documents (no embeddings)

### 3. Vector Store Methods

Added methods to `backend/rag/vector_store.py` for programmatic access:

```python
from rag.vector_store import SupabaseVectorStore

vector_store = SupabaseVectorStore()

# Get all embeddings with content
embeddings = vector_store.get_all_embeddings_with_content(limit=100)

# Get embeddings from a specific source
source_embeddings = vector_store.get_embeddings_by_source("Company Website")

# Get embedding statistics
stats = vector_store.get_embedding_stats()
```

## Usage Examples

### Quick System Overview
```bash
python scripts/audit_embeddings.py --summary
```

### Check Recent Scraping Activity
```bash
python scripts/audit_embeddings.py --recent 3
```

### Search for Specific Content
```bash
python scripts/audit_embeddings.py --search "pricing information"
```

### Export Everything for Analysis
```bash
python scripts/audit_embeddings.py --export "full_content_dump.json"
```

### Run SQL Queries in Supabase

1. Go to your Supabase dashboard
2. Navigate to the SQL Editor
3. Copy and paste queries from `audit_embeddings.sql`
4. Replace `'SEARCH_TERM_HERE'` with your search terms
5. Replace `'YOUR_DOCUMENT_ID_HERE'` with actual document IDs

## Data Structure

Understanding the relationships:

```
data_sources (websites, files, etc.)
    ↓
documents (text content, chunked)
    ↓  
embeddings (vector representations)
```

**Key fields:**
- `documents.content`: The actual text that was scraped/processed
- `documents.title`: Document title or filename
- `data_sources.name`: Human-readable source name
- `data_sources.url`: Source URL (for websites)
- `embeddings.model_name`: Which embedding model was used

## Troubleshooting

### Common Issues

1. **No content showing up**: Check if documents exist in the `documents` table
2. **Missing embeddings**: Look for documents without corresponding embeddings
3. **Empty content**: Check the scraping/upload process
4. **Database connection errors**: Verify your Supabase credentials

### Useful Queries

```sql
-- Count documents by status
SELECT status, COUNT(*) FROM documents GROUP BY status;

-- Find documents with no embeddings
SELECT d.title FROM documents d 
LEFT JOIN embeddings e ON d.id = e.document_id 
WHERE e.id IS NULL;

-- Check data source status
SELECT name, source_type, status, COUNT(documents.id) as doc_count
FROM data_sources 
LEFT JOIN documents ON data_sources.id = documents.data_source_id 
GROUP BY data_sources.id;
```

## Best Practices

1. **Regular audits**: Run the summary script weekly to monitor system health
2. **Content verification**: Spot-check scraped content for accuracy
3. **Source monitoring**: Ensure all data sources are active and updating
4. **Export backups**: Regularly export content for backup purposes
5. **Search testing**: Test searches to ensure embeddings are working correctly

## Next Steps

- Use the search functionality to test your RAG system's retrieval
- Monitor for documents missing embeddings
- Check that scraped content matches expectations
- Verify all data sources are functioning correctly 