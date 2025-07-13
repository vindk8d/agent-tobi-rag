-- SQL queries to audit and inspect text content in embeddings

-- 1. Basic query to see all documents with their embeddings
SELECT 
    d.id as document_id,
    d.title,
    d.content,
    d.document_type,
    d.word_count,
    d.character_count,
    d.created_at,
    ds.name as source_name,
    ds.url as source_url,
    e.id as embedding_id,
    e.model_name,
    e.created_at as embedding_created_at
FROM documents d
LEFT JOIN embeddings e ON d.id = e.document_id
LEFT JOIN data_sources ds ON d.data_source_id = ds.id
ORDER BY d.created_at DESC;

-- 2. Count of documents and embeddings per data source
SELECT 
    ds.name as source_name,
    ds.source_type,
    ds.url,
    COUNT(DISTINCT d.id) as document_count,
    COUNT(e.id) as embedding_count,
    SUM(d.word_count) as total_words,
    SUM(d.character_count) as total_characters
FROM data_sources ds
LEFT JOIN documents d ON ds.id = d.data_source_id
LEFT JOIN embeddings e ON d.id = e.document_id
GROUP BY ds.id, ds.name, ds.source_type, ds.url
ORDER BY document_count DESC;

-- 3. Recent scraping activity
SELECT 
    ds.name as source_name,
    ds.source_type,
    ds.url,
    d.title,
    LEFT(d.content, 200) as content_preview,
    d.document_type,
    d.status,
    d.created_at,
    d.word_count
FROM documents d
JOIN data_sources ds ON d.data_source_id = ds.id
WHERE d.created_at >= NOW() - INTERVAL '7 days'
ORDER BY d.created_at DESC;

-- 4. Documents with missing embeddings
SELECT 
    d.id,
    d.title,
    LEFT(d.content, 150) as content_preview,
    d.document_type,
    d.status,
    ds.name as source_name
FROM documents d
LEFT JOIN embeddings e ON d.id = e.document_id
LEFT JOIN data_sources ds ON d.data_source_id = ds.id
WHERE e.document_id IS NULL
ORDER BY d.created_at DESC;

-- 5. Full text search across all documents
SELECT 
    d.id,
    d.title,
    LEFT(d.content, 200) as content_preview,
    ts_rank_cd(to_tsvector('english', d.content), plainto_tsquery('english', 'SEARCH_TERM_HERE')) as rank,
    ds.name as source_name
FROM documents d
JOIN data_sources ds ON d.data_source_id = ds.id
WHERE to_tsvector('english', d.content) @@ plainto_tsquery('english', 'SEARCH_TERM_HERE')
ORDER BY rank DESC;

-- 6. Detailed view of specific document content
SELECT 
    d.id,
    d.title,
    d.content,
    d.document_type,
    d.word_count,
    d.character_count,
    d.chunk_index,
    d.metadata,
    ds.name as source_name,
    ds.url as source_url,
    ds.source_type,
    e.model_name,
    array_length(e.embedding, 1) as embedding_dimensions
FROM documents d
LEFT JOIN data_sources ds ON d.data_source_id = ds.id
LEFT JOIN embeddings e ON d.id = e.document_id
WHERE d.id = 'YOUR_DOCUMENT_ID_HERE'; 