# Spider Website Crawling Integration Guide

This guide explains how to use the new Spider-based website crawling functionality to discover and scrape child links and related sites.

## 🎯 What's New

The Spider integration adds powerful website crawling capabilities to your RAG system:

- **Automatic Child Link Discovery**: Finds and follows related pages automatically
- **Configurable Crawling**: Control depth, page limits, and URL patterns
- **Respectful Crawling**: Built-in delays and robots.txt compliance
- **Full RAG Integration**: Processes all discovered content into your knowledge base
- **API Endpoint**: Easy-to-use REST API for starting crawls

## 🚀 Key Features

### Before Spider Integration
- ✅ Scrape single web pages
- ❌ Manual URL management
- ❌ Limited content discovery

### After Spider Integration
- ✅ Crawl entire websites automatically
- ✅ Discover child links and related content
- ✅ Filter URLs by patterns
- ✅ Respect rate limits and robots.txt
- ✅ Process all content into RAG system

## 🛠️ Setup

### 1. Install Dependencies

The Spider client is already added to your requirements.txt:

```bash
cd backend
pip3 install -r requirements.txt
```

### 2. Get Spider API Key

Spider is a professional web crawling service. To use it:

1. Visit [Spider.cloud](https://spider.cloud)
2. Sign up for an account
3. Get your API key from the dashboard
4. Add it to your environment variables:

```bash
export SPIDER_API_KEY="your_api_key_here"
```

Or add it to your `.env` file:

```
SPIDER_API_KEY=your_api_key_here
```

### 3. Alternative: Use Without API Key

If you prefer not to use the Spider service, you can modify the crawler to use alternative methods or implement your own link discovery logic.

## 📡 API Usage

### Crawl Website Endpoint

**POST** `/datasources/crawl`

Crawls an entire website starting from the base URL and discovers all child links.

#### Request Body

```json
{
  "url": "https://python.langchain.com/docs/integrations/document_loaders/",
  "data_source_name": "LangChain Document Loaders",
  "max_pages": 25,
  "max_depth": 3,
  "delay": 1.0,
  "include_patterns": ["/docs/integrations/document_loaders/"],
  "exclude_patterns": [".pdf", ".zip", "/admin/", "/api/"]
}
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `url` | string | required | Base URL to start crawling from |
| `data_source_name` | string | optional | Name for the data source (defaults to domain name) |
| `max_pages` | integer | 50 | Maximum number of pages to crawl (1-200) |
| `max_depth` | integer | 3 | Maximum crawl depth (1-10) |
| `delay` | float | 1.0 | Delay between requests in seconds (0.1-10.0) |
| `include_patterns` | array | optional | URL patterns to include (e.g., ["/docs/", "/api/"]) |
| `exclude_patterns` | array | optional | URL patterns to exclude (e.g., [".pdf", "/admin/"]) |

#### Response

```json
{
  "success": true,
  "data": {
    "data_source_id": "12345678-1234-1234-1234-123456789012",
    "name": "LangChain Document Loaders",
    "url": "https://python.langchain.com/docs/integrations/document_loaders/",
    "status": "active",
    "crawl_config": {
      "max_pages": 25,
      "max_depth": 3,
      "delay": 1.0,
      "include_patterns": ["/docs/integrations/document_loaders/"],
      "exclude_patterns": [".pdf", ".zip", "/admin/", "/api/"]
    },
    "message": "Website crawl started successfully. This may take several minutes..."
  }
}
```

### Example cURL Command

```bash
curl -X POST http://localhost:8000/datasources/crawl \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://python.langchain.com/docs/integrations/document_loaders/",
    "data_source_name": "LangChain Document Loaders",
    "max_pages": 25,
    "max_depth": 3,
    "delay": 1.0,
    "include_patterns": ["/docs/integrations/document_loaders/"],
    "exclude_patterns": [".pdf", ".zip", "/admin/", "/api/"]
  }'
```

## 🔧 Configuration Examples

### Documentation Site Crawling

Perfect for crawling documentation sites like LangChain docs:

```json
{
  "url": "https://python.langchain.com/docs/integrations/document_loaders/",
  "data_source_name": "LangChain Document Loaders",
  "max_pages": 50,
  "max_depth": 4,
  "delay": 1.0,
  "include_patterns": ["/docs/integrations/document_loaders/"],
  "exclude_patterns": [".pdf", ".zip", "/admin/", "/search/"]
}
```

### Product Catalog Crawling

For e-commerce or product websites:

```json
{
  "url": "https://example-store.com/products/",
  "data_source_name": "Product Catalog",
  "max_pages": 100,
  "max_depth": 3,
  "delay": 2.0,
  "include_patterns": ["/products/", "/categories/"],
  "exclude_patterns": ["/cart/", "/checkout/", "/admin/", ".jpg", ".png"]
}
```

### Blog Archive Crawling

For blog sites and news websites:

```json
{
  "url": "https://example-blog.com/",
  "data_source_name": "Blog Archive",
  "max_pages": 75,
  "max_depth": 2,
  "delay": 1.5,
  "include_patterns": ["/posts/", "/articles/", "/blog/"],
  "exclude_patterns": ["/admin/", "/login/", "/comments/"]
}
```

## 🎯 Best Practices

### 1. URL Pattern Filtering

Use include/exclude patterns to focus on relevant content:

```json
{
  "include_patterns": [
    "/docs/",
    "/tutorials/",
    "/guides/"
  ],
  "exclude_patterns": [
    "/admin/",
    "/login/",
    "/search/",
    ".pdf",
    ".zip",
    ".jpg",
    ".png"
  ]
}
```

### 2. Respectful Crawling

- Use appropriate delays (1-2 seconds minimum)
- Limit page count for initial tests
- Respect robots.txt (enabled by default)
- Start with smaller max_depth values

### 3. Monitor Progress

Track crawl progress using the data source endpoints:

```bash
# Check status
curl http://localhost:8000/datasources/{data_source_id}

# View statistics
curl http://localhost:8000/datasources/stats/overview
```

## 🔍 Monitoring and Debugging

### Check Crawl Status

After starting a crawl, monitor its progress:

```bash
# Get data source details
GET /datasources/{data_source_id}

# Check recent activity
GET /datasources/stats/overview
```

### View Processed Content

Use your existing audit tools to inspect the crawled content:

```bash
# Run the audit script
python3 scripts/audit_embeddings.py

# Check specific data source
python3 scripts/audit_embeddings.py --data-source-id {data_source_id}
```

## 🚨 Troubleshooting

### Common Issues

1. **Spider API Key Error**
   ```
   Error: SPIDER_API_KEY environment variable not set
   ```
   **Solution**: Set the SPIDER_API_KEY environment variable

2. **Rate Limiting**
   ```
   Error: Too many requests
   ```
   **Solution**: Increase the delay parameter or reduce max_pages

3. **No Content Found**
   ```
   Warning: No documents found during crawl
   ```
   **Solution**: Check include/exclude patterns and verify URL accessibility

### Debug Mode

Enable debug logging to see detailed crawl information:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📊 Performance Considerations

### Crawl Speed vs. Respect

| Delay (seconds) | Speed | Respectfulness | Recommended For |
|----------------|--------|----------------|-----------------|
| 0.1-0.5 | Fast | Low | Internal/test sites |
| 1.0-2.0 | Medium | Medium | Most websites |
| 2.0+ | Slow | High | Large/busy sites |

### Resource Usage

- **Memory**: ~1-2MB per crawled page
- **Storage**: ~10-50KB per processed chunk
- **Time**: ~1-3 seconds per page (including processing)

## 🎉 Use Cases

### 1. Documentation Indexing
- Crawl entire documentation sites
- Keep knowledge base updated with latest docs
- Enable semantic search across all documentation

### 2. Competitive Intelligence
- Monitor competitor websites
- Track product changes and updates
- Analyze market trends

### 3. Knowledge Management
- Index corporate websites
- Build comprehensive knowledge bases
- Enable AI-powered search across all content

### 4. Content Migration
- Extract content from legacy websites
- Migrate to new systems
- Preserve institutional knowledge

## 🔮 Future Enhancements

Planned improvements for the Spider integration:

- **Scheduled Crawling**: Automatic periodic re-crawling
- **Content Change Detection**: Only update changed content
- **Advanced Filtering**: CSS selector-based content extraction
- **Batch Processing**: Process multiple URLs simultaneously
- **Custom Crawling Rules**: Site-specific crawling configurations

## 📚 Additional Resources

- [Spider Documentation](https://spider.cloud/docs)
- [LangChain Spider Integration](https://python.langchain.com/docs/integrations/document_loaders/spider)
- [Web Crawling Best Practices](https://developers.google.com/search/docs/crawling-indexing/overview)

## 🤝 Support

If you encounter issues or have questions:

1. Check the troubleshooting section above
2. Review the Spider API documentation
3. Verify your API key and rate limits
4. Check the logs for detailed error messages

The Spider integration makes it easy to crawl entire websites and discover all related content automatically. Start with small tests and gradually increase the scope as needed! 