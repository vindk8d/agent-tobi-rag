#!/usr/bin/env python3
"""
Test script for the new Spider-based website crawling functionality.
This demonstrates how to crawl entire websites and discover child links.
"""

import asyncio
import os
import sys
import json
from datetime import datetime
from typing import Dict, Any

# Add the backend directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from dotenv import load_dotenv
from rag.pipeline import DocumentProcessingPipeline
from scrapers.spider_crawler import SpiderCrawler

# Load environment variables
load_dotenv()

async def test_spider_crawler():
    """Test the Spider crawler directly"""
    print("🕷️  Testing Spider Crawler")
    print("=" * 50)
    
    # Initialize crawler with conservative settings for testing
    crawler = SpiderCrawler(
        max_pages=10,  # Small number for testing
        max_depth=2,   # Limited depth
        delay=1.0,     # Respectful delay
        include_patterns=['/docs/', '/integrations/'],  # Focus on documentation
        exclude_patterns=['/admin/', '.pdf', '.zip']   # Exclude certain patterns
    )
    
    # Test with the LangChain docs URL
    test_url = "https://python.langchain.com/docs/integrations/document_loaders/"
    
    print(f"📍 Testing URL: {test_url}")
    print(f"⚙️  Settings: max_pages={crawler.max_pages}, max_depth={crawler.max_depth}")
    print(f"🔍 Include patterns: {crawler.include_patterns}")
    print(f"❌ Exclude patterns: {crawler.exclude_patterns}")
    print()
    
    # Test crawling
    try:
        result = crawler.crawl_website(test_url, "LangChain Document Loaders Test")
        
        if result['success']:
            print("✅ Crawl successful!")
            print(f"📄 Pages found: {result.get('total_pages', 0)}")
            print(f"⏱️  Crawl time: {result.get('crawl_time', 0):.2f}s")
            print(f"🔗 URLs discovered: {result.get('discovered_urls', 0)}")
            print(f"📊 Pages crawled: {result.get('crawled_urls', 0)}")
            print()
            
            # Show sample of discovered content
            documents = result.get('documents', [])
            if documents:
                print("📝 Sample content from first page:")
                first_doc = documents[0]
                content_preview = first_doc.page_content[:200] + "..." if len(first_doc.page_content) > 200 else first_doc.page_content
                print(f"   Source: {first_doc.metadata.get('source', 'Unknown')}")
                print(f"   Content: {content_preview}")
                print()
            
            return result
        else:
            print(f"❌ Crawl failed: {result.get('error', 'Unknown error')}")
            return None
            
    except Exception as e:
        print(f"❌ Error during crawl: {e}")
        return None

async def test_pipeline_integration():
    """Test the pipeline integration with Spider crawling"""
    print("🔄 Testing Pipeline Integration")
    print("=" * 50)
    
    # Initialize pipeline
    pipeline = DocumentProcessingPipeline()
    
    # Test with a smaller, focused website
    test_url = "https://python.langchain.com/docs/integrations/document_loaders/"
    
    print(f"📍 Testing URL: {test_url}")
    print("🚀 Starting website processing...")
    print()
    
    try:
        result = await pipeline.process_website(
            base_url=test_url,
            data_source_name="LangChain Docs Test",
            max_pages=5,  # Small number for testing
            max_depth=2,
            delay=1.0,
            include_patterns=['/docs/'],
            exclude_patterns=['.pdf', '.zip', '/admin/']
        )
        
        if result['success']:
            print("✅ Pipeline processing successful!")
            print(f"📊 Summary:")
            print(f"   Data source ID: {result.get('data_source_id', 'N/A')}")
            print(f"   Pages processed: {result.get('processed_pages', 0)}")
            print(f"   Total pages found: {result.get('total_pages_found', 0)}")
            print(f"   Total chunks created: {result.get('total_chunks', 0)}")
            print(f"   Stored chunk IDs: {len(result.get('stored_chunk_ids', []))}")
            print(f"   Embedding IDs: {len(result.get('embedding_ids', []))}")
            print(f"   Crawl time: {result.get('crawl_time', 0):.2f}s")
            print(f"   URLs discovered: {result.get('discovered_urls', 0)}")
            print()
            
            # Show sample chunk IDs
            chunk_ids = result.get('stored_chunk_ids', [])
            if chunk_ids:
                print(f"📦 Sample chunk IDs: {chunk_ids[:3]}...")
            
            return result
        else:
            print(f"❌ Pipeline processing failed: {result.get('error', 'Unknown error')}")
            return None
            
    except Exception as e:
        print(f"❌ Error during pipeline processing: {e}")
        return None

async def demo_api_usage():
    """Demonstrate how to use the new API endpoint"""
    print("📡 API Usage Example")
    print("=" * 50)
    
    # Example request payload
    api_request = {
        "url": "https://python.langchain.com/docs/integrations/document_loaders/",
        "data_source_name": "LangChain Document Loaders",
        "max_pages": 25,
        "max_depth": 3,
        "delay": 1.0,
        "include_patterns": ["/docs/integrations/document_loaders/"],
        "exclude_patterns": [".pdf", ".zip", "/admin/", "/api/"]
    }
    
    print("📝 Example API request to POST /datasources/crawl:")
    print(json.dumps(api_request, indent=2))
    print()
    
    print("🔄 This would:")
    print("   1. Create a new data source")
    print("   2. Start crawling the website in the background")
    print("   3. Discover child links and related pages")
    print("   4. Process all discovered content")
    print("   5. Store chunks and embeddings in the database")
    print("   6. Return immediately with a data source ID")
    print()
    
    print("✅ Expected response:")
    expected_response = {
        "success": True,
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
    print(json.dumps(expected_response, indent=2))

def print_summary():
    """Print a summary of the new functionality"""
    print("\n🎉 Spider Integration Summary")
    print("=" * 50)
    
    print("✨ New Features:")
    print("   • Spider-based website crawling")
    print("   • Automatic child link discovery")
    print("   • Configurable crawl depth and page limits")
    print("   • URL pattern filtering (include/exclude)")
    print("   • Respectful crawling with delays")
    print("   • Integration with existing RAG pipeline")
    print("   • New API endpoint: POST /datasources/crawl")
    print()
    
    print("🔧 Configuration Options:")
    print("   • max_pages: Maximum number of pages to crawl (1-200)")
    print("   • max_depth: Maximum crawl depth (1-10)")
    print("   • delay: Delay between requests (0.1-10.0 seconds)")
    print("   • include_patterns: URL patterns to include")
    print("   • exclude_patterns: URL patterns to exclude")
    print()
    
    print("🚀 Usage Examples:")
    print("   • Crawl documentation sites")
    print("   • Index entire product catalogs")
    print("   • Process blog archives")
    print("   • Build knowledge bases from websites")
    print()
    
    print("📚 Next Steps:")
    print("   1. Install dependencies: pip install -r requirements.txt")
    print("   2. Use the new API endpoint to crawl websites")
    print("   3. Monitor progress in the data sources dashboard")
    print("   4. Query your RAG system with the new content")

async def main():
    """Main test function"""
    print("🕷️  SPIDER CRAWLER INTEGRATION TEST")
    print("=" * 60)
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Test 1: Spider crawler directly
    crawler_result = await test_spider_crawler()
    
    if crawler_result:
        print("\n" + "=" * 60)
        
        # Test 2: Pipeline integration
        pipeline_result = await test_pipeline_integration()
        
        if pipeline_result:
            print("\n" + "=" * 60)
            
            # Test 3: API usage demo
            await demo_api_usage()
    
    # Print summary
    print_summary()
    
    print(f"\n⏰ Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    asyncio.run(main()) 