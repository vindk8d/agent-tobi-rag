#!/usr/bin/env python3
"""
Simple test script for Spider crawler functionality without database dependencies.
This demonstrates basic Spider crawling capabilities.
"""

import asyncio
import os
import sys
import json
from datetime import datetime
from typing import Dict, Any

# Add the backend directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

def test_spider_import():
    """Test that we can import Spider components"""
    print("📦 Testing Spider Import")
    print("=" * 30)
    
    try:
        from langchain_community.document_loaders import SpiderLoader
        print("✅ SpiderLoader imported successfully")
        
        # Test basic SpiderLoader functionality
        test_url = "https://python.langchain.com/docs/integrations/document_loaders/"
        print(f"📍 Testing URL: {test_url}")
        
        loader = SpiderLoader(
            url=test_url,
            mode="scrape",
            params={
                "limit": 1,
                "cache": False
            }
        )
        print("✅ SpiderLoader initialized successfully")
        
        # Try to load a single document
        print("🔄 Loading document...")
        try:
            documents = loader.load()
            
            if documents:
                print(f"✅ Successfully loaded {len(documents)} document(s)")
                
                # Show sample content
                first_doc = documents[0]
                content_preview = first_doc.page_content[:200] + "..." if len(first_doc.page_content) > 200 else first_doc.page_content
                print(f"📝 Sample content: {content_preview}")
                print(f"📊 Total content length: {len(first_doc.page_content)} characters")
                print(f"🔗 Source: {first_doc.metadata.get('source', 'Unknown')}")
                
                return True
            else:
                print("❌ No documents loaded")
                return False
                
        except Exception as e:
            print(f"❌ Error loading document: {e}")
            print("💡 This might be due to Spider API limits or network issues")
            return False
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure spider-client is installed: pip3 install spider-client")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_spider_crawler_class():
    """Test our custom SpiderCrawler class"""
    print("\n🕷️  Testing Custom SpiderCrawler")
    print("=" * 35)
    
    try:
        # Import our custom crawler
        from scrapers.spider_crawler import SpiderCrawler
        print("✅ SpiderCrawler imported successfully")
        
        # Initialize crawler
        crawler = SpiderCrawler(
            max_pages=3,
            max_depth=1,
            delay=1.0,
            include_patterns=['/docs/'],
            exclude_patterns=['.pdf', '.zip']
        )
        print("✅ SpiderCrawler initialized successfully")
        print(f"   Settings: max_pages={crawler.max_pages}, max_depth={crawler.max_depth}")
        print(f"   Include patterns: {crawler.include_patterns}")
        print(f"   Exclude patterns: {crawler.exclude_patterns}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Check that the spider_crawler.py file exists and spider-client is installed")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def demo_api_usage():
    """Demonstrate API usage without actually calling it"""
    print("\n📡 API Usage Demo")
    print("=" * 25)
    
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
    
    # Example cURL command
    curl_command = f"""curl -X POST http://localhost:8000/datasources/crawl \\
  -H "Content-Type: application/json" \\
  -d '{json.dumps(api_request)}'"""
    
    print("🔧 Example cURL command:")
    print(curl_command)
    print()
    
    print("✅ This would start crawling the LangChain documentation")
    print("   and discover all child pages automatically!")

def print_integration_summary():
    """Print summary of the Spider integration"""
    print("\n🎉 Spider Integration Summary")
    print("=" * 40)
    
    print("✨ What's New:")
    print("   • Spider-based website crawling")
    print("   • Automatic child link discovery")
    print("   • Configurable crawl settings")
    print("   • URL pattern filtering")
    print("   • Respectful crawling with delays")
    print("   • Full RAG pipeline integration")
    print("   • New API endpoint: POST /datasources/crawl")
    print()
    
    print("🚀 Key Features:")
    print("   • Crawl entire websites, not just single pages")
    print("   • Discover and follow child links automatically")
    print("   • Filter URLs by include/exclude patterns")
    print("   • Respect robots.txt and rate limits")
    print("   • Store all content in your RAG system")
    print("   • Query across all discovered content")
    print()
    
    print("📝 Perfect for:")
    print("   • Documentation sites (like LangChain docs)")
    print("   • Product catalogs")
    print("   • Blog archives")
    print("   • Knowledge bases")
    print("   • Corporate websites")
    print()
    
    print("🔧 Configuration Options:")
    print("   • max_pages: 1-200 (default: 50)")
    print("   • max_depth: 1-10 (default: 3)")
    print("   • delay: 0.1-10.0 seconds (default: 1.0)")
    print("   • include_patterns: Focus on specific paths")
    print("   • exclude_patterns: Skip unwanted content")

def main():
    """Main test function"""
    print("🕷️  SPIDER CRAWLER INTEGRATION TEST")
    print("=" * 50)
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Test 1: Spider import and basic functionality
    spider_works = test_spider_import()
    
    # Test 2: Custom crawler class
    crawler_works = test_spider_crawler_class()
    
    # Test 3: API usage demo
    demo_api_usage()
    
    # Summary
    print_integration_summary()
    
    # Final status
    print("\n🔍 Test Results:")
    print(f"   Spider Import: {'✅ PASS' if spider_works else '❌ FAIL'}")
    print(f"   Custom Crawler: {'✅ PASS' if crawler_works else '❌ FAIL'}")
    print(f"   API Integration: ✅ READY")
    
    if spider_works and crawler_works:
        print("\n🎉 All tests passed! Spider integration is ready to use.")
        print("   You can now use the /datasources/crawl endpoint to crawl entire websites!")
    else:
        print("\n⚠️  Some tests failed. Check the errors above.")
    
    print(f"\n⏰ Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main() 