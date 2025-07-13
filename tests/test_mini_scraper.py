#!/usr/bin/env python3
"""
Test script to scrape Mini Philippines website and generate downloadable output.
"""

import asyncio
import os
import sys
import json
from datetime import datetime
from typing import Dict, Any, List
import csv

# Add the backend directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from dotenv import load_dotenv
from scrapers.spider_crawler import SpiderCrawler
from bs4 import BeautifulSoup
import re

# Load environment variables
load_dotenv()

def extract_meaningful_content(html_content: str, url: str = "") -> Dict[str, Any]:
    """
    Extract meaningful text content from HTML for salesperson copilot.
    Focuses on product information, specifications, pricing, and sales-relevant content.
    """
    if not html_content:
        return {"cleaned_content": "", "title": "", "content_type": "empty"}
    
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Extract title
        title = ""
        title_tag = soup.find('title')
        if title_tag:
            title = title_tag.get_text(strip=True)
        
        # Remove unwanted elements (more selective)
        for element in soup(['script', 'style', 'noscript', 'iframe', 'object', 'embed']):
            element.decompose()
        
        # Remove elements with common non-content class/id patterns (more selective)
        unwanted_patterns = [
            'cookie', 'popup', 'modal', 'overlay', 'advertisement', 'ad'
        ]
        
        for pattern in unwanted_patterns:
            for element in soup.find_all(attrs={'class': re.compile(pattern, re.I)}):
                element.decompose()
            for element in soup.find_all(attrs={'id': re.compile(pattern, re.I)}):
                element.decompose()
        
        # Priority content selectors for automotive/sales content
        priority_selectors = [
            # Product information
            '[class*="product"]', '[class*="model"]', '[class*="vehicle"]',
            '[class*="car"]', '[class*="spec"]', '[class*="feature"]',
            # Pricing and offers
            '[class*="price"]', '[class*="cost"]', '[class*="offer"]',
            '[class*="promo"]', '[class*="deal"]', '[class*="finance"]',
            # Content areas
            '[class*="content"]', '[class*="main"]', '[class*="article"]',
            '[class*="description"]', '[class*="details"]', '[class*="info"]',
            # Common semantic tags
            'main', 'article', 'section', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
            'p', 'ul', 'ol', 'li', 'table', 'tr', 'td', 'th'
        ]
        
        # Extract content from priority selectors
        meaningful_content = []
        for selector in priority_selectors:
            elements = soup.select(selector)
            for element in elements:
                text = element.get_text(separator=' ', strip=True)
                if text and len(text) > 10:  # Lowered threshold for more content
                    meaningful_content.append(text)
        
        # If no priority content found, get all text but clean it up
        if not meaningful_content:
            all_text = soup.get_text(separator=' ', strip=True)
            meaningful_content = [all_text]
        
        # For JavaScript-heavy sites, also try to extract any text that might be useful
        # Look for any text in divs, spans, or other elements that might contain product info
        additional_selectors = ['div', 'span', 'p', 'td', 'th', 'li']
        for selector in additional_selectors:
            elements = soup.find_all(selector)
            for element in elements:
                text = element.get_text(separator=' ', strip=True)
                # Look for automotive-related keywords
                if text and len(text) > 5 and any(keyword in text.lower() for keyword in 
                    ['mini', 'cooper', 'countryman', 'price', 'php', '₱', 'engine', 'fuel', 'model', 'year', 'spec']):
                    meaningful_content.append(text)
        
        # Combine and clean the content
        combined_content = ' '.join(meaningful_content)
        
        # Clean up the text
        combined_content = re.sub(r'\s+', ' ', combined_content)  # Multiple spaces to single
        combined_content = re.sub(r'\n\s*\n', '\n', combined_content)  # Multiple newlines
        combined_content = combined_content.strip()
        
        # Determine content type based on keywords
        content_type = "general"
        automotive_keywords = ['mini', 'car', 'vehicle', 'model', 'engine', 'price', 
                              'specification', 'feature', 'dealer', 'finance', 'lease']
        if any(keyword in combined_content.lower() for keyword in automotive_keywords):
            content_type = "automotive"
        
        # Extract key information snippets
        key_info = extract_key_sales_info(combined_content)
        
        return {
            "cleaned_content": combined_content,
            "title": title,
            "content_type": content_type,
            "character_count": len(combined_content),
            "key_info": key_info,
            "url": url
        }
        
    except Exception as e:
        return {
            "cleaned_content": f"Error processing content: {str(e)}",
            "title": "",
            "content_type": "error",
            "character_count": 0,
            "key_info": {},
            "url": url
        }

def extract_key_sales_info(content: str) -> Dict[str, Any]:
    """Extract key sales information from content using pattern matching."""
    key_info = {}
    
    # Price patterns
    price_patterns = [
        r'₱[\d,]+(?:\.\d{2})?',  # Philippine Peso
        r'\$[\d,]+(?:\.\d{2})?',  # US Dollar
        r'price[:\s]*[\d,]+',     # Price: 1,000,000
        r'starting at[:\s]*[\d,]+', # Starting at 1,000,000
    ]
    
    prices = []
    for pattern in price_patterns:
        matches = re.findall(pattern, content, re.IGNORECASE)
        prices.extend(matches)
    
    if prices:
        key_info['prices'] = list(set(prices))
    
    # Model/product names
    model_patterns = [
        r'mini\s+\w+',  # Mini Cooper, Mini Countryman, etc.
        r'model\s+\w+',  # Model names
        r'new\s+\w+\s+\w+',  # New model releases
    ]
    
    models = []
    for pattern in model_patterns:
        matches = re.findall(pattern, content, re.IGNORECASE)
        models.extend(matches)
    
    if models:
        key_info['models'] = list(set(models))
    
    # Features and specifications
    if 'engine' in content.lower():
        key_info['has_engine_info'] = True
    if 'fuel' in content.lower():
        key_info['has_fuel_info'] = True
    if 'safety' in content.lower():
        key_info['has_safety_info'] = True
    if any(word in content.lower() for word in ['finance', 'loan', 'lease']):
        key_info['has_finance_info'] = True
    
    return key_info

def save_to_json(data: Dict[str, Any], filename: str):
    """Save data to JSON file"""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"💾 Saved to {filename}")

def save_to_csv(documents: List[Dict], filename: str):
    """Save documents to CSV file"""
    if not documents:
        print("❌ No documents to save to CSV")
        return
    
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # Header
        writer.writerow(['URL', 'Title', 'Content_Type', 'Original_Length', 'Cleaned_Length', 'Key_Info', 'Content_Preview'])
        
        # Data rows
        for doc in documents:
            url = doc.get('metadata', {}).get('source', 'Unknown')
            title = doc.get('metadata', {}).get('extracted_title', 'No Title')
            content_type = doc.get('metadata', {}).get('content_type', 'unknown')
            original_length = len(doc.get('page_content', ''))
            cleaned_content = doc.get('cleaned_content', '')
            cleaned_length = len(cleaned_content)
            key_info = str(doc.get('metadata', {}).get('key_info', {}))
            content_preview = cleaned_content[:200] + "..." if len(cleaned_content) > 200 else cleaned_content
            
            writer.writerow([url, title, content_type, original_length, cleaned_length, key_info, content_preview])
    
    print(f"📊 Saved to {filename}")

def save_full_content(documents: List[Dict], filename: str):
    """Save full content to text file"""
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("MINI PHILIPPINES WEBSITE CRAWL RESULTS\n")
        f.write("=" * 50 + "\n")
        f.write(f"Crawled at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total documents: {len(documents)}\n\n")
        
        for i, doc in enumerate(documents, 1):
            f.write(f"\n--- DOCUMENT {i} ---\n")
            f.write(f"URL: {doc.get('metadata', {}).get('source', 'Unknown')}\n")
            f.write(f"Title: {doc.get('metadata', {}).get('extracted_title', 'No Title')}\n")
            f.write(f"Content Type: {doc.get('metadata', {}).get('content_type', 'unknown')}\n")
            f.write(f"Original Length: {len(doc.get('page_content', ''))}\n")
            f.write(f"Cleaned Length: {len(doc.get('cleaned_content', ''))}\n")
            f.write(f"Key Info: {doc.get('metadata', {}).get('key_info', {})}\n")
            f.write(f"Crawl Index: {doc.get('metadata', {}).get('crawl_index', 'N/A')}\n")
            f.write("\n🧹 CLEANED CONTENT (For Copilot):\n")
            f.write("-" * 40 + "\n")
            f.write(doc.get('cleaned_content', ''))
            f.write("\n\n📄 ORIGINAL HTML (Reference):\n")
            f.write("-" * 40 + "\n")
            f.write(doc.get('page_content', '')[:1000] + "..." if len(doc.get('page_content', '')) > 1000 else doc.get('page_content', ''))
            f.write("\n" + "=" * 50 + "\n")
    
    print(f"📄 Full content saved to {filename}")

async def test_mini_scraper():
    """Test scraping Mini Philippines website"""
    print("🏎️  MINI PHILIPPINES WEBSITE SCRAPER TEST")
    print("=" * 60)
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Check if API key is available
    api_key = os.getenv('SPIDER_API_KEY')
    if not api_key:
        print("❌ SPIDER_API_KEY not found in environment variables")
        print("💡 Make sure you've added it to your .env file")
        return
    
    print(f"✅ Spider API key found: {api_key[:10]}...")
    print()
    
    # Configuration for Mini Philippines website
    target_url = "https://www.mini.com.ph/en_PH/home.html"
    
    print(f"🎯 Target URL: {target_url}")
    print()
    
    # Initialize crawler with settings optimized for Mini website
    crawler = SpiderCrawler(
        max_pages=50,  # Increased to capture more diverse content
        max_depth=3,   # Good depth for product/model pages
        delay=1.5,     # Respectful delay
        include_patterns=[],  # No include patterns - let it discover all relevant content
        exclude_patterns=[
            '/admin/', '/login/', '/search/', '/cart/', '/checkout/',
            '.jpg', '.jpeg', '.png', '.gif', '.pdf', '.zip', '.css', '.js',
            '/media/', '/assets/', '/static/', '/images/', '/css/', '/js/',
            '/wp-content/', '/wp-admin/', '/api/', '/ajax/', '/form/',
            '/newsletter/', '/contact/', '/careers/', '/legal/', '/privacy/',
            '/cookies/', '/terms/', '/sitemap'
        ]
    )
    
    print("⚙️  Crawler Configuration:")
    print(f"   Max pages: {crawler.max_pages}")
    print(f"   Max depth: {crawler.max_depth}")
    print(f"   Delay: {crawler.delay}s")
    print(f"   Include patterns: {crawler.include_patterns}")
    print(f"   Exclude patterns: {crawler.exclude_patterns}")
    print()
    
    # Start crawling
    print("🚀 Starting crawl...")
    try:
        result = crawler.crawl_website(target_url, "Mini Philippines")
        
        if result['success']:
            print("✅ Crawl completed successfully!")
            print()
            
            # Display results summary
            print("📊 CRAWL RESULTS SUMMARY:")
            print(f"   Total pages found: {result.get('total_pages', 0)}")
            print(f"   Crawl time: {result.get('crawl_time', 0):.2f}s")
            print(f"   URLs discovered: {result.get('discovered_urls', 0)}")
            print(f"   Pages crawled: {result.get('crawled_urls', 0)}")
            print()
            
            # Process documents
            documents = result.get('documents', [])
            if documents:
                print(f"📄 Processing {len(documents)} documents...")
                
                # Convert documents to serializable format and extract meaningful content
                doc_data = []
                for doc in documents:
                    # Extract meaningful content from HTML
                    extracted = extract_meaningful_content(doc.page_content, doc.metadata.get('source', ''))
                    
                    doc_dict = {
                        'page_content': doc.page_content,  # Keep original for reference
                        'cleaned_content': extracted['cleaned_content'],  # Extracted meaningful text
                        'metadata': {
                            **doc.metadata,
                            'extracted_title': extracted['title'],
                            'content_type': extracted['content_type'],
                            'cleaned_char_count': extracted['character_count'],
                            'key_info': extracted['key_info']
                        }
                    }
                    doc_data.append(doc_dict)
                
                # Create output directory
                output_dir = "mini_crawl_output"
                os.makedirs(output_dir, exist_ok=True)
                
                # Generate timestamp for files
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                # Save results in multiple formats
                print("\n💾 Saving results...")
                
                # 1. Complete JSON output (excluding non-serializable documents)
                json_result = result.copy()
                json_result['documents'] = doc_data  # Use serializable version
                json_filename = f"{output_dir}/mini_crawl_complete_{timestamp}.json"
                save_to_json(json_result, json_filename)
                
                # 2. Documents only JSON
                docs_json_filename = f"{output_dir}/mini_crawl_documents_{timestamp}.json"
                save_to_json(doc_data, docs_json_filename)
                
                # 3. CSV summary
                csv_filename = f"{output_dir}/mini_crawl_summary_{timestamp}.csv"
                save_to_csv(doc_data, csv_filename)
                
                # 4. Full content text file
                txt_filename = f"{output_dir}/mini_crawl_full_content_{timestamp}.txt"
                save_full_content(doc_data, txt_filename)
                
                # 5. Analysis report
                analysis_filename = f"{output_dir}/mini_crawl_analysis_{timestamp}.txt"
                generate_analysis_report(doc_data, result, analysis_filename)
                
                print(f"\n📁 All files saved to: {output_dir}/")
                print("🎉 Download these files to evaluate the crawl results!")
                
                # Display sample content
                print("\n📝 SAMPLE CLEANED CONTENT PREVIEW:")
                print("-" * 40)
                for i, doc_data_item in enumerate(doc_data[:3]):  # Show first 3 documents
                    print(f"\n{i+1}. {doc_data_item['metadata'].get('source', 'Unknown URL')}")
                    print(f"   Title: {doc_data_item['metadata'].get('extracted_title', 'No title')}")
                    print(f"   Content Type: {doc_data_item['metadata'].get('content_type', 'unknown')}")
                    print(f"   Key Info: {doc_data_item['metadata'].get('key_info', {})}")
                    cleaned_content = doc_data_item.get('cleaned_content', '')
                    content_preview = cleaned_content[:200] + "..." if len(cleaned_content) > 200 else cleaned_content
                    print(f"   Cleaned Content: {content_preview}")
                
                if len(doc_data) > 3:
                    print(f"\n... and {len(doc_data) - 3} more documents")
                
                return True
            else:
                print("❌ No documents found in crawl results")
                return False
        else:
            print(f"❌ Crawl failed: {result.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"❌ Error during crawl: {e}")
        return False

def generate_analysis_report(documents: List[Dict], result: Dict, filename: str):
    """Generate analysis report of crawl results"""
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("MINI PHILIPPINES WEBSITE CRAWL ANALYSIS\n")
        f.write("=" * 50 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Overall statistics
        f.write("OVERALL STATISTICS:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Total pages crawled: {len(documents)}\n")
        f.write(f"Total original characters: {sum(len(doc.get('page_content', '')) for doc in documents)}\n")
        f.write(f"Total cleaned characters: {sum(len(doc.get('cleaned_content', '')) for doc in documents)}\n")
        f.write(f"Average cleaned content length: {sum(len(doc.get('cleaned_content', '')) for doc in documents) // len(documents) if documents else 0}\n")
        f.write(f"Content compression ratio: {(sum(len(doc.get('cleaned_content', '')) for doc in documents) / max(sum(len(doc.get('page_content', '')) for doc in documents), 1)) * 100:.1f}%\n")
        f.write(f"Crawl time: {result.get('crawl_time', 0):.2f} seconds\n")
        f.write(f"URLs discovered: {result.get('discovered_urls', 0)}\n")
        f.write(f"Success rate: {len(documents) / result.get('discovered_urls', 1) * 100:.1f}%\n\n")
        
        # URL analysis
        f.write("URL ANALYSIS:\n")
        f.write("-" * 15 + "\n")
        urls = [doc.get('metadata', {}).get('source', 'Unknown') for doc in documents]
        unique_domains = set()
        for url in urls:
            if url != 'Unknown':
                try:
                    from urllib.parse import urlparse
                    domain = urlparse(url).netloc
                    unique_domains.add(domain)
                except:
                    pass
        
        f.write(f"Unique domains: {len(unique_domains)}\n")
        f.write(f"Domains: {', '.join(unique_domains)}\n\n")
        
        # Content analysis
        f.write("CONTENT ANALYSIS:\n")
        f.write("-" * 18 + "\n")
        cleaned_lengths = [len(doc.get('cleaned_content', '')) for doc in documents]
        if cleaned_lengths:
            f.write(f"Shortest cleaned content: {min(cleaned_lengths)} characters\n")
            f.write(f"Longest cleaned content: {max(cleaned_lengths)} characters\n")
            f.write(f"Median cleaned content: {sorted(cleaned_lengths)[len(cleaned_lengths)//2]} characters\n")
        
        # Content type analysis
        f.write("\nCONTENT TYPE ANALYSIS:\n")
        f.write("-" * 22 + "\n")
        content_types = {}
        automotive_pages = 0
        for doc in documents:
            content_type = doc.get('metadata', {}).get('content_type', 'unknown')
            content_types[content_type] = content_types.get(content_type, 0) + 1
            if content_type == 'automotive':
                automotive_pages += 1
        
        for content_type, count in content_types.items():
            f.write(f"{content_type}: {count} pages\n")
        f.write(f"Automotive relevance: {automotive_pages}/{len(documents)} pages\n")
        
        # Key sales information found
        f.write("\nSALES INFORMATION FOUND:\n")
        f.write("-" * 25 + "\n")
        pages_with_prices = sum(1 for doc in documents if doc.get('metadata', {}).get('key_info', {}).get('prices'))
        pages_with_models = sum(1 for doc in documents if doc.get('metadata', {}).get('key_info', {}).get('models'))
        pages_with_engine = sum(1 for doc in documents if doc.get('metadata', {}).get('key_info', {}).get('has_engine_info'))
        pages_with_finance = sum(1 for doc in documents if doc.get('metadata', {}).get('key_info', {}).get('has_finance_info'))
        
        f.write(f"Pages with pricing info: {pages_with_prices}\n")
        f.write(f"Pages with model info: {pages_with_models}\n")
        f.write(f"Pages with engine info: {pages_with_engine}\n")
        f.write(f"Pages with finance info: {pages_with_finance}\n")
        
        # Top keywords (simple frequency analysis)
        f.write("\nTOP KEYWORDS (CLEANED CONTENT):\n")
        f.write("-" * 32 + "\n")
        all_content = ' '.join(doc.get('cleaned_content', '') for doc in documents).lower()
        words = all_content.split()
        word_freq = {}
        for word in words:
            word = word.strip('.,!?;:"()[]{}')
            if len(word) > 3:  # Only count words longer than 3 characters
                word_freq[word] = word_freq.get(word, 0) + 1
        
        top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:20]
        for word, count in top_words:
            f.write(f"{word}: {count}\n")
        
        f.write("\nPAGE DETAILS:\n")
        f.write("-" * 15 + "\n")
        for i, doc in enumerate(documents, 1):
            f.write(f"{i}. {doc.get('metadata', {}).get('source', 'Unknown')}\n")
            f.write(f"   Title: {doc.get('metadata', {}).get('extracted_title', 'No title')}\n")
            f.write(f"   Content type: {doc.get('metadata', {}).get('content_type', 'unknown')}\n")
            f.write(f"   Original length: {len(doc.get('page_content', ''))} characters\n")
            f.write(f"   Cleaned length: {len(doc.get('cleaned_content', ''))} characters\n")
            f.write(f"   Key info: {doc.get('metadata', {}).get('key_info', {})}\n")
            f.write(f"   Crawl index: {doc.get('metadata', {}).get('crawl_index', 'N/A')}\n\n")
    
    print(f"📊 Analysis report saved to {filename}")

async def main():
    """Main function"""
    success = await test_mini_scraper()
    
    if success:
        print("\n🎉 SUCCESS! Mini Philippines website crawl completed.")
        print("📥 Download the files from the 'mini_crawl_output' directory to evaluate results.")
        print("\nFile types generated:")
        print("  • Complete JSON: Full crawl results with metadata")
        print("  • Documents JSON: Just the document content")
        print("  • CSV Summary: Spreadsheet-friendly format")
        print("  • Full Content TXT: All text content in readable format")
        print("  • Analysis Report: Statistics and insights")
    else:
        print("\n❌ Crawl failed. Check the error messages above.")

if __name__ == "__main__":
    asyncio.run(main()) 