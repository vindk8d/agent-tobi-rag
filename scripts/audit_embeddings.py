#!/usr/bin/env python3
"""
Audit and fix embeddings in the database.
This script helps diagnose and repair issues with document embeddings.
"""

import os
import sys
import json
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple

# Add the backend directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

# Backend imports (after path setup)
from database import db_client
from supabase import Client


class EmbeddingsAuditor:
    def __init__(self):
        self.client: Client = db_client.client

    def get_document_summary(self) -> Dict[str, Any]:
        """Get a summary of all documents and embeddings."""
        try:
            # Get document counts
            documents_result = self.client.table("documents").select("id", count="exact").execute()
            total_documents = documents_result.count or 0

            # Get embeddings count
            embeddings_result = self.client.table("embeddings").select("id", count="exact").execute()
            total_embeddings = embeddings_result.count or 0

            # Get sources count
            sources_result = self.client.table("data_sources").select("id", count="exact").execute()
            total_sources = sources_result.count or 0

            # Get basic document stats
            docs_with_stats = self.client.table("documents").select("word_count,character_count,created_at").execute()

            total_words = sum(doc.get('word_count', 0) or 0 for doc in docs_with_stats.data)
            total_characters = sum(doc.get('character_count', 0) or 0 for doc in docs_with_stats.data)

            dates = [doc['created_at'] for doc in docs_with_stats.data if doc.get('created_at')]
            oldest_document = min(dates) if dates else None
            newest_document = max(dates) if dates else None

            return {
                'total_documents': total_documents,
                'total_embeddings': total_embeddings,
                'total_words': total_words,
                'total_characters': total_characters,
                'total_sources': total_sources,
                'oldest_document': oldest_document,
                'newest_document': newest_document
            }
        except Exception as e:
            print(f"Error getting document summary: {e}")
            return {}

    def get_documents_by_source(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get documents grouped by their data source."""
        try:
            # Get all data sources
            sources_result = self.client.table("data_sources").select("*").execute()
            sources = sources_result.data or []

            source_stats = []
            for source in sources:
                # Get documents for this source
                docs_result = self.client.table("documents").select("id,word_count,created_at").eq("data_source_id", source['id']).execute()
                documents = docs_result.data or []

                # Get embeddings for documents from this source
                doc_ids = [doc['id'] for doc in documents]
                embeddings_count = 0
                if doc_ids:
                    embeddings_result = self.client.table("embeddings").select("id", count="exact").in_("document_id", doc_ids).execute()
                    embeddings_count = embeddings_result.count or 0

                total_words = sum(doc.get('word_count', 0) or 0 for doc in documents)
                latest_document = max([doc['created_at'] for doc in documents], default=None)

                source_stats.append({
                    'source_name': source.get('name', 'Unknown'),
                    'source_type': source.get('source_type', 'Unknown'),
                    'url': source.get('url', ''),
                    'source_status': source.get('status', 'Unknown'),
                    'document_count': len(documents),
                    'embedding_count': embeddings_count,
                    'total_words': total_words,
                    'latest_document': latest_document
                })

            # Sort by document count descending
            source_stats.sort(key=lambda x: x['document_count'], reverse=True)
            return source_stats[:limit]
        except Exception as e:
            print(f"Error getting documents by source: {e}")
            return []

    def get_recent_documents(self, days: int = 7, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recently scraped documents."""
        try:
            from datetime import datetime, timedelta

            # Calculate the date threshold
            date_threshold = datetime.now() - timedelta(days=days)
            date_threshold_str = date_threshold.isoformat()

            # Get recent documents
            docs_result = self.client.table("documents").select("*").gte("created_at", date_threshold_str).order("created_at", desc=True).limit(limit).execute()
            documents = docs_result.data or []

            # Enhance with source and embedding info
            enhanced_docs = []
            for doc in documents:
                # Get source info
                source_info = {}
                if doc.get('data_source_id'):
                    source_result = self.client.table("data_sources").select("name,url").eq("id", doc['data_source_id']).execute()
                    if source_result.data:
                        source_info = source_result.data[0]

                # Check if has embedding
                embedding_result = self.client.table("embeddings").select("id").eq("document_id", doc['id']).execute()
                has_embedding = "Yes" if embedding_result.data else "No"

                # Create content preview
                content = doc.get('content', '')
                content_preview = content[:200] + '...' if len(content) > 200 else content

                enhanced_docs.append({
                    'id': doc['id'],
                    'title': doc.get('title', 'Untitled'),
                    'content_preview': content_preview,
                    'document_type': doc.get('document_type', 'Unknown'),
                    'status': doc.get('status', 'Unknown'),
                    'word_count': doc.get('word_count', 0),
                    'created_at': doc.get('created_at'),
                    'source_name': source_info.get('name', 'Unknown'),
                    'source_url': source_info.get('url', ''),
                    'has_embedding': has_embedding
                })

            return enhanced_docs
        except Exception as e:
            print(f"Error getting recent documents: {e}")
            return []

    def search_documents(self, search_term: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search for documents containing specific text."""
        try:
            # Simple text search using ilike (case-insensitive)
            docs_result = self.client.table("documents").select("*").ilike("content", f"%{search_term}%").limit(limit).execute()
            documents = docs_result.data or []

            # Enhance with source info and relevance
            enhanced_docs = []
            for doc in documents:
                # Get source info
                source_info = {}
                if doc.get('data_source_id'):
                    source_result = self.client.table("data_sources").select("name,url").eq("id", doc['data_source_id']).execute()
                    if source_result.data:
                        source_info = source_result.data[0]

                # Create content preview
                content = doc.get('content', '')
                content_preview = content[:300] + '...' if len(content) > 300 else content

                # Simple relevance calculation (number of occurrences)
                relevance = content.lower().count(search_term.lower()) / len(content.split()) if content else 0

                enhanced_docs.append({
                    'id': doc['id'],
                    'title': doc.get('title', 'Untitled'),
                    'content_preview': content_preview,
                    'document_type': doc.get('document_type', 'Unknown'),
                    'word_count': doc.get('word_count', 0),
                    'source_name': source_info.get('name', 'Unknown'),
                    'source_url': source_info.get('url', ''),
                    'relevance': relevance
                })

            # Sort by relevance
            enhanced_docs.sort(key=lambda x: x['relevance'], reverse=True)
            return enhanced_docs
        except Exception as e:
            print(f"Error searching documents: {e}")
            return []

    def get_document_content(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get the full content of a specific document."""
        try:
            # Get document
            doc_result = self.client.table("documents").select("*").eq("id", document_id).execute()
            if not doc_result.data:
                return None

            doc = doc_result.data[0]

            # Get source info
            source_info = {}
            if doc.get('data_source_id'):
                source_result = self.client.table("data_sources").select("*").eq("id", doc['data_source_id']).execute()
                if source_result.data:
                    source_info = source_result.data[0]

            # Get embedding info
            embedding_info = {}
            embedding_result = self.client.table("embeddings").select("*").eq("document_id", document_id).execute()
            if embedding_result.data:
                embedding_info = embedding_result.data[0]

            return {
                'id': doc['id'],
                'title': doc.get('title', 'Untitled'),
                'content': doc.get('content', ''),
                'document_type': doc.get('document_type', 'Unknown'),
                'word_count': doc.get('word_count', 0),
                'character_count': doc.get('character_count', 0),
                'chunk_index': doc.get('chunk_index', 0),
                'status': doc.get('status', 'Unknown'),
                'created_at': doc.get('created_at'),
                'metadata': doc.get('metadata', {}),
                'source_name': source_info.get('name', 'Unknown'),
                'source_url': source_info.get('url', ''),
                'source_type': source_info.get('source_type', 'Unknown'),
                'model_name': embedding_info.get('model_name', 'No embedding'),
                'embedding_dimensions': 1536 if embedding_info else 0
            }
        except Exception as e:
            print(f"Error getting document content: {e}")
            return None

    def get_orphaned_documents(self) -> List[Dict[str, Any]]:
        """Get documents that don't have embeddings."""
        try:
            # Get all documents
            docs_result = self.client.table("documents").select("*").execute()
            documents = docs_result.data or []

            # Get all document IDs that have embeddings
            embeddings_result = self.client.table("embeddings").select("document_id").execute()
            embedded_doc_ids = {e['document_id'] for e in embeddings_result.data or []}

            # Find orphaned documents
            orphaned_docs = []
            for doc in documents:
                if doc['id'] not in embedded_doc_ids:
                    # Get source info
                    source_info = {}
                    if doc.get('data_source_id'):
                        source_result = self.client.table("data_sources").select("name").eq("id", doc['data_source_id']).execute()
                        if source_result.data:
                            source_info = source_result.data[0]

                    # Create content preview
                    content = doc.get('content', '')
                    content_preview = content[:150] + '...' if len(content) > 150 else content

                    orphaned_docs.append({
                        'id': doc['id'],
                        'title': doc.get('title', 'Untitled'),
                        'content_preview': content_preview,
                        'document_type': doc.get('document_type', 'Unknown'),
                        'status': doc.get('status', 'Unknown'),
                        'created_at': doc.get('created_at'),
                        'source_name': source_info.get('name', 'Unknown')
                    })

            # Sort by creation date
            orphaned_docs.sort(key=lambda x: x['created_at'] or '', reverse=True)
            return orphaned_docs
        except Exception as e:
            print(f"Error getting orphaned documents: {e}")
            return []

    def print_summary(self):
        """Print a summary of the document/embedding store."""
        print("📊 RAG System Content Audit")
        print("=" * 50)

        # Get overall summary
        summary = self.get_document_summary()
        print(f"Total Documents: {summary.get('total_documents', 0)}")
        print(f"Total Embeddings: {summary.get('total_embeddings', 0)}")
        print(f"Total Words: {summary.get('total_words', 0):,}")
        print(f"Total Characters: {summary.get('total_characters', 0):,}")
        print(f"Total Sources: {summary.get('total_sources', 0)}")
        print(f"Date Range: {summary.get('oldest_document', 'N/A')} to {summary.get('newest_document', 'N/A')}")
        print()

        # Get documents by source
        print("📚 Content by Source:")
        print("-" * 30)
        sources = self.get_documents_by_source(10)
        for source in sources:
            print(f"• {source['source_name']} ({source['source_type']})")
            print(f"  Documents: {source['document_count']}, Embeddings: {source['embedding_count']}")
            print(f"  Words: {source['total_words']:,}, URL: {source['url']}")
            print()

        # Check for orphaned documents
        orphaned = self.get_orphaned_documents()
        if orphaned:
            print(f"⚠️  Found {len(orphaned)} documents without embeddings:")
            for doc in orphaned[:5]:  # Show first 5
                print(f"• {doc['title']} ({doc['document_type']}) - {doc['source_name']}")
            if len(orphaned) > 5:
                print(f"  ... and {len(orphaned) - 5} more")
        else:
            print("✅ All documents have embeddings")

    def export_content_to_file(self, filename: str = "audit_export.json"):
        """Export all document content to a JSON file for inspection."""
        try:
            # Get all documents
            docs_result = self.client.table("documents").select("*").order("created_at", desc=True).execute()
            documents = docs_result.data or []

            # Enhance with source info
            enhanced_docs = []
            for doc in documents:
                # Get source info
                source_info = {}
                if doc.get('data_source_id'):
                    source_result = self.client.table("data_sources").select("name,url,source_type").eq("id", doc['data_source_id']).execute()
                    if source_result.data:
                        source_info = source_result.data[0]

                enhanced_docs.append({
                    'id': doc['id'],
                    'title': doc.get('title', 'Untitled'),
                    'content': doc.get('content', ''),
                    'document_type': doc.get('document_type', 'Unknown'),
                    'word_count': doc.get('word_count', 0),
                    'created_at': doc.get('created_at'),
                    'source_name': source_info.get('name', 'Unknown'),
                    'source_url': source_info.get('url', ''),
                    'source_type': source_info.get('source_type', 'Unknown')
                })

            if enhanced_docs:
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(enhanced_docs, f, indent=2, ensure_ascii=False, default=str)
                print(f"✅ Exported {len(enhanced_docs)} documents to {filename}")
            else:
                print("❌ No data to export")
        except Exception as e:
            print(f"Error exporting content: {e}")


def main():
    """Main function to run the auditor."""
    auditor = EmbeddingsAuditor()

    import argparse
    parser = argparse.ArgumentParser(description="Audit RAG embeddings and content")
    parser.add_argument("--summary", action="store_true", help="Show system summary")
    parser.add_argument("--recent", type=int, default=7, help="Show recent documents (days)")
    parser.add_argument("--search", type=str, help="Search for documents containing text")
    parser.add_argument("--document", type=str, help="Show full content of specific document ID")
    parser.add_argument("--export", type=str, help="Export all content to JSON file")

    args = parser.parse_args()

    if args.summary:
        auditor.print_summary()

    if args.recent:
        print(f"📅 Recent Documents (last {args.recent} days):")
        print("-" * 40)
        recent = auditor.get_recent_documents(args.recent)
        for doc in recent:
            print(f"• {doc['title']} ({doc['document_type']})")
            print(f"  Source: {doc['source_name']}")
            print(f"  Words: {doc['word_count']}, Created: {doc['created_at']}")
            print(f"  Preview: {doc['content_preview']}...")
            print()

    if args.search:
        print(f"🔍 Search Results for '{args.search}':")
        print("-" * 40)
        results = auditor.search_documents(args.search)
        for doc in results:
            print(f"• {doc['title']} (Relevance: {doc['relevance']:.3f})")
            print(f"  Source: {doc['source_name']}")
            print(f"  Preview: {doc['content_preview']}...")
            print()

    if args.document:
        doc = auditor.get_document_content(args.document)
        if doc:
            print(f"📄 Document: {doc['title']}")
            print("-" * 40)
            print(f"ID: {doc['id']}")
            print(f"Type: {doc['document_type']}")
            print(f"Source: {doc['source_name']}")
            print(f"Words: {doc['word_count']}, Characters: {doc['character_count']}")
            print(f"Created: {doc['created_at']}")
            print(f"Embedding Model: {doc['model_name']}")
            print(f"Embedding Dimensions: {doc['embedding_dimensions']}")
            print("\nContent:")
            print("-" * 20)
            print(doc['content'])
        else:
            print(f"❌ Document {args.document} not found")

    if args.export:
        auditor.export_content_to_file(args.export)

    # If no arguments, show summary
    if not any(vars(args).values()):
        auditor.print_summary()

if __name__ == "__main__":
    main()
