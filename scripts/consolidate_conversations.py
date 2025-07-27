#!/usr/bin/env python3
"""
Consolidate Multiple Conversations Per User

This script consolidates users who have multiple conversations into a single 
conversation per user, preserving all messages in chronological order.

Usage:
    python scripts/consolidate_conversations.py [--dry-run]
    
Options:
    --dry-run    Show what would be done without making changes
"""

import asyncio
import argparse
import sys
import os
from datetime import datetime
from typing import List, Dict, Any

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from database import db_client
except ImportError:
    print("❌ Could not import database. Make sure you're running from the project root.")
    sys.exit(1)


class ConversationConsolidator:
    """Consolidates multiple conversations per user into single conversations."""
    
    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.stats = {
            'users_processed': 0,
            'conversations_consolidated': 0,
            'messages_moved': 0,
            'conversations_deleted': 0
        }
    
    async def run(self):
        """Main consolidation process."""
        print("🔍 Analyzing conversations for consolidation...")
        
        # Get all users with multiple conversations
        users_with_multiple = await self._get_users_with_multiple_conversations()
        
        if not users_with_multiple:
            print("✅ No users have multiple conversations. Nothing to consolidate.")
            return
        
        print(f"📊 Found {len(users_with_multiple)} users with multiple conversations:")
        for user in users_with_multiple:
            print(f"   - {user['display_name']} ({user['email']}): {user['conversation_count']} conversations")
        
        if self.dry_run:
            print("\n🔍 DRY RUN MODE - Showing what would be done:")
        else:
            print(f"\n🚀 Starting consolidation process...")
            
        # Process each user
        for user in users_with_multiple:
            await self._consolidate_user_conversations(user)
        
        # Print summary
        print(f"\n📈 Consolidation Summary:")
        print(f"   Users processed: {self.stats['users_processed']}")
        print(f"   Conversations consolidated: {self.stats['conversations_consolidated']}")
        print(f"   Messages moved: {self.stats['messages_moved']}")
        print(f"   Old conversations deleted: {self.stats['conversations_deleted']}")
        
        if self.dry_run:
            print("\n💡 Run without --dry-run to actually perform the consolidation.")
    
    async def _get_users_with_multiple_conversations(self) -> List[Dict[str, Any]]:
        """Get users who have multiple conversations."""
        # Get all users with their conversation counts using separate queries
        users_result = db_client.client.table('users').select('id, display_name, email').execute()
        
        if not users_result.data:
            return []
        
        users_with_multiple = []
        
        for user in users_result.data:
            # Count conversations for this user
            conversations_result = db_client.client.table('conversations').select('id, created_at').eq(
                'user_id', user['id']
            ).execute()
            
            conversation_count = len(conversations_result.data) if conversations_result.data else 0
            
            if conversation_count > 1:
                conversations = conversations_result.data
                first_conversation = min(conv['created_at'] for conv in conversations)
                latest_conversation = max(conv['created_at'] for conv in conversations)
                
                users_with_multiple.append({
                    'id': user['id'],
                    'display_name': user['display_name'],
                    'email': user['email'],
                    'conversation_count': conversation_count,
                    'first_conversation': first_conversation,
                    'latest_conversation': latest_conversation
                })
        
        # Sort by conversation count descending
        users_with_multiple.sort(key=lambda x: x['conversation_count'], reverse=True)
        return users_with_multiple
    
    async def _consolidate_user_conversations(self, user: Dict[str, Any]):
        """Consolidate all conversations for a single user."""
        user_id = user['id']
        user_display = f"{user['display_name']} ({user['email']})"
        
        print(f"\n👤 Processing {user_display}...")
        
        # Get all conversations for this user
        conversations = db_client.client.table('conversations').select(
            'id, title, created_at, updated_at, metadata'
        ).eq('user_id', user_id).order('created_at', desc=False).execute()
        
        if not conversations.data or len(conversations.data) <= 1:
            print(f"   ⚠️  User has {len(conversations.data)} conversations, skipping")
            return
        
        # Choose the primary conversation (oldest one)
        primary_conversation = conversations.data[0]
        other_conversations = conversations.data[1:]
        
        print(f"   📝 Primary conversation: {primary_conversation['id']}")
        print(f"   🔄 Consolidating {len(other_conversations)} other conversations")
        
        # Get all messages from other conversations
        total_messages_moved = 0
        
        for conv in other_conversations:
            conv_id = conv['id']
            print(f"   📥 Processing conversation {conv_id}...")
            
            # Get messages from this conversation
            messages = db_client.client.table('messages').select(
                'id, role, content, created_at, metadata'
            ).eq('conversation_id', conv_id).order('created_at', desc=False).execute()
            
            if messages.data:
                message_count = len(messages.data)
                print(f"      📨 Moving {message_count} messages...")
                
                if not self.dry_run:
                    # Move messages to primary conversation
                    for message in messages.data:
                        db_client.client.table('messages').update({
                            'conversation_id': primary_conversation['id']
                        }).eq('id', message['id']).execute()
                
                total_messages_moved += message_count
                self.stats['messages_moved'] += message_count
        
        if not self.dry_run:
            # Update primary conversation timestamp
            db_client.client.table('conversations').update({
                'updated_at': datetime.now().isoformat(),
                'title': 'Consolidated Conversation',
                'metadata': {
                    **primary_conversation.get('metadata', {}),
                    'consolidated_from': len(other_conversations),
                    'consolidated_at': datetime.now().isoformat()
                }
            }).eq('id', primary_conversation['id']).execute()
            
            # Delete other conversations
            for conv in other_conversations:
                db_client.client.table('conversations').delete().eq('id', conv['id']).execute()
                self.stats['conversations_deleted'] += 1
        
        self.stats['users_processed'] += 1
        self.stats['conversations_consolidated'] += len(other_conversations)
        
        print(f"   ✅ Consolidated {len(other_conversations)} conversations, moved {total_messages_moved} messages")


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Consolidate multiple conversations per user')
    parser.add_argument('--dry-run', action='store_true', 
                        help='Show what would be done without making changes')
    
    args = parser.parse_args()
    
    consolidator = ConversationConsolidator(dry_run=args.dry_run)
    await consolidator.run()


if __name__ == '__main__':
    asyncio.run(main()) 