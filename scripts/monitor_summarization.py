#!/usr/bin/env python3
"""
Monitor and display conversation summarization status.
Shows recent conversations, their message counts, and summarization status.
"""

import asyncio
import os
import sys
from datetime import datetime, timedelta, timezone

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

# Backend imports (after path setup)
from backend.core.database import db_client
from backend.core.config import get_settings


async def monitor_conversations():
    """Monitor recent conversations and their message counts."""
    print("🔍 Conversation Summarization Monitor")
    print("=" * 50)

    try:
        settings = await get_settings()
        client = db_client.client

        print(f"📊 Current Configuration:")
        print(f"   • Summary triggers at: {settings.memory_summary_interval} messages")
        print(f"   • Max messages: {settings.memory_max_messages}")
        print(f"   • Auto-summarization: {'Enabled' if settings.memory_auto_summarize else 'Disabled'}")
        print()

        # Get recent conversations with message counts
        conversations = await asyncio.to_thread(
            lambda: client.table("conversations")
            .select("id,user_id,title,created_at,updated_at")
            .gte("updated_at", (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat())
            .order("updated_at", desc=True)
            .limit(10)
            .execute()
        )

        print("📝 Recent Conversations (Last 24 hours):")
        print("-" * 80)

        for conv in conversations.data:
            # Count messages for each conversation
            messages_result = await asyncio.to_thread(
                lambda: client.table("messages")
                .select("id", count="exact")
                .eq("conversation_id", conv['id'])
                .execute()
            )

            message_count = messages_result.count or 0

            # Check if conversation has summary
            summary_result = await asyncio.to_thread(
                lambda: client.table("conversation_summaries")
                .select("id")
                .eq("conversation_id", conv['id'])
                .execute()
            )

            has_summary = len(summary_result.data) > 0

            # Status indicators
            status = ""
            if message_count >= settings.memory_summary_interval:
                if has_summary:
                    status = "✅ SUMMARIZED"
                else:
                    status = "🔄 SHOULD SUMMARIZE"
            else:
                status = f"📝 {message_count}/{settings.memory_summary_interval} messages"

            print(f"• {conv['id'][:8]}... | {conv['user_id'][:12]} | {message_count:2d} msgs | {status}")

        # Check recent summaries
        print(f"\n📚 Recent Summaries:")
        print("-" * 40)

        summaries = await asyncio.to_thread(
            lambda: client.table("conversation_summaries")
            .select("conversation_id,user_id,created_at,message_count")
            .gte("created_at", (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat())
            .order("created_at", desc=True)
            .limit(5)
            .execute()
        )

        if summaries.data:
            for summary in summaries.data:
                created = summary['created_at'][:19]  # Remove timezone info for display
                print(f"• {summary['conversation_id'][:8]}... | {summary['user_id'][:12]} | {summary['message_count']} msgs | {created}")
        else:
            print("   No summaries created in the last 24 hours")

        return True

    except Exception as e:
        print(f"❌ Error monitoring conversations: {e}")
        return False


async def check_system_health():
    """Check if the summarization system components are healthy."""
    print("\n🏥 System Health Check:")
    print("-" * 30)

    try:
        settings = await get_settings()

        # Check configuration
        config_ok = (
            settings.memory_summary_interval > 0
            and settings.memory_max_messages > 0
            and settings.memory_summary_interval <= settings.memory_max_messages
        )

        print(f"⚙️  Configuration: {'✅ OK' if config_ok else '❌ ERROR'}")

        # Check database connectivity
        try:
            await asyncio.to_thread(
                lambda: db_client.client.table("conversations").select("id").limit(1).execute()
            )
            db_ok = True
        except Exception:
            db_ok = False

        print(f"🗄️  Database: {'✅ OK' if db_ok else '❌ ERROR'}")

        # Check if summarization would be enabled
        auto_enabled = settings.memory_auto_summarize
        print(f"🤖 Auto-Summarization: {'✅ ENABLED' if auto_enabled else '❌ DISABLED'}")

        return config_ok and db_ok

    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False


async def main():
    """Run the monitoring dashboard."""
    print("🚀 Starting Conversation Summarization Monitor")
    print("=" * 80)

    # Check system health
    health_ok = await check_system_health()

    if not health_ok:
        print("\n⚠️  System health issues detected. Check configuration and database.")
        return

    # Monitor conversations
    await monitor_conversations()

    print(f"\n📋 Monitoring Tips:")
    print(f"   • Look for 'Auto-triggering summarization' in agent logs")
    settings = await get_settings()
    print(f"   • Conversations with {settings.memory_summary_interval}+ messages should auto-summarize")
    print(f"   • Check conversation_summaries table for stored summaries")
    print(f"   • Re-run this script to see updated statistics")

    print("\n" + "=" * 80)

if __name__ == "__main__":
    asyncio.run(main())
