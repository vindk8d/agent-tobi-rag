#!/usr/bin/env python3
"""
Simple token usage tracker for OpenAI API consumption monitoring.
Helps identify high-usage patterns and prevents quota exhaustion.
"""

import json
import logging
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class TokenUsageTracker:
    """
    Simple token usage tracker with basic alerting capabilities.
    In production, this would integrate with a proper metrics system.
    """

    def __init__(self, usage_file: str = "token_usage.json"):
        self.usage_file = Path(usage_file)
        self.usage_data = defaultdict(lambda: defaultdict(int))
        self.daily_limits = {
            "total": 100000,  # Daily token limit
            "alert_threshold": 80000,  # Alert when approaching limit
            "embedding": 50000,  # Daily embedding token limit
            "chat": 50000  # Daily chat token limit
        }
        self.load_usage_data()

    def load_usage_data(self):
        """Load existing usage data from file."""
        try:
            if self.usage_file.exists():
                with open(self.usage_file, 'r') as f:
                    data = json.load(f)
                    self.usage_data = defaultdict(lambda: defaultdict(int), data)
        except Exception as e:
            logger.warning(f"Could not load usage data: {e}")

    def save_usage_data(self):
        """Save usage data to file."""
        try:
            with open(self.usage_file, 'w') as f:
                json.dump(dict(self.usage_data), f, indent=2)
        except Exception as e:
            logger.error(f"Could not save usage data: {e}")

    def track_usage(self, operation: str, tokens: int, operation_type: str = "chat"):
        """
        Track token usage for a specific operation.

        Args:
            operation: Name of the operation (e.g., "agent_call", "summarization", "embedding")
            tokens: Number of tokens consumed
            operation_type: Type of operation ("chat", "embedding", "completion")
        """
        today = datetime.now().strftime("%Y-%m-%d")

        # Track by operation and date
        self.usage_data[today][operation] += tokens
        self.usage_data[today][f"total_{operation_type}"] += tokens
        self.usage_data[today]["total"] += tokens

        # Check if we should alert
        self.check_usage_alerts(today)

        # Save periodically
        if self.usage_data[today]["total"] % 1000 == 0:
            self.save_usage_data()

    def check_usage_alerts(self, date: str):
        """Check if usage is approaching limits and log alerts."""
        total_today = self.usage_data[date]["total"]

        if total_today >= self.daily_limits["alert_threshold"]:
            logger.warning(f"⚠️ APPROACHING DAILY LIMIT: {total_today}/{self.daily_limits['total']} tokens used today")

        if total_today >= self.daily_limits["total"]:
            logger.error(f"🚨 DAILY LIMIT EXCEEDED: {total_today}/{self.daily_limits['total']} tokens used today")

    def get_usage_stats(self, days: int = 7) -> Dict[str, Any]:
        """Get usage statistics for the last N days."""
        stats = {
            "daily_usage": {},
            "operation_breakdown": defaultdict(int),
            "total_tokens": 0
        }

        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        for i in range(days):
            date = (start_date + timedelta(days=i)).strftime("%Y-%m-%d")
            daily_total = self.usage_data[date]["total"]
            stats["daily_usage"][date] = daily_total
            stats["total_tokens"] += daily_total

            # Aggregate operation breakdown
            for operation, tokens in self.usage_data[date].items():
                if operation not in ["total", "total_chat", "total_embedding"]:
                    stats["operation_breakdown"][operation] += tokens

        return stats

    def print_usage_report(self, days: int = 7):
        """Print a formatted usage report."""
        stats = self.get_usage_stats(days)

        print(f"\n📊 Token Usage Report (Last {days} days)")
        print("=" * 50)

        # Daily usage
        print("\n📅 Daily Usage:")
        for date, tokens in stats["daily_usage"].items():
            status = "🟢" if tokens < self.daily_limits["alert_threshold"] else "🟡" if tokens < self.daily_limits["total"] else "🔴"
            print(f"  {date}: {tokens:,} tokens {status}")

        # Operation breakdown
        print("\n🔧 Operation Breakdown:")
        sorted_ops = sorted(stats["operation_breakdown"].items(), key=lambda x: x[1], reverse=True)
        for operation, tokens in sorted_ops:
            percentage = (tokens / stats["total_tokens"]) * 100 if stats["total_tokens"] > 0 else 0
            print(f"  {operation}: {tokens:,} tokens ({percentage:.1f}%)")

        # Summary
        print(f"\n📈 Total Tokens: {stats['total_tokens']:,}")
        print(f"📊 Daily Average: {stats['total_tokens'] // days:,}")
        print(f"🎯 Current Limit: {self.daily_limits['total']:,}")

        # Recommendations
        if stats["total_tokens"] > self.daily_limits["total"] * days * 0.8:
            print("\n⚠️ OPTIMIZATION RECOMMENDATIONS:")
            print("  - Consider reducing max_tokens in agent configuration")
            print("  - Implement response caching for repeated queries")
            print("  - Optimize memory summarization frequency")
            print("  - Review tool calling patterns for efficiency")


# Global instance
usage_tracker = TokenUsageTracker()


def track_tokens(operation: str, tokens: int, operation_type: str = "chat"):
    """Convenience function to track token usage."""
    usage_tracker.track_usage(operation, tokens, operation_type)


def get_usage_report(days: int = 7):
    """Get usage statistics."""
    return usage_tracker.get_usage_stats(days)


def print_usage_report(days: int = 7):
    """Print formatted usage report."""
    usage_tracker.print_usage_report(days)

# Example usage:
if __name__ == "__main__":
    # Example usage tracking
    track_tokens("agent_call", 1500, "chat")
    track_tokens("summarization", 500, "chat")
    track_tokens("embedding", 100, "embedding")

    # Print report
    print_usage_report(7)
