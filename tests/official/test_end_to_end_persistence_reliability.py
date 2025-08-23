#!/usr/bin/env python3
"""
Comprehensive end-to-end test: verifies message persistence after the full flow completes.

This test intentionally checks persistence ONLY AFTER the entire step-by-step
conversation flow, to validate reliability without depending on intermediate checks.

Assertions:
1) All user inputs sent during the flow exist in DB for the conversation
2) Assistant messages count >= 2 (initial prompt + final quotation)
3) Ordering and roles are sane: first=user, second=assistant, last=assistant
4) Final assistant message contains quotation content markers
5) No message-less conversation; total messages >= expected minimum
"""

import asyncio
import os
from datetime import datetime
from typing import List, Dict, Any

import httpx


API_URL = os.getenv("CHAT_API_URL", "http://localhost:8000/api/chat/message")


async def _post_message(client: httpx.AsyncClient, conversation_id: str, user_id: str, msg: str) -> Dict[str, Any]:
    resp = await client.post(
        API_URL,
        json={
            "message": msg,
            "conversation_id": conversation_id,
            "user_id": user_id,
        },
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()


async def _fetch_messages_from_db(conversation_id: str) -> List[Dict[str, Any]]:
    """Fetch messages from the DB via db_client (synchronous client wrapped in to_thread)."""
    try:
        # Lazy import so test can run outside container
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "backend"))
        from core.database import db_client  # type: ignore

        def _query():
            return (
                db_client.client.table("messages")
                .select("id, role, content, created_at")
                .eq("conversation_id", conversation_id)
                .order("created_at", desc=False)
                .execute()
            )

        result = await asyncio.to_thread(_query)
        return result.data or []
    except Exception as e:
        raise AssertionError(f"Failed to fetch messages from DB: {e}")


async def run_test() -> None:
    user_id = os.getenv("TEST_USER_ID", "f26449e2-dce9-4b29-acd0-cb39a1f671fd")
    conversation_id = f"persistence-{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}"

    # Inputs (deterministic, mirrors the usual HITL flow)
    user_inputs = [
        "Generate an Informal Quote",
        "John Smith - ABC Corporation",
        "Honda Civic",
        "john.smith@abc.com | +1-555-0123",
        "Approve",
    ]

    async with httpx.AsyncClient() as client:
        # Step-by-step flow
        for idx, text in enumerate(user_inputs):
            response = await _post_message(client, conversation_id, user_id, text)
            # Optional small delay to allow any background tasks (titles, summaries) to run without
            # affecting message persistence, which is now direct.
            await asyncio.sleep(0.3 if idx < len(user_inputs) - 1 else 1.0)

    # After the entire flow completes, verify persistence
    messages = await _fetch_messages_from_db(conversation_id)
    assert messages, "No messages found for conversation (should never be empty)."

    # Sanity counts
    total = len(messages)
    num_user = sum(1 for m in messages if m.get("role") in ("user", "human"))
    num_assistant = sum(1 for m in messages if m.get("role") == "assistant")

    # Expect at least all user inputs and two assistant messages (prompt + final)
    assert total >= len(user_inputs) + 1, f"Too few messages: total={total}"
    assert num_user >= len(user_inputs), f"Missing user messages: {num_user}/{len(user_inputs)}"
    assert num_assistant >= 2, f"Expected >=2 assistant messages, got {num_assistant}"

    # Order checks: first=user, second=assistant, last=assistant
    assert messages[0]["role"] in ("user", "human"), f"First message should be user, got {messages[0]['role']}"
    assert messages[1]["role"] == "assistant", f"Second message should be assistant, got {messages[1]['role']}"
    assert messages[-1]["role"] == "assistant", f"Last message should be assistant, got {messages[-1]['role']}"

    # Ensure all user inputs exist in DB content (simple containment check)
    db_contents = [m.get("content", "") for m in messages]
    for expected in user_inputs:
        assert any(expected in c for c in db_contents), f"Missing user input in DB: {expected}"

    # Final assistant content should include a quotation marker
    final_assistant = messages[-1]["content"] or ""
    markers = ["Quotation", "Professional Vehicle Quotation", "Pricing", "Total"]
    assert any(m in final_assistant for m in markers), "Final assistant message missing quotation markers."

    print("\n✅ Persistence reliability test passed.")
    print(f"Conversation: {conversation_id}")
    print(f"Total messages: {total} (user={num_user}, assistant={num_assistant})")


if __name__ == "__main__":
    asyncio.run(run_test())


