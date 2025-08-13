"""
Background Task Infrastructure for Streamlined Memory Management.

This module provides a simplified, efficient background task management system
that replaces the complex multi-layer memory architecture with direct LangGraph patterns.

Key Features:
- Single BackgroundTaskManager class replacing multi-layer memory system
- Task scheduling, prioritization, and retry logic (max 3 retries)
- Comprehensive logging and error handling for task execution
- Integration with existing Supabase tables (messages, conversation_summaries)
- LangChain role mapping compatibility (human→user, ai→assistant)
- Background task monitoring and health checks

Performance Goals:
- 60-80% response time improvement by removing blocking operations
- 70% AgentState size reduction following LangGraph best practices
- Token conservation through smart caching and minimal LLM calls
"""

import asyncio
import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any
from uuid import uuid4
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor

from backend.core.config import get_settings
from backend.core.database import db_client

logger = logging.getLogger(__name__)


class TaskPriority(Enum):
    """Task priority levels for background processing."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"


@dataclass
class BackgroundTask:
    """
    Background task definition with scheduling and retry logic.
    
    Simplified task structure focusing on essential data only.
    """
    id: str = field(default_factory=lambda: str(uuid4()))
    task_type: str = ""
    priority: TaskPriority = TaskPriority.NORMAL
    status: TaskStatus = TaskStatus.PENDING
    
    # Task data - keep minimal for performance
    data: Dict[str, Any] = field(default_factory=dict)
    
    # Execution tracking
    created_at: datetime = field(default_factory=datetime.utcnow)
    scheduled_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Retry logic (max 3 retries as per PRD)
    retry_count: int = 0
    max_retries: int = 3
    retry_delay: float = 1.0  # seconds, exponential backoff
    
    # Error handling
    last_error: Optional[str] = None
    error_history: List[str] = field(default_factory=list)
    
    # Context for task execution
    conversation_id: Optional[str] = None
    user_id: Optional[str] = None
    customer_id: Optional[str] = None
    employee_id: Optional[str] = None


class BackgroundTaskManager:
    """
    Single unified background task manager replacing multi-layer memory system.
    
    Integrates functionality from ConversationConsolidator and other complex components
    into a streamlined, efficient system following LangGraph best practices.
    
    Key Responsibilities:
    1. Task scheduling, prioritization, and execution
    2. Message persistence to existing Supabase messages table
    3. Conversation summary generation using existing conversation_summaries table
    4. Configurable threshold-based processing
    5. Comprehensive error handling and retry logic
    6. Background task monitoring and health checks
    """
    
    def __init__(self):
        self.settings = None  # Will be loaded asynchronously
        self.is_running = False
        self.task_queue: Dict[TaskPriority, List[BackgroundTask]] = {
            priority: [] for priority in TaskPriority
        }
        self.active_tasks: Dict[str, BackgroundTask] = {}
        self.completed_tasks: List[BackgroundTask] = []
        self.failed_tasks: List[BackgroundTask] = []
        
        # Worker configuration
        self.max_concurrent_tasks = 10
        self.worker_pool: Optional[ThreadPoolExecutor] = None
        self.processing_loop_task: Optional[asyncio.Task] = None
        
        # Health monitoring
        self.health_stats = {
            'total_tasks_processed': 0,
            'successful_tasks': 0,
            'failed_tasks': 0,
            'average_processing_time': 0.0,
            'last_health_check': datetime.utcnow(),
            'queue_sizes': {priority.name: 0 for priority in TaskPriority}
        }
        
        # Configuration (will be loaded from settings)
        self.employee_max_messages = 12
        self.customer_max_messages = 15
        self.summary_threshold = 10
        
        logger.info("BackgroundTaskManager initialized")
    
    async def _ensure_initialized(self):
        """Ensure the task manager is initialized asynchronously."""
        if self.settings is None:
            self.settings = await get_settings()
            
            # Load configurable thresholds from settings
            self.employee_max_messages = getattr(self.settings, 'EMPLOYEE_MAX_MESSAGES', 12)
            self.customer_max_messages = getattr(self.settings, 'CUSTOMER_MAX_MESSAGES', 15)
            self.summary_threshold = getattr(self.settings, 'SUMMARY_THRESHOLD', 10)
            
            logger.info(f"BackgroundTaskManager configured: employee_max={self.employee_max_messages}, "
                       f"customer_max={self.customer_max_messages}, summary_threshold={self.summary_threshold}")
    
    async def start(self):
        """Start the background task processing system."""
        await self._ensure_initialized()
        
        if self.is_running:
            logger.warning("BackgroundTaskManager is already running")
            return
        
        self.is_running = True
        self.worker_pool = ThreadPoolExecutor(max_workers=self.max_concurrent_tasks)
        
        # Start the main processing loop
        self.processing_loop_task = asyncio.create_task(self._processing_loop())
        
        logger.info(f"BackgroundTaskManager started with {self.max_concurrent_tasks} workers")
    
    async def stop(self):
        """Stop the background task processing system."""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Cancel processing loop
        if self.processing_loop_task:
            self.processing_loop_task.cancel()
            try:
                await self.processing_loop_task
            except asyncio.CancelledError:
                pass
        
        # Shutdown worker pool
        if self.worker_pool:
            self.worker_pool.shutdown(wait=True)
        
        logger.info("BackgroundTaskManager stopped")
    
    def schedule_task(self, task: BackgroundTask, delay_seconds: float = 0) -> str:
        """
        Schedule a background task for execution.
        
        Args:
            task: The background task to schedule
            delay_seconds: Optional delay before task execution
            
        Returns:
            Task ID for tracking
        """
        if delay_seconds > 0:
            task.scheduled_at = datetime.utcnow() + timedelta(seconds=delay_seconds)
        else:
            task.scheduled_at = datetime.utcnow()
        
        # Add to appropriate priority queue
        self.task_queue[task.priority].append(task)
        
        # Update health stats
        self.health_stats['queue_sizes'][task.priority.name] += 1
        
        logger.debug(f"Scheduled task {task.id} ({task.task_type}) with priority {task.priority.name}")
        return task.id
    
    def schedule_message_storage(self, conversation_id: str, user_id: str, 
                                messages: List[Dict[str, Any]], priority: TaskPriority = TaskPriority.NORMAL) -> str:
        """
        Schedule message storage to existing Supabase messages table.
        
        Args:
            conversation_id: Conversation ID
            user_id: User ID
            messages: List of messages to store
            priority: Task priority
            
        Returns:
            Task ID
        """
        task = BackgroundTask(
            task_type="message_storage",
            priority=priority,
            data={
                "messages": messages,
                "table": "messages"  # Use existing Supabase table
            },
            conversation_id=conversation_id,
            user_id=user_id
        )
        
        return self.schedule_task(task)
    
    def schedule_summary_generation(self, conversation_id: str, user_id: str,
                                   customer_id: Optional[str] = None, employee_id: Optional[str] = None,
                                   priority: TaskPriority = TaskPriority.NORMAL) -> str:
        """
        Schedule conversation summary generation using existing conversation_summaries table.
        
        Args:
            conversation_id: Conversation ID
            user_id: User ID
            customer_id: Customer ID (if customer user)
            employee_id: Employee ID (if employee user)
            priority: Task priority
            
        Returns:
            Task ID
        """
        # Determine message threshold based on user type
        max_messages = self.customer_max_messages if customer_id else self.employee_max_messages
        
        task = BackgroundTask(
            task_type="summary_generation",
            priority=priority,
            data={
                "max_messages": max_messages,
                "summary_threshold": self.summary_threshold,
                "table": "conversation_summaries"  # Use existing Supabase table
            },
            conversation_id=conversation_id,
            user_id=user_id,
            customer_id=customer_id,
            employee_id=employee_id
        )
        
        return self.schedule_task(task)
    
    def schedule_context_loading(self, user_id: str, conversation_id: str,
                                customer_id: Optional[str] = None, employee_id: Optional[str] = None,
                                priority: TaskPriority = TaskPriority.HIGH) -> str:
        """
        Schedule lazy context loading for user data.
        
        Args:
            user_id: User ID
            conversation_id: Conversation ID
            customer_id: Customer ID (if customer user)
            employee_id: Employee ID (if employee user)
            priority: Task priority (default HIGH for immediate needs)
            
        Returns:
            Task ID
        """
        task = BackgroundTask(
            task_type="context_loading",
            priority=priority,
            data={
                "load_user_context": True,
                "load_conversation_history": True
            },
            conversation_id=conversation_id,
            user_id=user_id,
            customer_id=customer_id,
            employee_id=employee_id
        )
        
        return self.schedule_task(task)
    
    async def _processing_loop(self):
        """Main background processing loop."""
        logger.info("Background task processing loop started")
        
        while self.is_running:
            try:
                # Process tasks by priority (CRITICAL -> HIGH -> NORMAL -> LOW)
                tasks_processed = 0
                
                for priority in [TaskPriority.CRITICAL, TaskPriority.HIGH, TaskPriority.NORMAL, TaskPriority.LOW]:
                    queue = self.task_queue[priority]
                    
                    # Process ready tasks from this priority level
                    ready_tasks = [
                        task for task in queue 
                        if task.scheduled_at <= datetime.utcnow() and len(self.active_tasks) < self.max_concurrent_tasks
                    ]
                    
                    for task in ready_tasks:
                        queue.remove(task)
                        self.health_stats['queue_sizes'][priority.name] -= 1
                        
                        # Start task execution
                        asyncio.create_task(self._execute_task(task))
                        tasks_processed += 1
                        
                        # Respect concurrency limit
                        if len(self.active_tasks) >= self.max_concurrent_tasks:
                            break
                    
                    # Don't process lower priority queues if we're at capacity
                    if len(self.active_tasks) >= self.max_concurrent_tasks:
                        break
                
                # Update health check
                if tasks_processed > 0:
                    self.health_stats['last_health_check'] = datetime.utcnow()
                
                # Sleep briefly to prevent busy waiting
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error in background processing loop: {e}")
                await asyncio.sleep(1.0)  # Longer sleep on error
    
    async def _execute_task(self, task: BackgroundTask):
        """
        Execute a background task with comprehensive error handling and retry logic.
        
        Args:
            task: The task to execute
        """
        task.status = TaskStatus.RUNNING
        task.started_at = datetime.utcnow()
        self.active_tasks[task.id] = task
        
        try:
            logger.debug(f"Executing task {task.id} ({task.task_type})")
            
            # Route to appropriate handler based on task type
            if task.task_type == "message_storage":
                await self._handle_message_storage(task)
            elif task.task_type == "summary_generation":
                await self._handle_summary_generation(task)
            elif task.task_type == "context_loading":
                await self._handle_context_loading(task)
            else:
                raise ValueError(f"Unknown task type: {task.task_type}")
            
            # Mark task as completed
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.utcnow()
            self.completed_tasks.append(task)
            
            # Update health stats
            self.health_stats['total_tasks_processed'] += 1
            self.health_stats['successful_tasks'] += 1
            
            processing_time = (task.completed_at - task.started_at).total_seconds()
            current_avg = self.health_stats['average_processing_time']
            total_tasks = self.health_stats['total_tasks_processed']
            self.health_stats['average_processing_time'] = (
                (current_avg * (total_tasks - 1) + processing_time) / total_tasks
            )
            
            logger.debug(f"Task {task.id} completed successfully in {processing_time:.2f}s")
            
        except Exception as e:
            await self._handle_task_failure(task, str(e))
        
        finally:
            # Remove from active tasks
            self.active_tasks.pop(task.id, None)
    
    async def _handle_task_failure(self, task: BackgroundTask, error: str):
        """
        Handle task failure with retry logic.
        
        Args:
            task: The failed task
            error: Error message
        """
        task.last_error = error
        task.error_history.append(f"{datetime.utcnow().isoformat()}: {error}")
        
        logger.warning(f"Task {task.id} failed: {error}")
        
        # Check if we should retry
        if task.retry_count < task.max_retries:
            task.retry_count += 1
            task.status = TaskStatus.RETRYING
            
            # Calculate exponential backoff delay
            delay = task.retry_delay * (2 ** (task.retry_count - 1))
            
            # Reschedule with delay
            task.scheduled_at = datetime.utcnow() + timedelta(seconds=delay)
            self.task_queue[task.priority].append(task)
            self.health_stats['queue_sizes'][task.priority.name] += 1
            
            logger.info(f"Retrying task {task.id} in {delay:.1f}s (attempt {task.retry_count}/{task.max_retries})")
        else:
            # Max retries exceeded
            task.status = TaskStatus.FAILED
            task.completed_at = datetime.utcnow()
            self.failed_tasks.append(task)
            
            # Update health stats
            self.health_stats['total_tasks_processed'] += 1
            self.health_stats['failed_tasks'] += 1
            
            logger.error(f"Task {task.id} failed permanently after {task.max_retries} retries")
    
    async def _handle_message_storage(self, task: BackgroundTask):
        """
        Handle message storage to existing Supabase messages table.
        Implements LangChain role mapping (human→user, ai→assistant).
        
        Args:
            task: Message storage task
        """
        messages = task.data.get("messages", [])
        if not messages:
            return
        
        # Prepare messages for Supabase with role mapping
        supabase_messages = []
        for msg in messages:
            # Map LangChain roles to database roles
            role = msg.get("role", "user")
            if role == "human":
                role = "user"
            elif role == "ai":
                role = "assistant"
            
            supabase_messages.append({
                "conversation_id": task.conversation_id,
                "role": role,
                "content": msg.get("content", ""),
                "user_id": task.user_id,
                "created_at": datetime.utcnow().isoformat(),
                "metadata": msg.get("metadata", {})
            })
        
        # Store to existing Supabase messages table
        def _store_messages():
            return db_client.client.table("messages").insert(supabase_messages).execute()
        
        result = await asyncio.to_thread(_store_messages)
        
        if not result.data:
            raise Exception("Failed to store messages to database")
        
        logger.debug(f"Stored {len(supabase_messages)} messages for conversation {task.conversation_id}")
    
    async def _handle_summary_generation(self, task: BackgroundTask):
        """
        Handle conversation summary generation using existing conversation_summaries table.
        Implements configurable threshold-based summary generation.
        
        Args:
            task: Summary generation task
        """
        conversation_id = task.conversation_id
        user_id = task.user_id
        max_messages = task.data.get("max_messages", self.customer_max_messages)
        summary_threshold = task.data.get("summary_threshold", self.summary_threshold)
        
        # Count messages in conversation
        def _count_messages():
            return (db_client.client.table("messages")
                   .select("id", count="exact")
                   .eq("conversation_id", conversation_id)
                   .execute())
        
        count_result = await asyncio.to_thread(_count_messages)
        message_count = count_result.count or 0
        
        # Check if summary generation is needed
        if message_count < summary_threshold:
            logger.debug(f"Conversation {conversation_id} has {message_count} messages, "
                        f"below threshold {summary_threshold}")
            return
        
        # Get recent messages for summary
        def _get_messages():
            return (db_client.client.table("messages")
                   .select("role,content,created_at")
                   .eq("conversation_id", conversation_id)
                   .order("created_at", desc=False)
                   .limit(max_messages)
                   .execute())
        
        messages_result = await asyncio.to_thread(_get_messages)
        messages = messages_result.data or []
        
        if not messages:
            logger.warning(f"No messages found for conversation {conversation_id}")
            return
        
        # Generate summary using simplified approach (no LLM call in background task)
        # This will be enhanced with actual LLM integration in later tasks
        summary_text = self._generate_simple_summary(messages)
        
        # Store to existing conversation_summaries table
        summary_data = {
            "conversation_id": conversation_id,
            "user_id": user_id,
            "summary_text": summary_text,
            "message_count": len(messages),
            "created_at": datetime.utcnow().isoformat(),
            "metadata": {
                "max_messages_used": max_messages,
                "summary_threshold": summary_threshold,
                "generated_by": "background_task_manager"
            }
        }
        
        def _store_summary():
            return db_client.client.table("conversation_summaries").insert(summary_data).execute()
        
        result = await asyncio.to_thread(_store_summary)
        
        if not result.data:
            raise Exception("Failed to store conversation summary")
        
        logger.debug(f"Generated summary for conversation {conversation_id} with {len(messages)} messages")
    
    async def _handle_context_loading(self, task: BackgroundTask):
        """
        Handle lazy context loading for user data.
        
        Args:
            task: Context loading task
        """
        user_id = task.user_id
        conversation_id = task.conversation_id
        
        # This is a placeholder for lazy context loading
        # Will be implemented with actual context retrieval logic
        logger.debug(f"Loading context for user {user_id}, conversation {conversation_id}")
        
        # Simulate context loading work
        await asyncio.sleep(0.01)
    
    def _generate_simple_summary(self, messages: List[Dict[str, Any]]) -> str:
        """
        Generate a simple summary without LLM calls for token efficiency.
        
        Args:
            messages: List of messages to summarize
            
        Returns:
            Summary text
        """
        if not messages:
            return "Empty conversation"
        
        # Extract key information
        message_count = len(messages)
        first_message_time = messages[0].get("created_at", "")
        last_message_time = messages[-1].get("created_at", "")
        
        # Count roles
        user_messages = len([m for m in messages if m.get("role") == "user"])
        assistant_messages = len([m for m in messages if m.get("role") == "assistant"])
        
        # Simple summary
        summary = (f"Conversation with {message_count} messages "
                  f"({user_messages} user, {assistant_messages} assistant) "
                  f"from {first_message_time[:10]} to {last_message_time[:10]}")
        
        return summary
    
    def get_health_status(self) -> Dict[str, Any]:
        """
        Get current health status and statistics.
        
        Returns:
            Health status dictionary
        """
        return {
            "is_running": self.is_running,
            "active_tasks": len(self.active_tasks),
            "queue_sizes": self.health_stats['queue_sizes'].copy(),
            "total_processed": self.health_stats['total_tasks_processed'],
            "success_rate": (
                self.health_stats['successful_tasks'] / max(1, self.health_stats['total_tasks_processed'])
            ),
            "average_processing_time": self.health_stats['average_processing_time'],
            "last_health_check": self.health_stats['last_health_check'].isoformat(),
            "configuration": {
                "employee_max_messages": self.employee_max_messages,
                "customer_max_messages": self.customer_max_messages,
                "summary_threshold": self.summary_threshold,
                "max_concurrent_tasks": self.max_concurrent_tasks
            }
        }
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Get status of a specific task.
        
        Args:
            task_id: Task ID to check
            
        Returns:
            Task status dictionary or None if not found
        """
        # Check active tasks
        if task_id in self.active_tasks:
            task = self.active_tasks[task_id]
            return self._task_to_dict(task)
        
        # Check completed tasks
        for task in self.completed_tasks:
            if task.id == task_id:
                return self._task_to_dict(task)
        
        # Check failed tasks
        for task in self.failed_tasks:
            if task.id == task_id:
                return self._task_to_dict(task)
        
        # Check queued tasks
        for priority_queue in self.task_queue.values():
            for task in priority_queue:
                if task.id == task_id:
                    return self._task_to_dict(task)
        
        return None
    
    def _task_to_dict(self, task: BackgroundTask) -> Dict[str, Any]:
        """Convert BackgroundTask to dictionary for JSON serialization."""
        return {
            "id": task.id,
            "task_type": task.task_type,
            "priority": task.priority.name,
            "status": task.status.value,
            "created_at": task.created_at.isoformat(),
            "scheduled_at": task.scheduled_at.isoformat() if task.scheduled_at else None,
            "started_at": task.started_at.isoformat() if task.started_at else None,
            "completed_at": task.completed_at.isoformat() if task.completed_at else None,
            "retry_count": task.retry_count,
            "max_retries": task.max_retries,
            "last_error": task.last_error,
            "conversation_id": task.conversation_id,
            "user_id": task.user_id,
            "customer_id": task.customer_id,
            "employee_id": task.employee_id
        }

    # =================================================================
    # CONSOLIDATOR INTEGRATION METHODS
    # Migrated from ConversationConsolidator class (Task 4.11.4)
    # =================================================================

    async def get_conversation_details(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """
        Get conversation details from database.
        Migrated from ConversationConsolidator._get_conversation_details
        """
        try:
            def _get_conversation():
                return db_client.client.table("conversations").select("*").eq("id", conversation_id).single().execute()

            result = await asyncio.get_event_loop().run_in_executor(None, _get_conversation)
            
            if result.data:
                return result.data
            return None

        except Exception as e:
            logger.error(f"Error getting conversation details for {conversation_id}: {e}")
            return None

    async def get_conversation_messages(self, conversation_id: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get conversation messages from database.
        Migrated from ConversationConsolidator._get_conversation_messages
        """
        try:
            def _get_messages():
                query = db_client.client.table("messages").select("*").eq("conversation_id", conversation_id).order("created_at", desc=False)
                if limit:
                    query = query.limit(limit)
                return query.execute()

            result = await asyncio.get_event_loop().run_in_executor(None, _get_messages)
            return result.data if result.data else []

        except Exception as e:
            logger.error(f"Error getting conversation messages for {conversation_id}: {e}")
            return []

    async def get_user_context_for_new_conversation(self, user_id: str) -> Dict[str, Any]:
        """
        Get user context for new conversations using conversation summaries.
        Migrated and simplified from ConversationConsolidator.get_user_context_for_new_conversation
        """
        try:
            def _get_user_context():
                # Get latest conversation summaries for this user
                summaries = db_client.client.table("conversation_summaries").select("*").eq("user_id", user_id).order("created_at", desc=True).limit(3).execute()
                
                # Count total conversations
                conversations = db_client.client.table("conversations").select("id").eq("user_id", user_id).execute()
                
                return {
                    "summaries": summaries.data if summaries.data else [],
                    "conversation_count": len(conversations.data) if conversations.data else 0
                }

            result = await asyncio.get_event_loop().run_in_executor(None, _get_user_context)
            
            if result["summaries"]:
                # Combine recent summaries into a single context
                latest_summary = "\n\n".join([s.get("summary", "") for s in result["summaries"][:2]])
                return {
                    "has_history": True,
                    "latest_summary": latest_summary,
                    "conversation_count": result["conversation_count"]
                }
            else:
                return {
                    "has_history": False,
                    "latest_summary": "",
                    "conversation_count": 0
                }

        except Exception as e:
            logger.error(f"Error getting user context for {user_id}: {e}")
            return {"has_history": False, "latest_summary": "", "conversation_count": 0}

    async def get_conversation_summary(self, conversation_id: str) -> Optional[str]:
        """
        Get conversation summary from database.
        Migrated from ConversationConsolidator.get_conversation_summary
        """
        try:
            def _get_summary():
                return db_client.client.table("conversation_summaries").select("summary").eq("conversation_id", conversation_id).order("created_at", desc=True).limit(1).single().execute()

            result = await asyncio.get_event_loop().run_in_executor(None, _get_summary)
            return result.data.get("summary") if result.data else None

        except Exception as e:
            logger.debug(f"No summary found for conversation {conversation_id}: {e}")
            return None

    async def consolidate_conversations(self, user_id: str) -> Dict[str, Any]:
        """
        Consolidate old conversations for a user.
        Simplified version migrated from ConversationConsolidator.consolidate_conversations
        """
        try:
            # For now, just return success - full consolidation can be implemented later if needed
            # The key functionality (summary generation) is already handled by background tasks
            logger.info(f"Conversation consolidation requested for user {user_id} - handled by background tasks")
            return {"consolidated_count": 0, "message": "Consolidation handled by background task system"}

        except Exception as e:
            logger.error(f"Error in conversation consolidation for user {user_id}: {e}")
            return {"consolidated_count": 0, "error": str(e)}


# Global instance
background_task_manager = BackgroundTaskManager()
