"""Type stubs for langchain_core.messages."""

from typing import Any, Dict, List, Optional, Union

class BaseMessage:
    content: str
    type: str
    role: str
    def __init__(self, content: str, **kwargs: Any) -> None: ...

class HumanMessage(BaseMessage):
    def __init__(self, content: str, **kwargs: Any) -> None: ...

class AIMessage(BaseMessage):
    def __init__(self, content: str, **kwargs: Any) -> None: ...

class SystemMessage(BaseMessage):
    def __init__(self, content: str, **kwargs: Any) -> None: ...

class ToolMessage(BaseMessage):
    def __init__(self, content: str, tool_call_id: str, **kwargs: Any) -> None: ...

__all__ = ["BaseMessage", "HumanMessage", "AIMessage", "SystemMessage", "ToolMessage"]






