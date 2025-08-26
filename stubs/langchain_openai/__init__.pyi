"""Type stubs for langchain_openai."""

from typing import Any, Dict, List, Optional, Union

class ChatOpenAI:
    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        api_key: Optional[str] = None,
        **kwargs: Any
    ) -> None: ...
    
    def bind_tools(self, tools: List[Any]) -> "ChatOpenAI": ...
    async def ainvoke(self, messages: List[Any]) -> Any: ...
    def invoke(self, messages: List[Any]) -> Any: ...

__all__ = ["ChatOpenAI"]






