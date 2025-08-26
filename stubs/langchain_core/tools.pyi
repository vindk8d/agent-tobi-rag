"""Type stubs for langchain_core.tools."""

from typing import Any, Callable, Dict, Optional

def tool(func: Callable[..., Any], name: Optional[str] = None, description: Optional[str] = None) -> Any: ...

__all__ = ["tool"]






