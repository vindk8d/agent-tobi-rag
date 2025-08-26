"""Type stubs for langsmith."""

from typing import Any, Callable, Optional

def traceable(name: Optional[str] = None) -> Callable[[Callable[..., Any]], Callable[..., Any]]: ...

__all__ = ["traceable"]






