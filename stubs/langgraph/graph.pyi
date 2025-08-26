"""Type stubs for langgraph.graph."""

from typing import Any, Dict, List, Optional, Callable

START: str
END: str

class StateGraph:
    def __init__(self, state_schema: Any) -> None: ...
    def add_node(self, name: str, node: Callable[..., Any]) -> "StateGraph": ...
    def add_edge(self, start: str, end: str) -> "StateGraph": ...
    def add_conditional_edges(self, start: str, condition: Callable[..., str], path_map: Optional[Dict[str, str]] = None) -> "StateGraph": ...
    def set_entry_point(self, name: str) -> "StateGraph": ...
    def compile(self, **kwargs: Any) -> Any: ...

__all__ = ["StateGraph", "START", "END"]






