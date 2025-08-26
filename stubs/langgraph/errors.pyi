"""Type stubs for langgraph.errors."""

class GraphInterrupt(Exception):
    def __init__(self, message: str = "") -> None: ...

__all__ = ["GraphInterrupt"]






