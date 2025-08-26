"""Type stubs for langchain_community.utilities."""

from typing import Any, List, Optional

class SQLDatabase:
    def __init__(self, **kwargs: Any) -> None: ...
    def run(self, query: str) -> str: ...
    def get_table_info(self, table_names: Optional[List[str]] = None) -> str: ...

__all__ = ["SQLDatabase"]






