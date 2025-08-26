"""Type stubs for pydantic."""

from typing import Any, Dict, Type, TypeVar, Optional

_BaseModelT = TypeVar("_BaseModelT", bound="BaseModel")

class BaseModel:
    def __init__(self, **kwargs: Any) -> None: ...
    def dict(self) -> Dict[str, Any]: ...
    def json(self) -> str: ...
    @classmethod
    def parse_obj(cls: Type[_BaseModelT], obj: Any) -> _BaseModelT: ...

def Field(default: Any = ..., **kwargs: Any) -> Any: ...

__all__ = ["BaseModel", "Field"]






