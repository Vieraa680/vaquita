from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class MemoryRecord:
    id: str
    content: str
    category: str = "general"
    timestamp: str = ""
    tags: tuple[str, ...] = field(default_factory=tuple)
    score: float = 0.0


@runtime_checkable
class BaseMemory(Protocol):
    def store(
        self,
        content: str,
        category: str = "general",
        tags: Sequence[str] | None = None,
    ) -> str: ...

    def search(
        self,
        query: str,
        limit: int = 5,
        category: str | None = None,
    ) -> list[MemoryRecord]: ...

    def delete(self, record_id: str) -> bool: ...


@runtime_checkable
class BaseProfileStore(Protocol):
    def get_fact(self, key: str, default: Any = None) -> Any: ...

    def set_fact(self, key: str, value: Any, section: str = "facts") -> None: ...

    def remove_fact(self, key: str, section: str = "facts") -> bool: ...

    def all_facts(self) -> dict[str, Any]: ...

    def format_context(self) -> str: ...
