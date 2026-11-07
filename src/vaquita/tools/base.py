from typing import Any, Protocol, runtime_checkable

from vaquita.core.providers import ToolDefinition


@runtime_checkable
class BaseTool(Protocol):
    @property
    def definition(self) -> ToolDefinition: ...

    def execute(self, **kwargs: Any) -> str: ...
