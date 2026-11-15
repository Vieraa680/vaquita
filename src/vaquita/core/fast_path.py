from typing import Protocol, runtime_checkable

from vaquita.core.router import IntentRoute


@runtime_checkable
class BaseFastPath(Protocol):
    def try_handle(
        self,
        user_input: str,
        route: IntentRoute | None = None,
    ) -> str | None:
        """Return an instant deterministic response if handled locally, else None."""
        ...
