from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class IntentRoute:
    intent: str
    confidence: float
    is_confident: bool
    probabilities: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class BaseIntentRouter(Protocol):
    def route(self, user_input: str) -> IntentRoute: ...
