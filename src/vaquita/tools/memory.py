from collections.abc import Sequence
from typing import Any

from vaquita.core.providers import ToolDefinition
from vaquita.memory.base import BaseMemory, BaseProfileStore
from vaquita.tools.base import BaseTool


class RememberFactTool(BaseTool):
    """Tool to persist user identity, preferences, or environment facts."""

    def __init__(self, profile_store: BaseProfileStore) -> None:
        self._store = profile_store

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="remember_fact",
            description=(
                "Store or update a persistent fact, preference, or project detail "
                "about the user in long-term memory."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": "Identifier or property name (e.g. 'editor', 'favorite_shell', 'project_goal').",
                    },
                    "value": {
                        "type": "string",
                        "description": "The fact or value to remember.",
                    },
                    "section": {
                        "type": "string",
                        "enum": ["profile", "preferences", "projects", "facts"],
                        "description": "Section where the fact belongs (default: 'facts').",
                    },
                },
                "required": ["key", "value"],
            },
        )

    def execute(self, **kwargs: Any) -> str:
        key = str(kwargs.get("key", "")).strip()
        value = kwargs.get("value")
        section = str(kwargs.get("section", "facts")).strip().lower()

        if not key:
            return "Error: 'key' cannot be empty."
        if value is None:
            return "Error: 'value' cannot be None."

        self._store.set_fact(key=key, value=value, section=section)
        return f"Fact remembered: [{section}] {key} = {value}"


class SaveEpisodeTool(BaseTool):
    """Tool to record episodic notes, logs, or past events."""

    def __init__(self, memory_store: BaseMemory) -> None:
        self._store = memory_store

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="save_episode",
            description="Record an episodic note, past event, or conversation log with timestamp and tags.",
            parameters={
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "Summary or detail of the event or fact to record.",
                    },
                    "category": {
                        "type": "string",
                        "description": "Optional category tag (e.g. 'work', 'system', 'personal').",
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of searchable keyword tags.",
                    },
                },
                "required": ["content"],
            },
        )

    def execute(self, **kwargs: Any) -> str:
        content = str(kwargs.get("content", "")).strip()
        category = str(kwargs.get("category", "general")).strip()
        tags_raw = kwargs.get("tags")
        tags: Sequence[str] = (
            [str(t) for t in tags_raw] if isinstance(tags_raw, (list, tuple)) else []
        )

        if not content:
            return "Error: 'content' cannot be empty."

        record_id = self._store.store(content=content, category=category, tags=tags)
        return f"Episode recorded (id={record_id}): {content}"


class RecallMemoryTool(BaseTool):
    """Tool to search long-term episodic memory for past notes and events."""

    def __init__(self, memory_store: BaseMemory) -> None:
        self._store = memory_store

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="recall_memory",
            description="Search long-term episodic memory for past events, notes, or facts matching keywords.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Keywords or phrase to search past memories for.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of records to return (default: 5).",
                    },
                },
                "required": ["query"],
            },
        )

    def execute(self, **kwargs: Any) -> str:
        query = str(kwargs.get("query", "")).strip()
        limit_val = kwargs.get("limit", 5)
        try:
            limit = int(limit_val)
        except (ValueError, TypeError):
            limit = 5

        if not query:
            return "Error: 'query' cannot be empty."

        records = self._store.search(query=query, limit=limit)
        if not records:
            return "No matching memories found."

        lines: list[str] = [f"Found {len(records)} relevant memories:"]
        for idx, rec in enumerate(records, 1):
            tags_str = f" [tags: {', '.join(rec.tags)}]" if rec.tags else ""
            lines.append(
                f"{idx}. ({rec.timestamp[:10]}) [{rec.category}] {rec.content}{tags_str} (score: {rec.score:.2f})"
            )

        return "\n".join(lines)
