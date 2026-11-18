from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
import re
import sqlite3
from typing import Any, Protocol, runtime_checkable
import uuid

from vaquita.core.providers import ToolDefinition
from vaquita.tools.base import BaseTool

DEFAULT_REMINDERS_DB_PATH = Path.home() / ".config" / "vaquita" / "reminders.db"
_RELATIVE_TIME_RE = re.compile(r"^\+?(\d+)\s*([mh])$", re.IGNORECASE)


@dataclass(frozen=True)
class Reminder:
    id: str
    message: str
    due_at: str
    status: str
    created_at: str


@runtime_checkable
class BaseReminderStore(Protocol):
    def add(self, message: str, due_at: str) -> Reminder: ...

    def list_reminders(self, status: str | None = "pending") -> list[Reminder]: ...

    def cancel(self, reminder_id: str) -> bool: ...

    def due_pending(self, now: datetime | None = None) -> list[Reminder]: ...

    def mark_done(self, reminder_id: str) -> bool: ...


def parse_due_at(raw_due: str, now: datetime | None = None) -> str:
    """Parse an ISO datetime or relative offset (+15m, +2h) into ISO-8601."""
    cleaned = raw_due.strip()
    if not cleaned:
        raise ValueError("due_at must not be empty.")

    base_now = now or datetime.now()
    rel_match = _RELATIVE_TIME_RE.match(cleaned)
    if rel_match:
        amount = int(rel_match.group(1))
        unit = rel_match.group(2).lower()
        delta = timedelta(minutes=amount) if unit == "m" else timedelta(hours=amount)
        return (base_now + delta).replace(microsecond=0).isoformat()

    try:
        parsed = datetime.fromisoformat(cleaned)
    except ValueError as exc:
        raise ValueError(
            f"Invalid due_at '{raw_due}'. Use ISO format (YYYY-MM-DDTHH:MM) or +15m/+2h."
        ) from exc
    return parsed.replace(microsecond=0).isoformat()


class SQLiteReminderStore(BaseReminderStore):
    """SQLite-backed store for local reminders and timers."""

    def __init__(self, db_path: Path | str | None = None) -> None:
        self.db_path = Path(db_path) if db_path is not None else DEFAULT_REMINDERS_DB_PATH
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS reminders (
                    id TEXT PRIMARY KEY,
                    message TEXT NOT NULL,
                    due_at TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_reminders_status_due ON reminders(status, due_at)"
            )
            conn.commit()

    @staticmethod
    def _row_to_reminder(row: sqlite3.Row) -> Reminder:
        return Reminder(
            id=str(row["id"]),
            message=str(row["message"]),
            due_at=str(row["due_at"]),
            status=str(row["status"]),
            created_at=str(row["created_at"]),
        )

    def add(self, message: str, due_at: str) -> Reminder:
        clean_msg = message.strip()
        if not clean_msg:
            raise ValueError("Reminder message must not be empty.")

        resolved_due = parse_due_at(due_at)
        reminder_id = uuid.uuid4().hex[:8]
        created_at = datetime.now().replace(microsecond=0).isoformat()

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO reminders (id, message, due_at, status, created_at)
                VALUES (?, ?, ?, 'pending', ?)
                """,
                (reminder_id, clean_msg, resolved_due, created_at),
            )
            conn.commit()

        return Reminder(
            id=reminder_id,
            message=clean_msg,
            due_at=resolved_due,
            status="pending",
            created_at=created_at,
        )

    def list_reminders(self, status: str | None = "pending") -> list[Reminder]:
        with self._get_connection() as conn:
            if status is None:
                cursor = conn.execute(
                    "SELECT id, message, due_at, status, created_at FROM reminders ORDER BY due_at ASC"
                )
            else:
                cursor = conn.execute(
                    """
                    SELECT id, message, due_at, status, created_at
                    FROM reminders
                    WHERE status = ?
                    ORDER BY due_at ASC
                    """,
                    (status.strip().lower(),),
                )
            rows = cursor.fetchall()
        return [self._row_to_reminder(row) for row in rows]

    def cancel(self, reminder_id: str) -> bool:
        with self._get_connection() as conn:
            cursor = conn.execute(
                "UPDATE reminders SET status = 'cancelled' WHERE id = ? AND status = 'pending'",
                (reminder_id.strip(),),
            )
            conn.commit()
            return cursor.rowcount > 0

    def due_pending(self, now: datetime | None = None) -> list[Reminder]:
        cutoff = (now or datetime.now()).replace(microsecond=0).isoformat()
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT id, message, due_at, status, created_at
                FROM reminders
                WHERE status = 'pending' AND due_at <= ?
                ORDER BY due_at ASC
                """,
                (cutoff,),
            )
            rows = cursor.fetchall()
        return [self._row_to_reminder(row) for row in rows]

    def mark_done(self, reminder_id: str) -> bool:
        with self._get_connection() as conn:
            cursor = conn.execute(
                "UPDATE reminders SET status = 'done' WHERE id = ? AND status = 'pending'",
                (reminder_id.strip(),),
            )
            conn.commit()
            return cursor.rowcount > 0


class SetReminderTool(BaseTool):
    def __init__(self, store: BaseReminderStore) -> None:
        self._store = store

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="set_reminder",
            description="Create a local reminder, alarm, or timer (agenda) for a due time like '+15m', '+2h', or '2026-11-20T09:00'.",
            parameters={
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "What to remind the user about.",
                    },
                    "due_at": {
                        "type": "string",
                        "description": "ISO datetime (YYYY-MM-DDTHH:MM) or relative offset (+15m, +2h).",
                    },
                },
                "required": ["message", "due_at"],
            },
        )

    def execute(self, **kwargs: Any) -> str:
        message = str(kwargs.get("message", "")).strip()
        due_at = str(kwargs.get("due_at", "")).strip()
        try:
            reminder = self._store.add(message=message, due_at=due_at)
        except ValueError as exc:
            return f"Error: {exc}"
        return f"Reminder saved [{reminder.id}] for {reminder.due_at}: {reminder.message}"


class ListRemindersTool(BaseTool):
    def __init__(self, store: BaseReminderStore) -> None:
        self._store = store

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="list_reminders",
            description="List pending or past local reminders and scheduled agenda events.",
            parameters={
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "description": "Filter by status: pending, done, cancelled, or all (default: pending).",
                    },
                },
            },
        )

    def execute(self, **kwargs: Any) -> str:
        raw_status = kwargs.get("status", "pending")
        status: str | None = str(raw_status).strip().lower() if raw_status is not None else "pending"
        if status == "all":
            status = None
        reminders = self._store.list_reminders(status=status)
        if not reminders:
            return "No reminders found."
        lines = [
            f"- [{r.id}] ({r.status}) {r.due_at}: {r.message}"
            for r in reminders
        ]
        return "\n".join(lines)


class CancelReminderTool(BaseTool):
    def __init__(self, store: BaseReminderStore) -> None:
        self._store = store

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="cancel_reminder",
            description="Cancel a pending reminder or alarm by its ID.",
            parameters={
                "type": "object",
                "properties": {
                    "reminder_id": {
                        "type": "string",
                        "description": "The ID of the reminder to cancel.",
                    },
                },
                "required": ["reminder_id"],
            },
        )

    def execute(self, **kwargs: Any) -> str:
        reminder_id = str(kwargs.get("reminder_id", "")).strip()
        if not reminder_id:
            return "Error: reminder_id must not be empty."
        if self._store.cancel(reminder_id):
            return f"Reminder '{reminder_id}' cancelled."
        return f"Reminder '{reminder_id}' not found or not pending."
