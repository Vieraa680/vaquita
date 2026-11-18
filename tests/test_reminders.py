from datetime import datetime, timedelta
from pathlib import Path

from vaquita.tools.reminders import (
    CancelReminderTool,
    ListRemindersTool,
    SQLiteReminderStore,
    SetReminderTool,
    parse_due_at,
)


def test_parse_due_at_handles_relative_and_iso() -> None:
    base = datetime(2026, 11, 17, 10, 0, 0)
    assert parse_due_at("+15m", now=base) == (base + timedelta(minutes=15)).isoformat()
    assert parse_due_at("2h", now=base) == (base + timedelta(hours=2)).isoformat()
    assert parse_due_at("2026-11-18T09:30") == "2026-11-18T09:30:00"


def test_sqlite_reminder_store_crud_and_due(tmp_path: Path) -> None:
    store = SQLiteReminderStore(db_path=tmp_path / "reminders.db")

    r1 = store.add("Tomar agua", "2026-11-17T10:00:00")
    r2 = store.add("Reunion", "2026-11-17T18:00:00")

    pending = store.list_reminders(status="pending")
    assert [r.id for r in pending] == [r1.id, r2.id]

    due = store.due_pending(now=datetime(2026, 11, 17, 12, 0, 0))
    assert [r.id for r in due] == [r1.id]

    assert store.mark_done(r1.id) is True
    assert store.cancel(r2.id) is True
    assert store.cancel(r2.id) is False
    assert store.list_reminders(status="pending") == []
    assert len(store.list_reminders(status=None)) == 2


def test_reminder_tools_flow(tmp_path: Path) -> None:
    store = SQLiteReminderStore(db_path=tmp_path / "reminders.db")
    set_tool = SetReminderTool(store)
    list_tool = ListRemindersTool(store)
    cancel_tool = CancelReminderTool(store)

    bad = set_tool.execute(message="Test", due_at="manana")
    assert bad.startswith("Error:")

    ok = set_tool.execute(message="Revisar logs", due_at="2026-11-18T08:00")
    assert "Reminder saved" in ok
    listed = list_tool.execute()
    assert "Revisar logs" in listed

    reminder_id = store.list_reminders()[0].id
    assert f"Reminder '{reminder_id}' cancelled." == cancel_tool.execute(reminder_id=reminder_id)
    assert list_tool.execute() == "No reminders found."
