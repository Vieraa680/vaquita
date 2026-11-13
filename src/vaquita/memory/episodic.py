from collections.abc import Sequence
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sqlite3
import uuid

from vaquita.memory.base import BaseMemory, MemoryRecord
from vaquita.neural.intent import tokenize

DEFAULT_MEMORY_DB_PATH = Path.home() / ".config" / "vaquita" / "memory.db"


class EpisodicMemoryStore(BaseMemory):
    """SQLite-backed episodic memory store with relevance scoring."""

    def __init__(self, db_path: Path | str | None = None) -> None:
        self.db_path = Path(db_path) if db_path is not None else DEFAULT_MEMORY_DB_PATH
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
                CREATE TABLE IF NOT EXISTS episodes (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    category TEXT NOT NULL DEFAULT 'general',
                    tags TEXT NOT NULL DEFAULT '[]',
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_episodes_category ON episodes(category)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_episodes_created_at ON episodes(created_at)"
            )
            conn.commit()

    def store(
        self,
        content: str,
        category: str = "general",
        tags: Sequence[str] | None = None,
    ) -> str:
        record_id = uuid.uuid4().hex[:12]
        now = datetime.now(timezone.utc).isoformat()
        tags_json = json.dumps(list(tags or []))

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO episodes (id, content, category, tags, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (record_id, content.strip(), category.strip().lower(), tags_json, now),
            )
            conn.commit()

        return record_id

    def delete(self, record_id: str) -> bool:
        with self._get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM episodes WHERE id = ?",
                (record_id,),
            )
            conn.commit()
            return cursor.rowcount > 0

    def search(
        self,
        query: str,
        limit: int = 5,
        category: str | None = None,
    ) -> list[MemoryRecord]:
        query_tokens = sorted(set(tokenize(query)))
        if not query_tokens:
            return []

        where_clauses: list[str] = []
        params: list[str] = []

        if category:
            where_clauses.append("category = ?")
            params.append(category.strip().lower())

        like_conditions = ["content LIKE ?" for _ in query_tokens]
        where_clauses.append(f"({' OR '.join(like_conditions)})")
        params.extend([f"%{token}%" for token in query_tokens])

        sql = "SELECT id, content, category, tags, created_at FROM episodes"
        if where_clauses:
            sql += " WHERE " + " AND ".join(where_clauses)

        with self._get_connection() as conn:
            cursor = conn.execute(sql, params)
            rows = cursor.fetchall()

        scored_records: list[MemoryRecord] = []
        for row in rows:
            content = str(row["content"])
            doc_tokens = tokenize(content)
            if not doc_tokens:
                continue

            matched = sum(1 for token in query_tokens if token in doc_tokens)
            if matched == 0:
                continue

            # Term overlap score normalized by token lengths
            overlap_score = matched / math.sqrt(len(query_tokens) * len(doc_tokens))
            tags_list = json.loads(row["tags"]) if row["tags"] else []

            record = MemoryRecord(
                id=str(row["id"]),
                content=content,
                category=str(row["category"]),
                timestamp=str(row["created_at"]),
                tags=tuple(str(t) for t in tags_list),
                score=round(overlap_score, 4),
            )
            scored_records.append(record)

        scored_records.sort(key=lambda rec: rec.score, reverse=True)
        return scored_records[: max(1, limit)]
