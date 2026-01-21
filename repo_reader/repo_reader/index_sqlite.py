# repo_reader/index_sqlite.py
from __future__ import annotations

import os
import sqlite3
from typing import Iterable, Tuple

DB_PATH_DEFAULT = ".skynet/repo_index.sqlite"


def ensure_db(db_path: str = DB_PATH_DEFAULT) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("""
    CREATE VIRTUAL TABLE IF NOT EXISTS chunks USING fts5(
        chunk_id UNINDEXED,
        path,
        start_line UNINDEXED,
        end_line UNINDEXED,
        content
    );
    """)
    return conn


def clear_index(conn: sqlite3.Connection) -> None:
    conn.execute("DELETE FROM chunks;")
    conn.commit()


def upsert_chunks(conn: sqlite3.Connection, rows: Iterable[Tuple[str, str, int, int, str]]) -> None:
    conn.executemany(
        "INSERT INTO chunks(chunk_id, path, start_line, end_line, content) VALUES (?,?,?,?,?)",
        list(rows),
    )
    conn.commit()
