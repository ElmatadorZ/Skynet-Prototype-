# repo_reader/retriever.py
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import List


@dataclass
class RetrievedChunk:
    chunk_id: str
    path: str
    start_line: int
    end_line: int
    text: str
    score: float


def search_chunks(conn: sqlite3.Connection, query: str, k: int = 8) -> List[RetrievedChunk]:
    # bm25() available for FTS5 ranking in many builds
    # If bm25 isn't available in your environment, we fallback to simple match ordering.
    try:
        rows = conn.execute(
            """
            SELECT chunk_id, path, start_line, end_line, content, bm25(chunks) as score
            FROM chunks
            WHERE chunks MATCH ?
            ORDER BY score
            LIMIT ?;
            """,
            (query, k),
        ).fetchall()
        return [
            RetrievedChunk(r[0], r[1], int(r[2]), int(r[3]), r[4], float(r[5]))
            for r in rows
        ]
    except sqlite3.OperationalError:
        rows = conn.execute(
            """
            SELECT chunk_id, path, start_line, end_line, content
            FROM chunks
            WHERE chunks MATCH ?
            LIMIT ?;
            """,
            (query, k),
        ).fetchall()
        return [
            RetrievedChunk(r[0], r[1], int(r[2]), int(r[3]), r[4], 0.0)
            for r in rows
        ]
