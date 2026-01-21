# repo_reader/chunker.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class Chunk:
    chunk_id: str
    path: str
    start_line: int
    end_line: int
    text: str


def chunk_text(path: str, text: str, max_lines: int = 120, overlap: int = 20) -> List[Chunk]:
    lines = text.splitlines()
    chunks: List[Chunk] = []
    i = 0
    n = len(lines)

    def make_id(p: str, s: int, e: int) -> str:
        return f"{p}:{s}-{e}"

    while i < n:
        start = i
        end = min(i + max_lines, n)
        chunk_lines = lines[start:end]
        ctext = "\n".join(chunk_lines)

        chunks.append(
            Chunk(
                chunk_id=make_id(path, start + 1, end),
                path=path,
                start_line=start + 1,
                end_line=end,
                text=ctext,
            )
        )

        if end == n:
            break
        i = end - overlap

    return chunks
