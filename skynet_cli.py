# skynet_cli.py
from __future__ import annotations

import argparse
import os
from typing import List

from google import genai
from google.genai import types

from repo_reader.github_loader import load_github_repo_zip
from repo_reader.chunker import chunk_text
from repo_reader.index_sqlite import ensure_db, clear_index, upsert_chunks, DB_PATH_DEFAULT
from repo_reader.retriever import search_chunks


# --------------- Skynet System (สั้น แต่คม) ---------------
SKYNET_SYSTEM = r"""
You are Skynet, the strategic co-architect AI for ElmatadorZ.

Rules:
- Use First Principles + System Thinking.
- When repo context is provided, cite file paths and line ranges.
- If unsure, say uncertainty and propose verification steps.
- Be concise and actionable.
""".strip()


def format_repo_context(chunks) -> str:
    blocks: List[str] = []
    for c in chunks:
        blocks.append(
            f"FILE: {c.path} (lines {c.start_line}-{c.end_line})\n"
            f"{c.text}\n"
        )
    return "\n---\n".join(blocks)


def cmd_ingest(args):
    repo = args.repo  # "ElmatadorZ/Skynet-Prototype"
    ref = args.ref
    token = os.getenv("GITHUB_TOKEN")  # optional
    db_path = args.db

    files = load_github_repo_zip(repo=repo, ref=ref, token=token)

    conn = ensure_db(db_path)
    if args.clear:
        clear_index(conn)

    rows = []
    for f in files:
        chunks = chunk_text(f.path, f.text, max_lines=args.max_lines, overlap=args.overlap)
        for ch in chunks:
            rows.append((ch.chunk_id, ch.path, ch.start_line, ch.end_line, ch.text))

    upsert_chunks(conn, rows)
    print(f"[OK] Ingested {len(files)} files, {len(rows)} chunks into {db_path}")


def cmd_ask(args):
    client = genai.Client()  # uses GEMINI_API_KEY
    model_id = args.model
    db_path = args.db

    conn = ensure_db(db_path)
    hits = search_chunks(conn, args.question, k=args.k)

    repo_context = format_repo_context(hits) if hits else "(No repo context found.)"

    user_prompt = f"""
[REPO_CONTEXT]
{repo_context}
[/REPO_CONTEXT]

User question:
{args.question}

Answer requirements:
- If you reference repo code, mention FILE + line range.
- Provide next steps as bullet points.
""".strip()

    config = types.GenerateContentConfig(
        system_instruction=SKYNET_SYSTEM,
        temperature=0.4,
        max_output_tokens=1200,
    )

    resp = client.models.generate_content(
        model=model_id,
        contents=[types.Content(role="user", parts=[types.Part.from_text(text=user_prompt)])],
        config=config,
    )

    print(resp.text or "(No output)")


def main():
    p = argparse.ArgumentParser(description="Skynet CLI (Repo Reader Edition)")
    sub = p.add_subparsers(dest="cmd", required=True)

    # ingest
    pi = sub.add_parser("ingest", help="Download repo zip and build local search index")
    pi.add_argument("--repo", required=True, help='GitHub repo "owner/name"')
    pi.add_argument("--ref", default="main", help="Branch/tag (default: main)")
    pi.add_argument("--db", default=DB_PATH_DEFAULT, help="SQLite index path")
    pi.add_argument("--clear", action="store_true", help="Clear existing index")
    pi.add_argument("--max-lines", type=int, default=120)
    pi.add_argument("--overlap", type=int, default=20)
    pi.set_defaults(func=cmd_ingest)

    # ask
    pa = sub.add_parser("ask", help="Ask Skynet with repo context")
    pa.add_argument("question", help="Question for Skynet")
    pa.add_argument("--model", default=os.getenv("SKYNET_GEMINI_MODEL", "gemini-2.0-flash"))
    pa.add_argument("--db", default=DB_PATH_DEFAULT)
    pa.add_argument("-k", type=int, default=8, help="Top-k chunks to retrieve")
    pa.set_defaults(func=cmd_ask)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
