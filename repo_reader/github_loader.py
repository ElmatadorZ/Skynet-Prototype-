# repo_reader/github_loader.py
from __future__ import annotations

import io
import os
import zipfile
import urllib.request
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class RepoFile:
    path: str
    text: str


def _download_zip(url: str) -> bytes:
    with urllib.request.urlopen(url) as resp:
        return resp.read()


def load_github_repo_zip(repo: str, ref: str = "main", token: Optional[str] = None) -> List[RepoFile]:
    """
    repo: "ElmatadorZ/Skynet-Prototype" (owner/name)
    ref : branch/tag/commit เช่น "main"
    token: optional GitHub token for private repos (not required for public)
    """
    # GitHub zip URL format
    zip_url = f"https://github.com/{repo}/archive/refs/heads/{ref}.zip"

    req = urllib.request.Request(zip_url)
    if token:
        req.add_header("Authorization", f"token {token}")
    with urllib.request.urlopen(req) as resp:
        zip_bytes = resp.read()

    zf = zipfile.ZipFile(io.BytesIO(zip_bytes))
    files: List[RepoFile] = []

    for name in zf.namelist():
        # ignore directories
        if name.endswith("/"):
            continue

        # filter: keep only text-y files
        lower = name.lower()
        if not any(lower.endswith(ext) for ext in [".py", ".md", ".txt", ".json", ".yaml", ".yml"]):
            continue

        # read file
        raw = zf.read(name)
        try:
            text = raw.decode("utf-8", errors="replace")
        except Exception:
            continue

        # strip the top folder "repo-ref/"
        parts = name.split("/", 1)
        relpath = parts[1] if len(parts) > 1 else name

        files.append(RepoFile(path=relpath, text=text))

    return files
