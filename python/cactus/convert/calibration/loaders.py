from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def read_jsonl(path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def load_manifest(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def language_text(row: dict[str, Any]) -> str:
    if "messages" in row:
        return "\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in row["messages"])
    if "prompt_text" in row or "completion_text" in row:
        return (row.get("prompt_text", "") + row.get("completion_text", "")).strip()
    return (row.get("prompt", "") + row.get("completion", "")).strip()


def embedding_texts(row: dict[str, Any]) -> list[str]:
    if "texts" in row:
        return [str(x) for x in row["texts"]]
    if "text" in row:
        return [str(row["text"])]
    return []


def load_text_calibration(manifest: dict[str, Any], limits: dict[str, int | None]) -> list[str]:
    texts: list[str] = []
    language = manifest.get("language")
    if language:
        path = language["path"] if isinstance(language, dict) else language
        texts.extend(language_text(r) for r in read_jsonl(path, limits.get("language")))
    embedding = manifest.get("embedding")
    if embedding:
        path = embedding["path"] if isinstance(embedding, dict) else embedding
        for row in read_jsonl(path, limits.get("embedding")):
            texts.extend(embedding_texts(row))
    return [t for t in texts if t]
