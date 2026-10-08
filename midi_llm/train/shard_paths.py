"""Path helpers for SFT JSONL shards."""

from __future__ import annotations

from pathlib import Path


def infer_validation_shard(train_shard: str | Path) -> Path | None:
    """Return sibling ``*_validation.jsonl`` for ``*_train.jsonl`` if it exists."""
    train = Path(train_shard)
    name = train.name
    if "_train." not in name:
        return None
    candidate = train.with_name(name.replace("_train.", "_validation.", 1))
    return candidate if candidate.is_file() else None
