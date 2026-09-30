import json
from pathlib import Path

import pytest

from midi_llm.train.shard_meta import infer_repr_name_from_shard, resolve_repr_name


def _write_shard(path: Path, repr_name: str) -> None:
    record = {
        "task": "edit",
        "repr_name": repr_name,
        "prompt": "edit",
        "completion": "1 2 3",
    }
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")


def test_infer_repr_name_from_shard(tmp_path: Path):
    shard = tmp_path / "edit.jsonl"
    _write_shard(shard, "remi")
    assert infer_repr_name_from_shard(shard) == "remi"


def test_resolve_repr_name_from_shard_only(tmp_path: Path):
    shard = tmp_path / "edit.jsonl"
    _write_shard(shard, "remi")
    assert resolve_repr_name(shard, None) == "remi"


def test_resolve_repr_name_requires_metadata(tmp_path: Path):
    shard = tmp_path / "bad.jsonl"
    shard.write_text(
        json.dumps({"prompt": "x", "completion": "y", "task": "edit", "repr_name": ""}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="requires --repr"):
        resolve_repr_name(shard, None)


def test_resolve_repr_name_conflict(tmp_path: Path):
    shard = tmp_path / "edit.jsonl"
    _write_shard(shard, "remi")
    with pytest.raises(ValueError, match="conflicts"):
        resolve_repr_name(shard, "octuple")
