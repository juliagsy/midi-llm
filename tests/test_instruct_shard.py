import json
from pathlib import Path

import pytest

from midi_llm.data.instruct import build_instruct_shard


def test_build_instruct_shard_from_fixture(miditok_available, instruct_mini_manifest: Path, tmp_path: Path):
    if not miditok_available:
        pytest.skip("miditok not installed")

    out = tmp_path / "edit_remi.jsonl"
    count = build_instruct_shard(
        instruct_mini_manifest,
        repr_name="remi",
        output_path=out,
        split="train",
    )
    assert count == 1
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == count
    assert "Edit the input MIDI" in lines[0]


def test_build_instruct_shard_fails_on_empty(miditok_available, tmp_path: Path):
    if not miditok_available:
        pytest.skip("miditok not installed")

    manifest = tmp_path / "empty.jsonl"
    row = {
        "item_id": "x",
        "midi_in": "missing/in.mid",
        "instruction": "transpose up",
        "split": "train",
        "gold_midi": "missing/gold.mid",
    }
    manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
    out = tmp_path / "out.jsonl"

    with pytest.raises(RuntimeError, match="no instruct examples written"):
        build_instruct_shard(manifest, repr_name="remi", output_path=out, split="train")
