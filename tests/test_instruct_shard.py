from pathlib import Path

import pytest

from midi_llm.data.instruct import build_instruct_shard


def test_build_instruct_shard_from_pilot(miditok_available, tmp_path: Path):
    if not miditok_available:
        pytest.skip("miditok not installed")

    pilot = Path(__file__).resolve().parents[2] / "musicinstruct" / "data" / "pilot" / "pilot.jsonl"
    if not pilot.is_file():
        pytest.skip("musicinstruct pilot manifest not found")

    out = tmp_path / "edit_remi.jsonl"
    count = build_instruct_shard(pilot, repr_name="remi", output_path=out, split="train")
    assert count > 0
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == count
    assert "Edit the input MIDI" in lines[0]
