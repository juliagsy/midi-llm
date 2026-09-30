from pathlib import Path

from midi_llm.data.syntax import build_syntax_shard


def test_build_syntax_shard(sample_midi: Path, tmp_path: Path, miditok_available):
    if not miditok_available:
        import pytest

        pytest.skip("miditok not installed")

    out = tmp_path / "syntax.jsonl"
    count = build_syntax_shard(sample_midi.parent, repr_name="remi", output_path=out, max_files=1)
    assert count == 1
    line = out.read_text(encoding="utf-8").strip()
    assert "completion" in line
