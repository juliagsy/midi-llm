from pathlib import Path

import pytest

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


def test_build_syntax_shard_raises_on_empty(tmp_path: Path, miditok_available):
    if not miditok_available:
        pytest.skip("miditok not installed")

    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    with pytest.raises(RuntimeError, match="no syntax examples"):
        build_syntax_shard(empty_dir, repr_name="remi", output_path=tmp_path / "syntax.jsonl")
