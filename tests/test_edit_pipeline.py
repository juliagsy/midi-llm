import json
from pathlib import Path

from midi_llm.eval.edit_pipeline import copy_source_as_baseline
from midi_llm.infer.prompts import build_edit_prompt, iter_manifest_records


def test_build_edit_prompt(sample_midi, miditok_available, tmp_path: Path):
    if not miditok_available:
        import pytest

        pytest.skip("miditok not installed")

    manifest = tmp_path / "mini.jsonl"
    record = {
        "item_id": "t1",
        "split": "test",
        "instruction": "Transpose up 2 semitones.",
        "midi_in": "in.mid",
        "gold_midi": "out.mid",
    }
    (tmp_path / "in.mid").write_bytes(sample_midi.read_bytes())
    manifest.write_text(json.dumps(record) + "\n", encoding="utf-8")

    item_id, model_input, plain = build_edit_prompt(manifest, record, repr_name="remi")
    assert item_id == "t1"
    assert "Transpose up 2" in plain
    assert "Output MIDI" in plain
    assert len(model_input) >= len(plain)


def test_copy_source_baseline(sample_midi, tmp_path: Path):
    manifest = tmp_path / "mini.jsonl"
    record = {
        "item_id": "t1",
        "split": "test",
        "midi_in": "in.mid",
        "instruction": "x",
    }
    (tmp_path / "in.mid").write_bytes(sample_midi.read_bytes())
    manifest.write_text(json.dumps(record) + "\n", encoding="utf-8")

    preds = tmp_path / "preds.jsonl"
    result = copy_source_as_baseline(manifest, preds, split="test")
    assert result.n_written == 1
    assert result.n_failed == 0
    row = json.loads(preds.read_text(encoding="utf-8").strip())
    assert row["item_id"] == "t1"


def test_copy_source_isolates_missing_midi(sample_midi, tmp_path: Path):
    manifest = tmp_path / "mini.jsonl"
    rows = [
        {
            "item_id": "ok",
            "split": "test",
            "midi_in": "in.mid",
            "instruction": "x",
        },
        {
            "item_id": "bad",
            "split": "test",
            "midi_in": "missing.mid",
            "instruction": "x",
        },
    ]
    (tmp_path / "in.mid").write_bytes(sample_midi.read_bytes())
    manifest.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    preds = tmp_path / "preds.jsonl"
    result = copy_source_as_baseline(manifest, preds, split="test")
    assert result.n_written == 1
    assert result.n_failed == 1
    assert (tmp_path / "copy_source_failures.jsonl").is_file()


def test_copy_source_raises_when_all_missing(tmp_path: Path):
    import pytest

    manifest = tmp_path / "mini.jsonl"
    record = {
        "item_id": "bad",
        "split": "test",
        "midi_in": "missing.mid",
        "instruction": "x",
    }
    manifest.write_text(json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="zero predictions"):
        copy_source_as_baseline(manifest, tmp_path / "preds.jsonl", split="test")


def test_iter_manifest_records(tmp_path: Path):
    manifest = tmp_path / "m.jsonl"
    base = {"midi_in": "in.mid", "instruction": "test"}
    manifest.write_text(
        json.dumps({"item_id": "a", "split": "train", **base})
        + "\n"
        + json.dumps({"item_id": "b", "split": "test", **base})
        + "\n",
        encoding="utf-8",
    )
    ids = [r["item_id"] for r in iter_manifest_records(manifest, split="test")]
    assert ids == ["b"]
