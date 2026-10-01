import json
from pathlib import Path

from midi_llm.eval.musicinstruct_runner import predictions_from_completions


def test_predictions_from_completions_deduplicates_item_ids(tmp_path: Path, sample_midi):
    manifest = tmp_path / "manifest.jsonl"
    record = {
        "item_id": "t1",
        "split": "test",
        "instruction": "x",
        "midi_in": "in.mid",
    }
    (tmp_path / "in.mid").write_bytes(sample_midi.read_bytes())
    manifest.write_text(json.dumps(record) + "\n", encoding="utf-8")

    completions = tmp_path / "completions.jsonl"
    rows = [
        {"item_id": "t1", "completion": "not-valid-midi"},
        {"item_id": "t1", "completion": "also-not-valid"},
    ]
    completions.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    preds = tmp_path / "preds.jsonl"
    result = predictions_from_completions(
        manifest,
        completions,
        repr_name="remi",
        output_predictions=preds,
        split="test",
    )
    assert result.n_duplicates == 1
    assert result.n_failed == 1
    assert result.n_written == 0
