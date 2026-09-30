import json
from pathlib import Path
from unittest.mock import MagicMock

from midi_llm.eval.musicinstruct_runner import predictions_from_completions


def test_decode_isolates_failures(tmp_path: Path, sample_midi):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "item_id": "a",
                "midi_in": "in.mid",
                "instruction": "test",
                "split": "test",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    completions = tmp_path / "completions.jsonl"
    completions.write_text(
        "\n".join(
            [
                json.dumps({"item_id": "a", "completion": "not valid tokens"}),
                json.dumps({"item_id": "a", "completion": "also bad"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    backend = MagicMock()
    backend.decode_to_midi.side_effect = ValueError("decode failed")

    import midi_llm.eval.musicinstruct_runner as runner

    original_get_repr = runner.get_repr
    runner.get_repr = lambda _name: backend
    try:
        result = predictions_from_completions(
            manifest,
            completions,
            repr_name="remi",
            output_predictions=tmp_path / "predictions.jsonl",
            split="test",
        )
    finally:
        runner.get_repr = original_get_repr

    assert result.n_written == 0
    assert result.n_failed == 2
    assert (tmp_path / "decode_failures.jsonl").is_file()
