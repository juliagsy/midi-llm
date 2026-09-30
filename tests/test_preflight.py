import json
from pathlib import Path

import pytest

from midi_llm.infer.preflight import preflight_manifest


def test_preflight_manifest_ok(tmp_path: Path, sample_midi):
    rel = sample_midi.relative_to(tmp_path)
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "item_id": "a",
                "midi_in": str(rel),
                "instruction": "test",
                "split": "test",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report = preflight_manifest(manifest, split="test")
    assert report.ok
    assert report.n_items == 1


def test_preflight_manifest_missing_midi(tmp_path: Path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "item_id": "a",
                "midi_in": "missing.mid",
                "instruction": "test",
                "split": "test",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError, match="preflight failed"):
        preflight_manifest(manifest, split="test")
