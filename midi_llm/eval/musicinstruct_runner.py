"""Decode model outputs and score via musicinstruct."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from midi_llm.midi_repr.registry import get_repr


def predictions_from_completions(
    manifest_path: str | Path,
    completions_path: str | Path,
    *,
    repr_name: str,
    output_predictions: str | Path,
    split: str | None = "test",
) -> int:
    """Map JSONL completions (item_id + completion text) to MIDI predictions.

    Each completion line: {"item_id": "...", "completion": "<midi payload>"}
    """
    manifest = Path(manifest_path)
    root = manifest.parent
    backend = get_repr(repr_name)
    out_dir = Path(output_predictions).parent / "decoded_midis"
    out_dir.mkdir(parents=True, exist_ok=True)

    items_by_id = {}
    with manifest.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if split is None or record.get("split") == split:
                items_by_id[record["item_id"]] = record

    count = 0
    with Path(completions_path).open(encoding="utf-8") as src, Path(output_predictions).open(
        "w", encoding="utf-8"
    ) as dst:
        for line in src:
            if not line.strip():
                continue
            row = json.loads(line)
            item_id = row["item_id"]
            if item_id not in items_by_id:
                continue
            completion = row["completion"]
            midi_out = out_dir / f"{item_id}.mid"

            if repr_name == "abc":
                backend.decode_to_midi(text=completion, output_path=midi_out)
            else:
                token_ids = [int(x) for x in completion.split()]
                backend.decode_to_midi(token_ids=token_ids, output_path=midi_out)

            pred = {"item_id": item_id, "midi_path": str(midi_out.relative_to(Path(output_predictions).parent))}
            if "plan" in row:
                pred["plan"] = row["plan"]
            dst.write(json.dumps(pred) + "\n")
            count += 1
    return count


def run_musicinstruct_eval(
    manifest_path: str | Path,
    predictions_path: str | Path,
    *,
    output_results: str | Path,
    split: str = "test",
) -> dict[str, Any]:
    """Invoke `musicinstruct score` CLI and load JSON results."""
    score_args = [
        "score",
        str(manifest_path),
        str(predictions_path),
        "--output",
        str(output_results),
        "--split",
        split,
    ]
    if shutil.which("musicinstruct"):
        subprocess.run(["musicinstruct", *score_args], check=True)
    else:
        try:
            from musicinstruct.cli import main as mi_main
        except ImportError as exc:
            raise FileNotFoundError(
                "musicinstruct not installed; pip install -e ../musicinstruct"
            ) from exc
        mi_main(score_args)
    return json.loads(Path(output_results).read_text(encoding="utf-8"))
