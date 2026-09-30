"""Decode model outputs and score via musicinstruct."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from midi_llm.data.jsonl_io import COMPLETION_FIELDS, MANIFEST_FIELDS, iter_jsonl
from midi_llm.midi_repr.registry import get_repr


@dataclass
class DecodeBatchResult:
    n_written: int = 0
    n_failed: int = 0
    n_skipped: int = 0
    failures: list[dict[str, str]] = field(default_factory=list)


def predictions_from_completions(
    manifest_path: str | Path,
    completions_path: str | Path,
    *,
    repr_name: str,
    output_predictions: str | Path,
    split: str | None = "test",
    failures_path: str | Path | None = None,
) -> DecodeBatchResult:
    """Map JSONL completions (item_id + completion text) to MIDI predictions.

    Each completion line: {"item_id": "...", "completion": "<midi payload>"}
    """
    manifest = Path(manifest_path)
    backend = get_repr(repr_name)
    predictions_root = Path(output_predictions).parent
    out_dir = predictions_root / "decoded_midis"
    out_dir.mkdir(parents=True, exist_ok=True)

    items_by_id: dict[str, dict[str, Any]] = {}
    for _line_no, record in iter_jsonl(
        manifest,
        required_fields=MANIFEST_FIELDS,
        label="manifest",
    ):
        if split is None or record.get("split") == split:
            items_by_id[record["item_id"]] = record

    result = DecodeBatchResult()
    fail_path = Path(failures_path) if failures_path else predictions_root / "decode_failures.jsonl"

    with Path(output_predictions).open("w", encoding="utf-8") as dst, fail_path.open(
        "w", encoding="utf-8"
    ) as fail_writer:
        for line_no, row in iter_jsonl(
            completions_path,
            required_fields=COMPLETION_FIELDS,
            label="completion",
        ):
            item_id = row["item_id"]
            if item_id not in items_by_id:
                result.n_skipped += 1
                continue

            completion = row["completion"]
            midi_out = out_dir / f"{item_id}.mid"

            try:
                if repr_name == "abc":
                    backend.decode_to_midi(text=completion, output_path=midi_out)
                else:
                    token_ids = [int(x) for x in completion.split()]
                    backend.decode_to_midi(token_ids=token_ids, output_path=midi_out)
            except Exception as exc:  # noqa: BLE001 — isolate per-item decode failures
                result.n_failed += 1
                failure = {
                    "item_id": item_id,
                    "line_no": str(line_no),
                    "error": str(exc),
                }
                result.failures.append(failure)
                fail_writer.write(json.dumps(failure) + "\n")
                continue

            pred = {
                "item_id": item_id,
                "midi_path": str(midi_out.relative_to(predictions_root)),
            }
            if "plan" in row:
                pred["plan"] = row["plan"]
            dst.write(json.dumps(pred) + "\n")
            result.n_written += 1

    return result


def run_musicinstruct_eval(
    manifest_path: str | Path,
    predictions_path: str | Path,
    *,
    output_results: str | Path,
    split: str = "test",
    timeout_sec: int = 600,
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
        try:
            completed = subprocess.run(
                ["musicinstruct", *score_args],
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout_sec,
            )
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError(
                f"musicinstruct score exceeded {timeout_sec}s timeout"
            ) from exc
        if completed.returncode != 0:
            stderr = completed.stderr.strip() or completed.stdout.strip() or "unknown error"
            raise subprocess.CalledProcessError(
                completed.returncode,
                completed.args,
                output=stderr,
            )
    else:
        try:
            from musicinstruct.cli import main as mi_main
        except ImportError as exc:
            raise FileNotFoundError(
                "musicinstruct not installed; pip install -e ../musicinstruct"
            ) from exc
        exit_code = mi_main(score_args)
        if exit_code != 0:
            raise subprocess.CalledProcessError(exit_code, ["musicinstruct", *score_args])
    return json.loads(Path(output_results).read_text(encoding="utf-8"))
