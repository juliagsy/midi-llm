"""End-to-end edit eval: completions → MIDI predictions → musicinstruct score."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from midi_llm.data.jsonl_io import MANIFEST_FIELDS, iter_jsonl
from midi_llm.eval.musicinstruct_runner import predictions_from_completions, run_musicinstruct_eval
from midi_llm.infer.edit import run_edit_inference


def decode_completions_to_predictions(
    manifest_path: str | Path,
    completions_path: str | Path,
    *,
    repr_name: str,
    output_predictions: str | Path,
    split: str = "test",
):
    return predictions_from_completions(
        manifest_path,
        completions_path,
        repr_name=repr_name,
        output_predictions=output_predictions,
        split=split,
    )


def run_edit_eval(
    manifest_path: str | Path,
    output_dir: str | Path,
    *,
    repr_name: str,
    split: str = "test",
    model_name: str | None = None,
    adapter_path: str | Path | None = None,
    max_items: int | None = None,
    max_new_tokens: int = 512,
    skip_infer: bool = False,
    completions_path: str | Path | None = None,
    joint_threshold: float = 0.9,
) -> dict[str, Any]:
    """Infer (optional), decode, and score against MIDI-Instruct."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if skip_infer:
        if completions_path is None:
            completions_path = out / "completions.jsonl"
        if not Path(completions_path).is_file():
            raise FileNotFoundError(f"completions not found: {completions_path}")
    else:
        meta = run_edit_inference(
            manifest_path,
            out,
            repr_name=repr_name,
            model_name=model_name,
            adapter_path=adapter_path,
            split=split,
            max_items=max_items,
            max_new_tokens=max_new_tokens,
        )
        completions_path = meta["completions_path"]

    predictions_path = out / "predictions.jsonl"
    decode_result = decode_completions_to_predictions(
        manifest_path,
        completions_path,
        repr_name=repr_name,
        output_predictions=predictions_path,
        split=split,
    )

    results_path = out / "results.json"
    try:
        results = run_musicinstruct_eval(
            manifest_path,
            predictions_path,
            output_results=results_path,
            split=split,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        # musicinstruct not installed — write decode-only summary
        results = {
            "error": "musicinstruct CLI unavailable; install ../musicinstruct",
            "n_predictions_decoded": decode_result.n_written,
            "n_decode_failed": decode_result.n_failed,
        }
        results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    summary = {
        "repr_name": repr_name,
        "split": split,
        "n_predictions_decoded": decode_result.n_written,
        "n_decode_failed": decode_result.n_failed,
        "n_decode_skipped": decode_result.n_skipped,
        "decode_failures_path": str(out / "decode_failures.jsonl"),
        "completions_path": str(completions_path),
        "predictions_path": str(predictions_path),
        "results_path": str(results_path),
        "overall": results.get("overall"),
    }
    (out / "eval_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def copy_source_as_baseline(
    manifest_path: str | Path,
    output_predictions: str | Path,
    *,
    split: str = "test",
) -> int:
    """Zero-edit baseline: predict midi_in unchanged (sanity / lower bound)."""
    manifest = Path(manifest_path)
    root = manifest.parent
    count = 0
    out_dir = Path(output_predictions).parent / "copy_source_midis"
    out_dir.mkdir(parents=True, exist_ok=True)

    with Path(output_predictions).open("w", encoding="utf-8") as dst:
        for _line_no, record in iter_jsonl(
            manifest,
            required_fields=MANIFEST_FIELDS,
            label="manifest",
        ):
            if split is not None and record.get("split") != split:
                continue
            src_midi = root / record["midi_in"]
            dest = out_dir / f"{record['item_id']}.mid"
            shutil.copy2(src_midi, dest)
            rel = dest.relative_to(Path(output_predictions).parent)
            dst.write(json.dumps({"item_id": record["item_id"], "midi_path": str(rel)}) + "\n")
            count += 1
    return count
