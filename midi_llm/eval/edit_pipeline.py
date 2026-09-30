"""End-to-end edit eval: completions → MIDI predictions → musicinstruct score."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from midi_llm.data.jsonl_io import MANIFEST_FIELDS, iter_jsonl
from midi_llm.eval.musicinstruct_runner import predictions_from_completions, run_musicinstruct_eval
from midi_llm.infer.edit import run_edit_inference


@dataclass
class CopySourceResult:
    n_written: int = 0
    n_failed: int = 0
    n_skipped: int = 0
    failures: list[dict[str, str]] = field(default_factory=list)


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
    seed: int | None = 42,
    score_timeout_sec: int = 600,
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
            seed=seed,
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
            timeout_sec=score_timeout_sec,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, TimeoutError) as exc:
        if isinstance(exc, FileNotFoundError):
            error = "musicinstruct CLI unavailable; install ../musicinstruct"
        elif isinstance(exc, TimeoutError):
            error = f"musicinstruct score timed out after {score_timeout_sec}s"
        else:
            error = exc.output or str(exc)
        results = {
            "error": error,
            "n_predictions_decoded": decode_result.n_written,
            "n_decode_failed": decode_result.n_failed,
        }
        results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    summary = {
        "repr_name": repr_name,
        "split": split,
        "seed": seed,
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
    failures_path: str | Path | None = None,
) -> CopySourceResult:
    """Zero-edit baseline: predict midi_in unchanged (sanity / lower bound)."""
    manifest = Path(manifest_path)
    root = manifest.parent
    predictions_root = Path(output_predictions).parent
    out_dir = predictions_root / "copy_source_midis"
    out_dir.mkdir(parents=True, exist_ok=True)
    result = CopySourceResult()
    fail_path = (
        Path(failures_path) if failures_path else predictions_root / "copy_source_failures.jsonl"
    )

    with Path(output_predictions).open("w", encoding="utf-8") as dst, fail_path.open(
        "w", encoding="utf-8"
    ) as fail_writer:
        for line_no, record in iter_jsonl(
            manifest,
            required_fields=MANIFEST_FIELDS,
            label="manifest",
        ):
            if split is not None and record.get("split") != split:
                result.n_skipped += 1
                continue
            src_midi = root / record["midi_in"]
            dest = out_dir / f"{record['item_id']}.mid"
            if not src_midi.is_file():
                result.n_failed += 1
                failure = {
                    "item_id": record["item_id"],
                    "line_no": str(line_no),
                    "error": f"missing midi_in: {src_midi}",
                }
                result.failures.append(failure)
                fail_writer.write(json.dumps(failure) + "\n")
                continue
            try:
                shutil.copy2(src_midi, dest)
            except OSError as exc:
                result.n_failed += 1
                failure = {
                    "item_id": record["item_id"],
                    "line_no": str(line_no),
                    "error": str(exc),
                }
                result.failures.append(failure)
                fail_writer.write(json.dumps(failure) + "\n")
                continue

            rel = dest.relative_to(predictions_root)
            dst.write(json.dumps({"item_id": record["item_id"], "midi_path": str(rel)}) + "\n")
            result.n_written += 1
    return result
