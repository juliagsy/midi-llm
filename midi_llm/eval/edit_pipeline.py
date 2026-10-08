"""End-to-end edit eval: completions → MIDI predictions → musicinstruct score."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from midi_llm.config import load_config
from midi_llm.config_helpers import (
    load_eval_item_ids,
    resolve_eval_temperature,
    resolve_max_new_tokens,
    resolve_score_timeout_sec,
)
from midi_llm.data.jsonl_io import MANIFEST_FIELDS, iter_jsonl
from midi_llm.eval.musicinstruct_runner import predictions_from_completions, run_musicinstruct_eval
from midi_llm.infer.edit import run_edit_inference
from midi_llm.infer.preflight import preflight_manifest


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
    max_new_tokens: int | None = None,
    temperature: float | None = None,
    skip_infer: bool = False,
    completions_path: str | Path | None = None,
    joint_threshold: float = 0.9,
    seed: int | None = 42,
    score_timeout_sec: int | None = None,
) -> dict[str, Any]:
    """Infer (optional), decode, and score against MIDI-Instruct."""
    cfg = load_config(repr_name)
    item_ids = load_eval_item_ids(cfg)
    token_cap = resolve_max_new_tokens(cfg, max_new_tokens)
    sample_temp = resolve_eval_temperature(cfg, temperature)
    timeout_sec = resolve_score_timeout_sec(cfg, score_timeout_sec)

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
            max_new_tokens=token_cap,
            temperature=sample_temp,
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
    scoring_ok = True
    scoring_error: str | None = None
    try:
        results = run_musicinstruct_eval(
            manifest_path,
            predictions_path,
            output_results=results_path,
            split=split,
            joint_threshold=joint_threshold,
            item_ids=item_ids,
            timeout_sec=timeout_sec,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, TimeoutError) as exc:
        scoring_ok = False
        if isinstance(exc, FileNotFoundError):
            scoring_error = "musicinstruct CLI unavailable; install ../musicinstruct"
        elif isinstance(exc, TimeoutError):
            scoring_error = f"musicinstruct score timed out after {timeout_sec}s"
        else:
            scoring_error = exc.output or str(exc)
        results = {
            "error": scoring_error,
            "n_predictions_decoded": decode_result.n_written,
            "n_decode_failed": decode_result.n_failed,
        }
        results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    if scoring_ok and results.get("overall") is None:
        scoring_ok = False
        scoring_error = results.get("error") or "musicinstruct score returned no overall metrics"

    summary = {
        "repr_name": repr_name,
        "split": split,
        "joint_threshold": joint_threshold,
        "n_eval_item_ids": len(item_ids) if item_ids is not None else None,
        "seed": seed,
        "max_new_tokens": token_cap,
        "temperature": sample_temp,
        "score_timeout_sec": timeout_sec,
        "n_predictions_decoded": decode_result.n_written,
        "n_decode_failed": decode_result.n_failed,
        "n_decode_skipped": decode_result.n_skipped,
        "n_decode_duplicates": decode_result.n_duplicates,
        "decode_failures_path": str(out / "decode_failures.jsonl"),
        "completions_path": str(completions_path),
        "predictions_path": str(predictions_path),
        "results_path": str(results_path),
        "scoring_ok": scoring_ok,
        "scoring_error": scoring_error,
        "overall": results.get("overall"),
    }
    (out / "eval_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def copy_source_as_baseline(
    manifest_path: str | Path,
    output_predictions: str | Path,
    *,
    split: str = "test",
    max_items: int | None = None,
    failures_path: str | Path | None = None,
) -> CopySourceResult:
    """Zero-edit baseline: predict midi_in unchanged (sanity / lower bound)."""
    manifest = Path(manifest_path)
    preflight_manifest(
        manifest_path,
        split=split,
        max_items=max_items,
        fail_on_missing=False,
    )
    root = manifest.parent
    predictions_root = Path(output_predictions).parent
    out_dir = predictions_root / "copy_source_midis"
    out_dir.mkdir(parents=True, exist_ok=True)
    result = CopySourceResult()
    fail_path = (
        Path(failures_path) if failures_path else predictions_root / "copy_source_failures.jsonl"
    )

    with (
        Path(output_predictions).open("w", encoding="utf-8") as dst,
        fail_path.open("w", encoding="utf-8") as fail_writer,
    ):
        written_limit = max_items
        for line_no, record in iter_jsonl(
            manifest,
            required_fields=MANIFEST_FIELDS,
            label="manifest",
        ):
            if split is not None and record.get("split") != split:
                result.n_skipped += 1
                continue
            if written_limit is not None and result.n_written >= written_limit:
                break
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

    if result.n_written == 0:
        raise RuntimeError(
            f"copy-source baseline wrote zero predictions for split={split!r} in {manifest}"
        )
    return result
