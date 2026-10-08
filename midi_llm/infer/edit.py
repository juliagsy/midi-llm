"""Run instruction-edit inference over a MIDI-Instruct manifest."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from midi_llm.config import load_config
from midi_llm.config_helpers import resolve_eval_temperature, resolve_max_new_tokens
from midi_llm.infer.generate import generate_completion, load_causal_lm
from midi_llm.infer.preflight import preflight_encode, preflight_manifest
from midi_llm.infer.prompts import build_edit_prompt, iter_manifest_records
from midi_llm.reproducibility import item_seed, set_global_seed


def run_edit_inference(
    manifest_path: str | Path,
    output_dir: str | Path,
    *,
    repr_name: str,
    model_name: str | None = None,
    adapter_path: str | Path | None = None,
    split: str = "test",
    max_items: int | None = None,
    max_new_tokens: int | None = None,
    temperature: float | None = None,
    chat_template: bool = True,
    seed: int | None = 42,
    skip_encode_preflight: bool = False,
) -> dict[str, Any]:
    """Generate edit completions and write JSONL artifacts under output_dir."""
    cfg = load_config(repr_name)
    backbone = model_name or cfg["model"]["backbone"]
    token_cap = resolve_max_new_tokens(cfg, max_new_tokens)
    sample_temp = resolve_eval_temperature(cfg, temperature)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    preflight = preflight_manifest(manifest_path, split=split, max_items=max_items)
    encode_preflight = None
    if not skip_encode_preflight:
        encode_preflight = preflight_encode(
            manifest_path,
            repr_name=repr_name,
            split=split,
            max_items=max_items,
            fail_on_encode_error=False,
        )

    if seed is not None:
        set_global_seed(seed)

    model, tokenizer, device = load_causal_lm(backbone, adapter_path=adapter_path)
    completions_path = out / "completions.jsonl"
    encode_failures_path = out / "encode_failures.jsonl"

    count = 0
    n_encode_failed = 0
    n_hit_max_tokens = 0
    with (
        completions_path.open("w", encoding="utf-8") as writer,
        encode_failures_path.open("w", encoding="utf-8") as encode_fail_writer,
    ):
        for record in iter_manifest_records(manifest_path, split=split):
            if max_items is not None and count >= max_items:
                break
            try:
                item_id, model_input, _plain_prompt = build_edit_prompt(
                    manifest_path,
                    record,
                    repr_name=repr_name,
                    chat_template=chat_template,
                )
            except Exception as exc:  # noqa: BLE001 — skip per-item encode failures
                n_encode_failed += 1
                encode_fail_writer.write(
                    json.dumps(
                        {
                            "item_id": record["item_id"],
                            "error": str(exc),
                        }
                    )
                    + "\n"
                )
                continue

            sample_seed = None if seed is None else item_seed(seed, item_id)
            gen = generate_completion(
                model,
                tokenizer,
                model_input,
                max_new_tokens=token_cap,
                temperature=sample_temp,
                seed=sample_seed,
            )
            if gen.hit_max_new_tokens:
                n_hit_max_tokens += 1
            writer.write(
                json.dumps(
                    {
                        "item_id": item_id,
                        "completion": gen.text,
                        "repr_name": repr_name,
                        "prompt_chars": len(model_input),
                        "n_new_tokens": gen.n_new_tokens,
                        "hit_max_new_tokens": gen.hit_max_new_tokens,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            count += 1

    if count == 0:
        raise RuntimeError(
            f"edit inference wrote zero completions for split={split!r} in {manifest_path}"
        )

    meta = {
        "manifest": str(manifest_path),
        "split": split,
        "repr_name": repr_name,
        "model": backbone,
        "adapter_path": str(adapter_path) if adapter_path else None,
        "device": device,
        "n_completions": count,
        "n_encode_failed": n_encode_failed,
        "n_hit_max_new_tokens": n_hit_max_tokens,
        "n_preflight_items": preflight.n_items,
        "n_encode_preflight_failed": (
            encode_preflight.n_encode_failed if encode_preflight is not None else None
        ),
        "seed": seed,
        "temperature": sample_temp,
        "max_new_tokens": token_cap,
        "completions_path": str(completions_path),
        "encode_failures_path": str(encode_failures_path),
    }
    (out / "infer_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta
