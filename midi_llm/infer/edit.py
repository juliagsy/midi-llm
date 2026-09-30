"""Run instruction-edit inference over a MIDI-Instruct manifest."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from midi_llm.config import load_config
from midi_llm.infer.generate import generate_completion, load_causal_lm
from midi_llm.infer.preflight import preflight_manifest
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
    max_new_tokens: int = 512,
    temperature: float = 0.2,
    chat_template: bool = True,
    seed: int | None = 42,
) -> dict[str, Any]:
    """Generate edit completions and write JSONL artifacts under output_dir."""
    cfg = load_config(repr_name)
    backbone = model_name or cfg["model"]["backbone"]
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    preflight = preflight_manifest(manifest_path, split=split, max_items=max_items)
    if seed is not None:
        set_global_seed(seed)

    model, tokenizer, device = load_causal_lm(backbone, adapter_path=adapter_path)
    completions_path = out / "completions.jsonl"

    count = 0
    with completions_path.open("w", encoding="utf-8") as writer:
        for record in iter_manifest_records(manifest_path, split=split):
            if max_items is not None and count >= max_items:
                break
            item_id, model_input, _plain_prompt = build_edit_prompt(
                manifest_path,
                record,
                repr_name=repr_name,
                chat_template=chat_template,
            )
            sample_seed = None if seed is None else item_seed(seed, item_id)
            completion = generate_completion(
                model,
                tokenizer,
                model_input,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                seed=sample_seed,
            )
            writer.write(
                json.dumps(
                    {
                        "item_id": item_id,
                        "completion": completion,
                        "repr_name": repr_name,
                        "prompt_chars": len(model_input),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            count += 1

    meta = {
        "manifest": str(manifest_path),
        "split": split,
        "repr_name": repr_name,
        "model": backbone,
        "adapter_path": str(adapter_path) if adapter_path else None,
        "device": device,
        "n_completions": count,
        "n_preflight_items": preflight.n_items,
        "seed": seed,
        "temperature": temperature,
        "completions_path": str(completions_path),
    }
    (out / "infer_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta
