"""Representation metadata embedded in SFT shards."""

from __future__ import annotations

from pathlib import Path

from midi_llm.config import load_config
from midi_llm.data.jsonl_io import SFT_FIELDS, iter_jsonl
from midi_llm.midi_repr.registry import DISABLED_REPR_NAMES, RepresentationDisabledError


def infer_repr_name_from_shard(shard_path: str | Path) -> str | None:
    """Read ``repr_name`` from the first record in an SFT JSONL shard."""
    for _line_no, record in iter_jsonl(shard_path, required_fields=SFT_FIELDS, label="SFT shard"):
        repr_name = record.get("repr_name")
        if isinstance(repr_name, str) and repr_name.strip():
            return repr_name.strip().lower()
        return None
    return None


def validate_repr_protocol(repr_name: str) -> None:
    """Ensure the representation config matches the documented BPE-text protocol."""
    key = repr_name.lower().strip()
    if key in DISABLED_REPR_NAMES:
        raise RepresentationDisabledError(
            f"representation {repr_name!r} is disabled in protocol validation"
        )

    cfg = load_config(repr_name)
    if cfg.get("enabled") is False:
        raise RepresentationDisabledError(
            f"representation {repr_name!r} is disabled (enabled: false in repr config)"
        )
    if cfg.get("vocab_extension"):
        raise ValueError(
            f"representation {repr_name!r} requests vocab_extension=true but training uses "
            "native Llama BPE on serialized MIDI text. Set vocab_extension: false "
            "and tokenizer_mode: bpe_text in repr configs, or implement embedding resize."
        )
    if cfg.get("tokenizer_mode") != "bpe_text":
        raise ValueError(
            f"representation {repr_name!r} has tokenizer_mode={cfg.get('tokenizer_mode')!r}; "
            "Paper 2 v0 expects tokenizer_mode: bpe_text"
        )


def resolve_repr_name(shard_path: str | Path, repr_name: str | None) -> str:
    """Resolve ``repr_name`` from CLI flag and/or shard metadata, then validate protocol."""
    inferred = infer_repr_name_from_shard(shard_path)
    if repr_name is None:
        if inferred is None:
            raise ValueError(
                "train-lora requires --repr or a shard whose first record includes repr_name"
            )
        resolved = inferred
    else:
        resolved = repr_name.strip().lower()
        if inferred is not None and inferred != resolved:
            raise ValueError(
                f"shard repr_name={inferred!r} conflicts with --repr={resolved!r}"
            )
    validate_repr_protocol(resolved)
    return resolved
