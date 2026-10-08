"""Resolve experiment knobs from merged YAML configs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def resolve_grad_accum_steps(
    cfg: dict[str, Any],
    *,
    seq_len: int,
    batch_size: int,
    override: int | None = None,
) -> int:
    """Map ``training.effective_batch_tokens`` to gradient accumulation steps."""
    if override is not None:
        return max(1, override)

    token_budget = cfg.get("training", {}).get("effective_batch_tokens")
    if token_budget is None:
        return 8

    per_step = max(1, seq_len * batch_size)
    return max(1, int(token_budget) // per_step)


def resolve_s3_grad_accum_steps(
    cfg: dict[str, Any],
    *,
    seq_len: int,
    batch_size: int,
    override: int | None = None,
) -> int:
    """Gradient accumulation for S3 edit LoRA (capped for small MIDI-Instruct shards)."""
    accum = resolve_grad_accum_steps(
        cfg,
        seq_len=seq_len,
        batch_size=batch_size,
        override=override,
    )
    cap = cfg.get("training", {}).get("s3_grad_accum_cap")
    if cap is None:
        return accum
    return max(1, min(accum, int(cap)))


def s3_steps_per_epoch(n_train: int, *, batch_size: int, grad_accum: int) -> int:
    """Optimizer steps to draw ~one epoch over ``n_train`` SFT rows."""
    samples_per_step = max(1, batch_size * grad_accum)
    return max(1, (n_train + samples_per_step - 1) // samples_per_step)


def cap_s3_max_steps(
    n_train: int,
    *,
    batch_size: int,
    grad_accum: int,
    max_epochs: float,
    requested_steps: int | None,
) -> tuple[int, int]:
    """Limit ``max_steps`` so training does not repeat the shard for many epochs."""
    steps_per_epoch = s3_steps_per_epoch(n_train, batch_size=batch_size, grad_accum=grad_accum)
    epoch_cap = max(1, int(steps_per_epoch * max(0.1, max_epochs)))
    if requested_steps is None:
        return epoch_cap, steps_per_epoch
    return min(requested_steps, epoch_cap), steps_per_epoch


def resolve_max_new_tokens(cfg: dict[str, Any], override: int | None = None) -> int:
    """Return generation cap for edit completions."""
    if override is not None:
        return max(1, override)
    return max(1, int(cfg.get("eval", {}).get("max_new_tokens", 1024)))


def resolve_eval_temperature(cfg: dict[str, Any], override: float | None = None) -> float:
    if override is not None:
        return override
    return float(cfg.get("eval", {}).get("temperature", 0.0))


def resolve_score_timeout_sec(cfg: dict[str, Any], override: int | None = None) -> int:
    if override is not None:
        return max(1, override)
    return max(1, int(cfg.get("eval", {}).get("score_timeout_sec", 600)))


def load_eval_item_ids(cfg: dict[str, Any]) -> set[str] | None:
    """Load optional eval item-id filter from ``eval.midicaps_eval_ids`` path."""
    raw = cfg.get("eval", {}).get("midicaps_eval_ids")
    if raw is None:
        return None

    path = Path(str(raw)).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"eval.midicaps_eval_ids file not found: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return {str(item_id) for item_id in data}
    if isinstance(data, dict) and "item_ids" in data:
        return {str(item_id) for item_id in data["item_ids"]}
    raise ValueError(
        f"eval item-id file must be a JSON list or {{\"item_ids\": [...]}}: {path}"
    )
