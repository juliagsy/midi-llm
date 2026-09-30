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
