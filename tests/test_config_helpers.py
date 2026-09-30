import json
from pathlib import Path

from midi_llm.config_helpers import load_eval_item_ids, resolve_grad_accum_steps


def test_resolve_grad_accum_from_token_budget() -> None:
    cfg = {"training": {"effective_batch_tokens": 4096}}
    assert resolve_grad_accum_steps(cfg, seq_len=512, batch_size=1) == 8


def test_resolve_grad_accum_honors_override() -> None:
    cfg = {"training": {"effective_batch_tokens": 4096}}
    assert resolve_grad_accum_steps(cfg, seq_len=512, batch_size=1, override=3) == 3


def test_load_eval_item_ids_from_list(tmp_path: Path) -> None:
    path = tmp_path / "ids.json"
    path.write_text(json.dumps(["a1", "b2"]), encoding="utf-8")
    cfg = {"eval": {"midicaps_eval_ids": str(path)}}
    assert load_eval_item_ids(cfg) == {"a1", "b2"}
