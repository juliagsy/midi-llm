import json
from pathlib import Path

from midi_llm.config_helpers import (
    cap_s3_max_steps,
    load_eval_item_ids,
    resolve_eval_temperature,
    resolve_grad_accum_steps,
    resolve_max_new_tokens,
    resolve_s3_grad_accum_steps,
    resolve_score_timeout_sec,
    s3_steps_per_epoch,
)


def test_resolve_grad_accum_from_token_budget() -> None:
    cfg = {"training": {"effective_batch_tokens": 4096}}
    assert resolve_grad_accum_steps(cfg, seq_len=512, batch_size=1) == 8


def test_s3_grad_accum_cap() -> None:
    cfg = {"training": {"effective_batch_tokens": 524288, "s3_grad_accum_cap": 8}}
    assert resolve_s3_grad_accum_steps(cfg, seq_len=2048, batch_size=1) == 8


def test_cap_s3_max_steps_by_epochs() -> None:
    steps, spe = cap_s3_max_steps(
        645,
        batch_size=1,
        grad_accum=8,
        max_epochs=3,
        requested_steps=300,
    )
    assert spe == 81
    assert steps == 243


def test_resolve_grad_accum_honors_override() -> None:
    cfg = {"training": {"effective_batch_tokens": 4096}}
    assert resolve_grad_accum_steps(cfg, seq_len=512, batch_size=1, override=3) == 3


def test_resolve_max_new_tokens_from_config() -> None:
    cfg = {"eval": {"max_new_tokens": 1024}}
    assert resolve_max_new_tokens(cfg) == 1024
    assert resolve_max_new_tokens(cfg, override=256) == 256


def test_resolve_eval_temperature_from_config() -> None:
    cfg = {"eval": {"temperature": 0.0}}
    assert resolve_eval_temperature(cfg) == 0.0
    assert resolve_eval_temperature(cfg, override=0.2) == 0.2


def test_resolve_score_timeout_from_config() -> None:
    cfg = {"eval": {"score_timeout_sec": 600}}
    assert resolve_score_timeout_sec(cfg) == 600
    assert resolve_score_timeout_sec(cfg, override=120) == 120


def test_load_eval_item_ids_from_list(tmp_path: Path) -> None:
    path = tmp_path / "ids.json"
    path.write_text(json.dumps(["a1", "b2"]), encoding="utf-8")
    cfg = {"eval": {"midicaps_eval_ids": str(path)}}
    assert load_eval_item_ids(cfg) == {"a1", "b2"}
