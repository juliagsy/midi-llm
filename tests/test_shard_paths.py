from pathlib import Path

from midi_llm.train.shard_paths import infer_validation_shard


def test_infer_validation_shard(tmp_path: Path) -> None:
    train = tmp_path / "edit_remi_train.jsonl"
    val = tmp_path / "edit_remi_validation.jsonl"
    train.write_text("{}\n", encoding="utf-8")
    val.write_text("{}\n", encoding="utf-8")
    assert infer_validation_shard(train) == val


def test_infer_validation_shard_missing(tmp_path: Path) -> None:
    train = tmp_path / "edit_remi_train.jsonl"
    train.write_text("{}\n", encoding="utf-8")
    assert infer_validation_shard(train) is None
