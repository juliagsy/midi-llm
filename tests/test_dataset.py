import json
from pathlib import Path

from midi_llm.train.dataset import SFTJsonlDataset, format_llama_instruct


def test_format_llama_instruct():
    parts = format_llama_instruct("### Task\nEdit\n", "1 2 3")
    assert "start_header_id" in parts["prompt"]
    assert "assistant" in parts["prompt"]
    assert parts["completion"].endswith("<|eot_id|>")


def test_sft_jsonl_dataset(tmp_path: Path):
    shard = tmp_path / "tiny.jsonl"
    record = {
        "prompt": "hello",
        "completion": "world",
        "task": "edit",
        "repr_name": "remi",
    }
    shard.write_text(json.dumps(record) + "\n", encoding="utf-8")

    ds = SFTJsonlDataset(shard)
    assert len(ds) == 1
    row = ds[0]
    assert "hello" in row["text"]
    assert row["completion"].startswith("world")
