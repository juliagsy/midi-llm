import json
from pathlib import Path

import pytest

from midi_llm.data.jsonl_io import MANIFEST_FIELDS, JsonlError, iter_jsonl


def test_iter_jsonl_reports_line_number(tmp_path: Path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"item_id": "a"}\nnot json\n', encoding="utf-8")
    with pytest.raises(JsonlError, match=":2:"):
        list(iter_jsonl(path))


def test_iter_jsonl_validates_required_fields(tmp_path: Path):
    path = tmp_path / "manifest.jsonl"
    path.write_text(json.dumps({"item_id": "x"}) + "\n", encoding="utf-8")
    with pytest.raises(JsonlError, match="missing required fields"):
        list(iter_jsonl(path, required_fields=MANIFEST_FIELDS, label="manifest"))


def test_iter_jsonl_ok(tmp_path: Path):
    path = tmp_path / "manifest.jsonl"
    row = {
        "item_id": "x",
        "midi_in": "in.mid",
        "instruction": "transpose",
        "split": "test",
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    records = list(iter_jsonl(path, required_fields=MANIFEST_FIELDS, label="manifest"))
    assert records == [(1, row)]
