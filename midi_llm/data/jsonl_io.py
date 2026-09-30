"""JSONL loading with line numbers and lightweight schema checks."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any


class JsonlError(ValueError):
    """Malformed JSONL row with file/line context."""

    def __init__(self, path: str | Path, line_no: int, message: str) -> None:
        self.path = Path(path)
        self.line_no = line_no
        self.message = message
        super().__init__(f"{self.path}:{line_no}: {message}")


MANIFEST_FIELDS = frozenset({"item_id", "midi_in", "instruction", "split"})
SFT_FIELDS = frozenset({"prompt", "completion", "task", "repr_name"})
COMPLETION_FIELDS = frozenset({"item_id", "completion"})
PREDICTION_FIELDS = frozenset({"item_id", "midi_path"})


def _validate_record(
    record: Mapping[str, Any],
    *,
    required: frozenset[str],
    path: Path,
    line_no: int,
    label: str,
) -> None:
    if not isinstance(record, dict):
        raise JsonlError(path, line_no, f"{label} record must be a JSON object")
    missing = required - record.keys()
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise JsonlError(path, line_no, f"{label} record missing required fields: {missing_list}")


def iter_jsonl(
    path: str | Path,
    *,
    required_fields: frozenset[str] | None = None,
    label: str = "JSONL",
) -> Iterator[tuple[int, dict[str, Any]]]:
    """Yield ``(line_no, record)`` for each non-empty line in a JSONL file."""
    file_path = Path(path)
    with file_path.open(encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise JsonlError(file_path, line_no, f"invalid JSON: {exc.msg}") from exc
            if required_fields is not None:
                _validate_record(
                    record,
                    required=required_fields,
                    path=file_path,
                    line_no=line_no,
                    label=label,
                )
            yield line_no, record


def load_jsonl(
    path: str | Path,
    *,
    required_fields: frozenset[str] | None = None,
    label: str = "JSONL",
) -> list[dict[str, Any]]:
    """Load all records from a JSONL file."""
    return [record for _, record in iter_jsonl(path, required_fields=required_fields, label=label)]
