"""Validate manifest inputs before loading a model."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from midi_llm.data._encode import encode_midi_file
from midi_llm.data.jsonl_io import MANIFEST_FIELDS, iter_jsonl


@dataclass(frozen=True)
class ManifestPreflightReport:
    n_items: int
    n_missing_midi_in: int
    missing_paths: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return self.n_missing_midi_in == 0


def preflight_manifest(
    manifest_path: str | Path,
    *,
    split: str | None = "test",
    max_items: int | None = None,
    fail_on_missing: bool = True,
) -> ManifestPreflightReport:
    """Check that manifest rows reference existing ``midi_in`` files."""
    manifest = Path(manifest_path)
    root = manifest.parent
    missing: list[str] = []
    count = 0

    for _line_no, record in iter_jsonl(
        manifest,
        required_fields=MANIFEST_FIELDS,
        label="manifest",
    ):
        if split is not None and record.get("split") != split:
            continue
        if max_items is not None and count >= max_items:
            break
        midi_in = root / record["midi_in"]
        if not midi_in.is_file():
            missing.append(str(midi_in))
        count += 1

    if count == 0:
        raise RuntimeError(f"manifest preflight found zero items for split={split!r} in {manifest}")

    if missing and fail_on_missing:
        preview = ", ".join(missing[:5])
        suffix = "..." if len(missing) > 5 else ""
        raise FileNotFoundError(
            f"manifest preflight failed: {len(missing)}/{count} midi_in files missing "
            f"(examples: {preview}{suffix})"
        )

    return ManifestPreflightReport(
        n_items=count,
        n_missing_midi_in=len(missing),
        missing_paths=tuple(missing),
    )


@dataclass(frozen=True)
class EncodePreflightReport:
    n_items: int
    n_encode_failed: int
    failures: tuple[dict[str, str], ...]

    @property
    def ok(self) -> bool:
        return self.n_encode_failed == 0


def preflight_encode(
    manifest_path: str | Path,
    *,
    repr_name: str,
    split: str | None = "test",
    max_items: int | None = None,
    fail_on_encode_error: bool = True,
) -> EncodePreflightReport:
    """Verify manifest rows encode under ``repr_name`` before loading a model."""
    manifest = Path(manifest_path)
    root = manifest.parent
    failures: list[dict[str, str]] = []
    count = 0

    for line_no, record in iter_jsonl(
        manifest,
        required_fields=MANIFEST_FIELDS,
        label="manifest",
    ):
        if split is not None and record.get("split") != split:
            continue
        if max_items is not None and count >= max_items:
            break
        midi_in = root / record["midi_in"]
        count += 1
        if not midi_in.is_file():
            continue
        try:
            encode_midi_file(repr_name, midi_in)
        except Exception as exc:  # noqa: BLE001 — collect per-item encode failures
            failures.append(
                {
                    "item_id": record["item_id"],
                    "line_no": str(line_no),
                    "midi_in": str(midi_in),
                    "error": str(exc),
                }
            )

    if count == 0:
        raise RuntimeError(f"encode preflight found zero items for split={split!r} in {manifest}")

    if failures and fail_on_encode_error:
        preview = failures[0]["item_id"]
        raise RuntimeError(
            f"encode preflight failed: {len(failures)}/{count} items (first item_id={preview})"
        )

    return EncodePreflightReport(
        n_items=count,
        n_encode_failed=len(failures),
        failures=tuple(failures),
    )


def summarize_manifest_skips(
    manifest_path: str | Path, *, split: str | None = "test"
) -> Counter[str]:
    """Count non-fatal manifest skip reasons (for diagnostics)."""
    manifest = Path(manifest_path)
    root = manifest.parent
    skipped: Counter[str] = Counter()
    for _line_no, record in iter_jsonl(
        manifest,
        required_fields=MANIFEST_FIELDS,
        label="manifest",
    ):
        if split is not None and record.get("split") != split:
            skipped["wrong_split"] += 1
            continue
        if not (root / record["midi_in"]).is_file():
            skipped["missing_midi_in"] += 1
    return skipped
