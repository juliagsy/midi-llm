"""MIDI-Instruct dataset mode and path resolution (pilot vs real v0.2)."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

DATASET_MODES = frozenset({"pilot", "real_v0.2"})


@dataclass(frozen=True)
class InstructDatasetPaths:
    """Resolved paths for a MIDI-Instruct manifest tree."""

    mode: str
    run_id: str
    data_dir: Path
    manifest_path: Path
    manifest_name: str
    drive_data_dir: Path | None
    target_items: int
    suggested_max_steps: int


def default_run_id(mode: str) -> str:
    if mode not in DATASET_MODES:
        allowed = ", ".join(sorted(DATASET_MODES))
        raise ValueError(f"Unknown dataset mode {mode!r}; expected one of {allowed}")
    return "real_v0.2_v1" if mode == "real_v0.2" else "pilot_v1"


def resolve_instruct_paths(
    *,
    mode: str,
    musicinstruct_root: str | Path,
    run_id: str | None = None,
    drive_musicinstruct_root: str | Path | None = None,
) -> InstructDatasetPaths:
    """Map dataset mode to local (and optional Drive) manifest directories."""
    if mode not in DATASET_MODES:
        allowed = ", ".join(sorted(DATASET_MODES))
        raise ValueError(f"Unknown dataset mode {mode!r}; expected one of {allowed}")

    root = Path(musicinstruct_root)
    resolved_run_id = run_id or default_run_id(mode)

    if mode == "pilot":
        data_rel = Path("data/pilot")
        manifest_name = "pilot.jsonl"
        target_items = 300
        suggested_max_steps = 50
    else:
        data_rel = Path("data/v0.2")
        manifest_name = "manifest.jsonl"
        target_items = 1000
        suggested_max_steps = 300

    data_dir = root / data_rel
    drive_data_dir = None
    if drive_musicinstruct_root is not None:
        drive_data_dir = Path(drive_musicinstruct_root) / data_rel

    return InstructDatasetPaths(
        mode=mode,
        run_id=resolved_run_id,
        data_dir=data_dir,
        manifest_path=data_dir / manifest_name,
        manifest_name=manifest_name,
        drive_data_dir=drive_data_dir,
        target_items=target_items,
        suggested_max_steps=suggested_max_steps,
    )


def ensure_instruct_data_local(
    paths: InstructDatasetPaths,
    *,
    use_drive: bool = True,
    prefer_symlink: bool = True,
) -> Path:
    """Ensure manifest exists under ``paths.data_dir``; restore from Drive when needed."""
    manifest = paths.manifest_path
    if manifest.is_file():
        return manifest

    if not use_drive or paths.drive_data_dir is None:
        raise FileNotFoundError(
            f"Missing {manifest}. Run musicinstruct colab 01 or generate the dataset locally."
        )

    drive_manifest = paths.drive_data_dir / paths.manifest_name
    if not drive_manifest.is_file():
        raise FileNotFoundError(
            f"Missing Drive dataset at {drive_manifest}. "
            f"Run musicinstruct colab 01 with DATASET_MODE={paths.mode!r} and sync to Drive."
        )

    paths.data_dir.parent.mkdir(parents=True, exist_ok=True)
    if paths.data_dir.is_symlink():
        paths.data_dir.unlink()
    elif paths.data_dir.is_dir() and not manifest.is_file():
        shutil.rmtree(paths.data_dir)

    if prefer_symlink:
        paths.data_dir.symlink_to(paths.drive_data_dir, target_is_directory=True)
    else:
        shutil.copytree(paths.drive_data_dir, paths.data_dir, dirs_exist_ok=True)

    if not manifest.is_file():
        raise FileNotFoundError(f"Restored data but manifest still missing: {manifest}")
    return manifest
