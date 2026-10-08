from pathlib import Path

import pytest

from midi_llm.instruct_dataset import (
    default_run_id,
    ensure_instruct_data_local,
    resolve_instruct_paths,
)


def test_default_run_id():
    assert default_run_id("pilot") == "pilot_v1"
    assert default_run_id("real_v0.2") == "real_v0.2_v1"
    with pytest.raises(ValueError, match="Unknown dataset mode"):
        default_run_id("other")


def test_resolve_instruct_paths_pilot(tmp_path: Path):
    paths = resolve_instruct_paths(
        mode="pilot",
        musicinstruct_root=tmp_path,
        drive_musicinstruct_root="/drive/musicinstruct",
    )
    assert paths.run_id == "pilot_v1"
    assert paths.manifest_path == tmp_path / "data/pilot/pilot.jsonl"
    assert paths.drive_data_dir == Path("/drive/musicinstruct/data/pilot")
    assert paths.suggested_max_steps == 50


def test_resolve_instruct_paths_real(tmp_path: Path):
    paths = resolve_instruct_paths(
        mode="real_v0.2",
        musicinstruct_root=tmp_path,
        run_id="custom_run",
    )
    assert paths.run_id == "custom_run"
    assert paths.manifest_path == tmp_path / "data/v0.2/manifest.jsonl"
    assert paths.target_items == 1000
    assert paths.suggested_max_steps == 300


def test_ensure_instruct_data_local_symlink(tmp_path: Path):
    drive_root = tmp_path / "drive"
    mi_root = tmp_path / "musicinstruct"
    drive_data = drive_root / "data/pilot"
    drive_data.mkdir(parents=True)
    manifest = drive_data / "pilot.jsonl"
    manifest.write_text('{"item_id":"x","midi_in":"a.mid","instruction":"i","split":"train"}\n')

    paths = resolve_instruct_paths(
        mode="pilot",
        musicinstruct_root=mi_root,
        drive_musicinstruct_root=drive_root,
    )
    restored = ensure_instruct_data_local(paths, use_drive=True, prefer_symlink=True)
    assert restored.is_file()
    assert paths.data_dir.is_symlink()
    assert paths.data_dir.resolve() == drive_data.resolve()
