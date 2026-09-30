import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from midi_llm.eval.musicinstruct_runner import run_musicinstruct_eval


def test_run_musicinstruct_eval_timeout(tmp_path: Path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("{}\n", encoding="utf-8")
    preds = tmp_path / "preds.jsonl"
    preds.write_text("{}\n", encoding="utf-8")
    results = tmp_path / "results.json"

    with (
        patch("midi_llm.eval.musicinstruct_runner.shutil.which", return_value="/usr/bin/musicinstruct"),
        patch(
            "midi_llm.eval.musicinstruct_runner.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd=["musicinstruct"], timeout=5),
        ),
        pytest.raises(TimeoutError, match="exceeded"),
    ):
        run_musicinstruct_eval(
            manifest,
            preds,
            output_results=results,
            timeout_sec=5,
        )
