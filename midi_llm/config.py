"""Load merged YAML experiment configs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(
    repr_name: str | None = None,
    *,
    base_path: str | Path | None = None,
) -> dict[str, Any]:
    """Load `base.yaml` merged with `repr_<name>.yaml` when repr_name is set."""
    root = Path(base_path) if base_path else CONFIG_DIR
    with (root / "base.yaml").open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if repr_name:
        repr_file = root / f"repr_{repr_name}.yaml"
        if not repr_file.is_file():
            raise FileNotFoundError(f"missing representation config: {repr_file}")
        with repr_file.open(encoding="utf-8") as handle:
            repr_cfg = yaml.safe_load(handle)
        config = _deep_merge(config, repr_cfg)
    return config
