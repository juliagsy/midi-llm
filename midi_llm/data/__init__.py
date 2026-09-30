"""Training data templates and shard builders."""

from .templates import (
    TaskKind,
    build_caption_to_midi,
    build_edit_example,
    build_midi_to_caption,
)

__all__ = [
    "TaskKind",
    "build_caption_to_midi",
    "build_midi_to_caption",
    "build_edit_example",
]
