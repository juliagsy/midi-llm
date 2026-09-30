"""SFT prompt templates per training stage."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class TaskKind(str, Enum):
    CAPTION_TO_MIDI = "caption_to_midi"
    MIDI_TO_CAPTION = "midi_to_caption"
    EDIT = "edit"


class SFTExample(BaseModel):
    """One supervised fine-tuning record."""

    task: TaskKind
    repr_name: str
    prompt: str
    completion: str
    metadata: dict[str, Any] = Field(default_factory=dict)


def _wrap_midi_block(repr_name: str, payload: str) -> str:
    if repr_name == "abc":
        return f"### MIDI (ABC)\n{payload.strip()}\n"
    return f"### MIDI ({repr_name.upper()} tokens)\n{payload.strip()}\n"


def build_caption_to_midi(
    *,
    repr_name: str,
    caption: str,
    midi_payload: str,
    item_id: str | None = None,
) -> SFTExample:
    prompt = f"### Task\nGenerate MIDI from the caption.\n\n### Caption\n{caption.strip()}\n\n### MIDI\n"
    completion = midi_payload.strip()
    return SFTExample(
        task=TaskKind.CAPTION_TO_MIDI,
        repr_name=repr_name,
        prompt=prompt,
        completion=completion,
        metadata={"item_id": item_id} if item_id else {},
    )


def build_midi_to_caption(
    *,
    repr_name: str,
    caption: str,
    midi_payload: str,
    item_id: str | None = None,
) -> SFTExample:
    prompt = (
        "### Task\nDescribe the MIDI.\n\n"
        f"{_wrap_midi_block(repr_name, midi_payload)}"
        "### Caption\n"
    )
    completion = caption.strip()
    return SFTExample(
        task=TaskKind.MIDI_TO_CAPTION,
        repr_name=repr_name,
        prompt=prompt,
        completion=completion,
        metadata={"item_id": item_id} if item_id else {},
    )


def build_edit_example(
    *,
    repr_name: str,
    instruction: str,
    input_midi_payload: str,
    output_midi_payload: str,
    item_id: str,
    split: str,
) -> SFTExample:
    prompt = (
        "### Task\nEdit the input MIDI according to the instruction.\n\n"
        f"### Instruction\n{instruction.strip()}\n\n"
        f"{_wrap_midi_block(repr_name, input_midi_payload)}"
        "### Output MIDI\n"
    )
    completion = output_midi_payload.strip()
    return SFTExample(
        task=TaskKind.EDIT,
        repr_name=repr_name,
        prompt=prompt,
        completion=completion,
        metadata={"item_id": item_id, "split": split},
    )
