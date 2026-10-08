"""SFT tokenization helpers with truncation diagnostics."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

INSTRUCTION_MARKER = "### Instruction"
OUTPUT_MIDI_HEADER = "### Output MIDI\n"
MIDI_SECTION_MARKER = "### MIDI"
MIN_COMPLETION_TOKENS = 8


class SFTExampleUnfit(ValueError):
    """SFT row cannot fit ``max_seq_len`` without breaking the edit template."""


@dataclass(frozen=True)
class SFTTokenizeMeta:
    prompt_truncated: bool
    completion_truncated: bool


def _split_edit_prompt(prompt: str) -> tuple[str, str, str] | None:
    if INSTRUCTION_MARKER not in prompt or OUTPUT_MIDI_HEADER not in prompt:
        return None
    if not prompt.endswith(OUTPUT_MIDI_HEADER):
        return None
    core = prompt[: -len(OUTPUT_MIDI_HEADER)]
    midi_pos = core.find(MIDI_SECTION_MARKER)
    if midi_pos < 0:
        return None
    return core[:midi_pos], core[midi_pos:], OUTPUT_MIDI_HEADER


def _truncate_edit_prompt_ids(
    tokenizer: Any,
    prompt: str,
    max_prompt_len: int,
) -> tuple[list[int], bool] | None:
    """Shorten the input MIDI block while keeping task + instruction headers."""
    parts = _split_edit_prompt(prompt)
    if parts is None:
        return None
    fixed, midi_block, out_hdr = parts
    fixed_ids = tokenizer(fixed, add_special_tokens=False)["input_ids"]
    out_ids = tokenizer(out_hdr, add_special_tokens=False)["input_ids"]
    reserve = len(fixed_ids) + len(out_ids)
    if reserve >= max_prompt_len:
        return None
    budget = max_prompt_len - reserve
    midi_ids = tokenizer(midi_block, add_special_tokens=False)["input_ids"]
    if len(midi_ids) <= budget:
        return fixed_ids + midi_ids + out_ids, False
    midi_ids = midi_ids[-budget:]
    return fixed_ids + midi_ids + out_ids, True


def _wrapped_token_count(
    tokenizer: Any,
    raw_prompt: str,
    raw_completion: str,
) -> int:
    from midi_llm.train.dataset import format_llama_instruct

    parts = format_llama_instruct(raw_prompt, raw_completion)
    prompt_ids = tokenizer(parts["prompt"], add_special_tokens=False)["input_ids"]
    completion_ids = tokenizer(parts["completion"], add_special_tokens=False)["input_ids"]
    return len(prompt_ids) + len(completion_ids)


def shrink_raw_edit_prompt(
    tokenizer: Any,
    raw_prompt: str,
    raw_completion: str,
    max_seq_len: int,
) -> tuple[str, bool]:
    """Shorten the MIDI payload in a shard edit prompt until the Llama-wrapped row fits."""
    if _wrapped_token_count(tokenizer, raw_prompt, raw_completion) <= max_seq_len:
        return raw_prompt, False

    parts = _split_edit_prompt(raw_prompt)
    if parts is None:
        raise SFTExampleUnfit("SFT row is not a shard edit prompt")

    fixed, midi_block, out_hdr = parts
    header_end = midi_block.find("\n", midi_block.find(MIDI_SECTION_MARKER))
    if header_end < 0:
        raise SFTExampleUnfit("SFT row has a malformed MIDI block")

    midi_header = midi_block[: header_end + 1]
    midi_body = midi_block[header_end + 1 :]
    midi_body = midi_body.removesuffix("\n")

    body_ids = tokenizer(midi_body, add_special_tokens=False)["input_ids"]
    truncated = False
    while body_ids:
        candidate = f"{fixed}{midi_header}{tokenizer.decode(body_ids)}\n{out_hdr}"
        if _wrapped_token_count(tokenizer, candidate, raw_completion) <= max_seq_len:
            return candidate, truncated
        truncated = True
        cut = max(1, len(body_ids) // 10)
        body_ids = body_ids[cut:]

    raise SFTExampleUnfit(
        "SFT row cannot fit max_seq_len even with an empty MIDI payload; increase max_seq_len"
    )


def tokenize_sft_record(
    tokenizer: Any,
    *,
    raw_prompt: str,
    raw_completion: str,
    max_seq_len: int,
    chat_template: bool = True,
) -> tuple[list[int], list[int], SFTTokenizeMeta]:
    """Tokenize one JSONL shard row (raw edit prompt/completion)."""
    prompt_truncated = False
    raw_p = raw_prompt
    if chat_template:
        from midi_llm.train.dataset import format_llama_instruct

        raw_p, prompt_truncated = shrink_raw_edit_prompt(
            tokenizer,
            raw_prompt,
            raw_completion,
            max_seq_len,
        )
        parts = format_llama_instruct(raw_p, raw_completion)
        prompt, completion = parts["prompt"], parts["completion"]
    else:
        prompt, completion = raw_p, raw_completion

    ids, label, meta = tokenize_sft_example(
        tokenizer,
        prompt=prompt,
        completion=completion,
        max_seq_len=max_seq_len,
    )
    combined = SFTTokenizeMeta(
        prompt_truncated=meta.prompt_truncated or prompt_truncated,
        completion_truncated=meta.completion_truncated,
    )
    return ids, label, combined


def _decode_prompt_ids(tokenizer: Any, prompt_ids: list[int]) -> str:
    if hasattr(tokenizer, "decode"):
        return tokenizer.decode(prompt_ids, skip_special_tokens=False)
    return "".join(chr(token_id) for token_id in prompt_ids if 0 <= token_id < 128)


def tokenize_sft_example(
    tokenizer: Any,
    *,
    prompt: str,
    completion: str,
    max_seq_len: int,
) -> tuple[list[int], list[int], SFTTokenizeMeta]:
    """Tokenize one SFT row, reserving completion tokens when truncating."""
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    completion_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]

    prompt_truncated = False
    completion_truncated = False

    had_instruction = INSTRUCTION_MARKER in prompt
    max_prompt_len = max(0, max_seq_len - MIN_COMPLETION_TOKENS)
    if len(prompt_ids) > max_prompt_len:
        smart = _truncate_edit_prompt_ids(tokenizer, prompt, max_prompt_len)
        if smart is not None:
            prompt_ids, prompt_truncated = smart
        else:
            prompt_ids = prompt_ids[-max_prompt_len:]
            prompt_truncated = True
            if had_instruction and INSTRUCTION_MARKER not in _decode_prompt_ids(
                tokenizer, prompt_ids
            ):
                raise SFTExampleUnfit(
                    "SFT example lost ### Instruction after prompt truncation; "
                    "increase max_seq_len or drop this row"
                )

    ids = prompt_ids + completion_ids
    if len(ids) > max_seq_len:
        overflow = len(ids) - max_seq_len
        if overflow >= len(completion_ids):
            completion_ids = []
            ids = prompt_ids[-max_seq_len:]
            completion_truncated = True
        else:
            completion_ids = completion_ids[:-overflow]
            ids = prompt_ids + completion_ids
            completion_truncated = True

    prompt_len = len(ids) - len(completion_ids)
    label = ids.copy()
    label[:prompt_len] = [-100] * prompt_len
    if not any(token != -100 for token in label):
        raise ValueError("SFT example produced all-masked labels after truncation")

    meta = SFTTokenizeMeta(
        prompt_truncated=prompt_truncated,
        completion_truncated=completion_truncated,
    )
    if completion_truncated or prompt_truncated:
        logger.debug(
            "SFT truncated to fit max_seq_len=%s (prompt=%s completion=%s)",
            max_seq_len,
            prompt_truncated,
            completion_truncated,
        )

    return ids, label, meta
