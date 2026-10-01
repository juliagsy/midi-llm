"""SFT tokenization helpers with truncation diagnostics."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

INSTRUCTION_MARKER = "### Instruction"
MIN_COMPLETION_TOKENS = 8


@dataclass(frozen=True)
class SFTTokenizeMeta:
    prompt_truncated: bool
    completion_truncated: bool


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
        prompt_ids = prompt_ids[-max_prompt_len:]
        prompt_truncated = True
        if had_instruction and INSTRUCTION_MARKER not in _decode_prompt_ids(tokenizer, prompt_ids):
            raise ValueError(
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
    if completion_truncated:
        logger.warning(
            "SFT completion truncated to fit max_seq_len=%s (prompt_truncated=%s)",
            max_seq_len,
            prompt_truncated,
        )
    elif prompt_truncated:
        logger.warning(
            "SFT prompt truncated from the left to fit max_seq_len=%s",
            max_seq_len,
        )

    return ids, label, meta
