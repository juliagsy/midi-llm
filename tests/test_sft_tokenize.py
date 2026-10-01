"""Tests for truncation-safe SFT label masking."""

import pytest

from midi_llm.train.sft_tokenize import tokenize_sft_example


class _FakeTokenizer:
    def __call__(self, text: str, add_special_tokens: bool = False):
        del add_special_tokens
        return {"input_ids": [ord(ch) for ch in text]}


def test_long_prompt_preserves_completion_labels() -> None:
    tokenizer = _FakeTokenizer()
    prompt = "p" * 200
    completion = "c" * 20
    ids, labels, meta = tokenize_sft_example(
        tokenizer,
        prompt=prompt,
        completion=completion,
        max_seq_len=64,
    )
    assert len(ids) <= 64
    assert any(label != -100 for label in labels)
    assert labels[-1] != -100
    assert meta.completion_truncated or meta.prompt_truncated


def test_all_masked_example_raises() -> None:
    tokenizer = _FakeTokenizer()
    with pytest.raises(ValueError, match="all-masked"):
        tokenize_sft_example(
            tokenizer,
            prompt="x" * 100,
            completion="y",
            max_seq_len=8,
        )


def test_instruction_loss_raises() -> None:
    tokenizer = _FakeTokenizer()
    prompt = "### Task\nEdit\n\n### Instruction\nTranspose up.\n\n" + ("m" * 200)
    completion = "out"
    with pytest.raises(ValueError, match="Instruction"):
        tokenize_sft_example(
            tokenizer,
            prompt=prompt,
            completion=completion,
            max_seq_len=64,
        )
