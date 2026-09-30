"""Tests for truncation-safe SFT label masking."""

import pytest

from midi_llm.train.lora_sft import _tokenize_sft_example


class _FakeTokenizer:
    def __call__(self, text: str, add_special_tokens: bool = False):
        del add_special_tokens
        return {"input_ids": [ord(ch) for ch in text]}


def test_long_prompt_preserves_completion_labels() -> None:
    tokenizer = _FakeTokenizer()
    prompt = "p" * 200
    completion = "c" * 20
    ids, labels = _tokenize_sft_example(
        tokenizer,
        prompt=prompt,
        completion=completion,
        max_seq_len=64,
    )
    assert len(ids) <= 64
    assert any(label != -100 for label in labels)
    assert labels[-1] != -100


def test_all_masked_example_raises() -> None:
    tokenizer = _FakeTokenizer()
    with pytest.raises(ValueError, match="all-masked"):
        _tokenize_sft_example(
            tokenizer,
            prompt="x" * 100,
            completion="y",
            max_seq_len=8,
        )
