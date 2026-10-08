import pytest

from midi_llm.train.lora_sft import filter_sft_records_for_seq_len
from midi_llm.train.sft_tokenize import SFTExampleUnfit, tokenize_sft_example


class _FakeTokenizer:
    def __call__(self, text: str, add_special_tokens: bool = False):
        del add_special_tokens
        return {"input_ids": [ord(ch) for ch in text]}


def test_filter_drops_unfit_rows() -> None:
    tokenizer = _FakeTokenizer()
    short = {"prompt": "### Task\nEdit\n\n### Instruction\nUp.\n\n", "completion": "1 2 3"}
    long_prompt = (
        "### Task\nEdit\n\n### Instruction\nTranspose up.\n\n### MIDI (REMI tokens)\n"
        + ("9 " * 500)
        + "\n### Output MIDI\n"
    )
    long = {"prompt": long_prompt, "completion": "1 2 3 4 5 6 7 8 9 0"}

    kept, skipped_unfit, skipped_truncated = filter_sft_records_for_seq_len(
        [short, long],
        tokenizer,
        max_seq_len=64,
        chat_template=False,
    )
    assert len(kept) == 1
    assert skipped_unfit + skipped_truncated == 1
    assert kept[0] is short


def test_unfit_is_skipped_not_raised_in_filter() -> None:
    tokenizer = _FakeTokenizer()
    prompt = "### Task\nEdit\n\n### Instruction\nTranspose up.\n\n" + ("m" * 200)
    records = [{"prompt": prompt, "completion": "out"}]
    kept, skipped_unfit, skipped_truncated = filter_sft_records_for_seq_len(
        records, tokenizer, max_seq_len=64, chat_template=False
    )
    assert kept == []
    assert skipped_unfit == 1
    assert skipped_truncated == 0
    with pytest.raises(SFTExampleUnfit):
        tokenize_sft_example(
            tokenizer, prompt=prompt, completion="out", max_seq_len=64
        )
