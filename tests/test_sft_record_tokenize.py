from midi_llm.train.sft_tokenize import shrink_raw_edit_prompt, tokenize_sft_record


class _FakeTokenizer:
    def __call__(self, text: str, add_special_tokens: bool = False):
        del add_special_tokens
        return {"input_ids": [ord(ch) for ch in text]}

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(i) for i in ids if 0 <= i < 128)


def _edit_prompt(midi_payload: str) -> str:
    return (
        "### Task\nEdit the input MIDI according to the instruction.\n\n"
        "### Instruction\nTranspose up.\n\n"
        f"### MIDI (REMI tokens)\n{midi_payload}\n"
        "### Output MIDI\n"
    )


def test_shrink_raw_edit_prompt_before_chat_wrap() -> None:
    tokenizer = _FakeTokenizer()
    raw = _edit_prompt(" ".join(str(i) for i in range(400)))
    shrunk, truncated = shrink_raw_edit_prompt(tokenizer, raw, "1 2 3 4 5 6 7 8 9", max_seq_len=600)
    assert truncated
    assert "### Instruction" in shrunk
    assert shrunk.endswith("### Output MIDI\n")


def test_tokenize_sft_record_raw_without_chat() -> None:
    tokenizer = _FakeTokenizer()
    raw = _edit_prompt("1 2 3")
    ids, labels, meta = tokenize_sft_record(
        tokenizer,
        raw_prompt=raw,
        raw_completion="9 8 7",
        max_seq_len=512,
        chat_template=False,
    )
    assert len(ids) == len(labels)
    assert any(label != -100 for label in labels)
    assert not meta.completion_truncated
