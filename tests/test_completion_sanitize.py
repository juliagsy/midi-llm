"""Regression tests for chat-token stripping in MIDI completion payloads."""

from midi_llm.midi_repr.payload import deserialize_midi_payload, sanitize_midi_completion_text


def test_sanitize_strips_eot_id() -> None:
    assert sanitize_midi_completion_text("1 2 3<|eot_id|>") == "1 2 3"


def test_deserialize_flat_tokens_with_eot_suffix() -> None:
    decoded = deserialize_midi_payload("remi", "42 17 99<|eot_id|>")
    assert decoded.token_ids == [42, 17, 99]


def test_deserialize_octuple_with_eot_suffix() -> None:
    decoded = deserialize_midi_payload("octuple", "43,4,4,23|44,12,4,23<|eot_id|>")
    assert decoded.compound_token_ids == [[43, 4, 4, 23], [44, 12, 4, 23]]
