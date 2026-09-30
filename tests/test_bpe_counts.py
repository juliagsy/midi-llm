from unittest.mock import patch

from midi_llm.midi_repr.base import EncodeResult, EncodeStats
from midi_llm.tokenization.bpe_counts import count_llama_bpe_tokens, serialized_payload


def test_serialized_payload_from_token_ids():
    encoded = EncodeResult(
        repr_name="remi",
        token_ids=[1, 2, 3],
        stats=EncodeStats.from_counts(n_notes=1, n_tokens=3, n_tracks=1, duration_sec=1.0),
    )
    assert serialized_payload(encoded) == "1 2 3"


def test_serialized_payload_from_text():
    encoded = EncodeResult(repr_name="abc", text="X:1\nK:C\nC2", stats=None)
    assert serialized_payload(encoded) == "X:1\nK:C\nC2"


def test_serialized_payload_from_compound_tokens():
    encoded = EncodeResult(
        repr_name="octuple",
        compound_token_ids=[[1, 2, 3], [4, 5, 6]],
    )
    assert serialized_payload(encoded) == "1,2,3|4,5,6"


def test_count_llama_bpe_tokens_uses_tokenizer():
    class FakeTokenizer:
        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            assert add_special_tokens is False
            return [0] * (len(text.split()) * 2)

    with patch("midi_llm.tokenization.bpe_counts._load_tokenizer", return_value=FakeTokenizer()):
        assert count_llama_bpe_tokens("10 20 30 40", "fake-model") == 8


def test_token_stats_bpe_without_network(sample_midi, miditok_available):
    if not miditok_available:
        import pytest

        pytest.skip("miditok not installed")

    from midi_llm.tokenization.stats import token_stats_for_midi

    class FakeTokenizer:
        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            return [0] * max(1, len(text.split()))

    with patch("midi_llm.tokenization.bpe_counts._load_tokenizer", return_value=FakeTokenizer()):
        row = token_stats_for_midi(sample_midi, "remi", model_name="fake-model")
    assert row.error is None
    assert row.n_repr_tokens > 0
    assert row.n_bpe_tokens is not None
    assert row.n_bpe_tokens > 0
    assert row.bpe_tokens_per_note is not None
