import pytest

from midi_llm.midi_repr.registry import get_repr


def test_remi_roundtrip(sample_midi, miditok_available):
    if not miditok_available:
        pytest.skip("miditok not installed")
    backend = get_repr("remi")
    result = backend.roundtrip(sample_midi)
    assert result.success, result.error
    assert result.output is not None
    assert result.stats_before is not None
    assert result.stats_before.n_tokens > 0


def test_octuple_encode(sample_midi, miditok_available):
    if not miditok_available:
        pytest.skip("miditok not installed")
    backend = get_repr("octuple")
    encoded = backend.encode(sample_midi)
    assert encoded.token_ids
    assert encoded.stats
    assert encoded.stats.tokens_per_note > 0
