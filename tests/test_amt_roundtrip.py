import importlib.util

import pytest

from midi_llm.midi_repr.registry import get_repr


@pytest.mark.skipif(
    importlib.util.find_spec("anticipation") is None,
    reason="anticipation not installed",
)
def test_amt_roundtrip(sample_midi):
    backend = get_repr("amt")
    result = backend.roundtrip(sample_midi)
    assert result.success, result.error
