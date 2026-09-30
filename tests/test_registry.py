import pytest

from midi_llm.midi_repr.registry import REPR_NAMES, get_repr, list_reprs


def test_list_reprs():
    assert list_reprs() == list(REPR_NAMES)


def test_unknown_repr():
    with pytest.raises(ValueError, match="unknown representation"):
        get_repr("invalid")
