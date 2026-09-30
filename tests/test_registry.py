import pytest

from midi_llm.midi_repr.registry import (
    DISABLED_REPR_NAMES,
    REPR_NAMES,
    RepresentationDisabledError,
    clear_repr_cache,
    get_repr,
    list_reprs,
)


def test_list_reprs_active_only():
    assert list_reprs() == list(REPR_NAMES)
    assert "abc" not in list_reprs()
    assert "abc" in list_reprs(include_disabled=True)


def test_unknown_repr():
    with pytest.raises(ValueError, match="unknown representation"):
        get_repr("invalid")


def test_abc_disabled():
    assert "abc" in DISABLED_REPR_NAMES
    with pytest.raises(RepresentationDisabledError, match="disabled"):
        get_repr("abc")


def test_get_repr_caches_backends():
    clear_repr_cache()
    first = get_repr("remi")
    second = get_repr("remi")
    assert first is second
    clear_repr_cache()
