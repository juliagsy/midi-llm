from midi_llm.reproducibility import item_seed


def test_item_seed_is_stable_and_distinct() -> None:
    a = item_seed(42, "seed_00_mute_0")
    b = item_seed(42, "seed_00_mute_0")
    c = item_seed(42, "seed_00_mute_1")
    assert a == b
    assert a != c
