from midi_llm.data.templates import TaskKind, build_edit_example


def test_build_edit_example():
    ex = build_edit_example(
        repr_name="remi",
        instruction="Transpose up 3 semitones.",
        input_midi_payload="1 2 3",
        output_midi_payload="4 5 6",
        item_id="item_1",
        split="train",
    )
    assert ex.task == TaskKind.EDIT
    assert "Transpose up 3" in ex.prompt
    assert ex.completion == "4 5 6"
    assert "REMI" in ex.prompt
