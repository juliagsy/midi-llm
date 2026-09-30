from pathlib import Path

import pretty_midi

from midi_llm.midi_repr._fidelity import compare_midi_fidelity


def test_compare_midi_fidelity_identical(tmp_path: Path):
    path = tmp_path / "same.mid"
    midi = pretty_midi.PrettyMIDI(initial_tempo=120)
    inst = pretty_midi.Instrument(program=0)
    inst.notes.append(pretty_midi.Note(80, 60, 0.0, 0.5))
    midi.instruments.append(inst)
    midi.write(str(path))

    report = compare_midi_fidelity(path, path)
    assert report.ok
    assert report.note_recall == 1.0
    assert report.note_precision == 1.0


def test_compare_midi_fidelity_detects_missing_notes(tmp_path: Path):
    src = tmp_path / "src.mid"
    out = tmp_path / "out.mid"

    src_midi = pretty_midi.PrettyMIDI(initial_tempo=120)
    src_inst = pretty_midi.Instrument(program=0)
    for pitch in (60, 64, 67):
        src_inst.notes.append(pretty_midi.Note(80, pitch, 0.0, 0.5))
    src_midi.instruments.append(src_inst)
    src_midi.write(str(src))

    out_midi = pretty_midi.PrettyMIDI(initial_tempo=120)
    out_inst = pretty_midi.Instrument(program=0)
    out_inst.notes.append(pretty_midi.Note(80, 60, 0.0, 0.5))
    out_midi.instruments.append(out_inst)
    out_midi.write(str(out))

    report = compare_midi_fidelity(src, out)
    assert not report.ok
    assert report.note_recall < 0.9
