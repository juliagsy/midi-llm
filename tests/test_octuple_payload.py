import json
from pathlib import Path

import pytest

from midi_llm.data._encode import encode_midi_file
from midi_llm.data.instruct import build_instruct_shard
from midi_llm.eval.musicinstruct_runner import predictions_from_completions
from midi_llm.midi_repr.payload import deserialize_midi_payload, serialize_midi_payload
from midi_llm.midi_repr.registry import get_repr


def test_octuple_serialize_deserialize_roundtrip(sample_midi, miditok_available):
    if not miditok_available:
        pytest.skip("miditok not installed")

    backend = get_repr("octuple")
    encoded = backend.encode(sample_midi)
    assert encoded.compound_token_ids
    assert encoded.token_ids is None

    payload = serialize_midi_payload("octuple", encoded)
    assert "|" in payload
    assert "[" not in payload

    decoded = deserialize_midi_payload("octuple", payload)
    assert decoded.compound_token_ids == encoded.compound_token_ids


def test_octuple_roundtrip_fidelity(sample_midi, miditok_available):
    if not miditok_available:
        pytest.skip("miditok not installed")

    backend = get_repr("octuple")
    result = backend.roundtrip(sample_midi)
    assert result.success, result.error
    assert result.fidelity_ok is True


def test_octuple_legacy_payload_still_decodes(sample_midi, miditok_available):
    if not miditok_available:
        pytest.skip("miditok not installed")

    backend = get_repr("octuple")
    encoded = backend.encode(sample_midi)
    legacy = " ".join(str(compound) for compound in encoded.compound_token_ids)
    decoded = deserialize_midi_payload("octuple", legacy)
    assert decoded.compound_token_ids == encoded.compound_token_ids


def test_octuple_shard_completion_decode(
    miditok_available,
    instruct_mini_manifest: Path,
    tmp_path: Path,
):
    if not miditok_available:
        pytest.skip("miditok not installed")

    shard = tmp_path / "edit_octuple.jsonl"
    count = build_instruct_shard(
        instruct_mini_manifest,
        repr_name="octuple",
        output_path=shard,
        split="train",
    )
    assert count == 1

    record = json.loads(shard.read_text(encoding="utf-8").strip())
    completion = record["completion"]

    completions = tmp_path / "completions.jsonl"
    item_id = json.loads(instruct_mini_manifest.read_text(encoding="utf-8").splitlines()[0])[
        "item_id"
    ]
    completions.write_text(
        json.dumps({"item_id": item_id, "completion": completion}) + "\n",
        encoding="utf-8",
    )

    preds = tmp_path / "predictions.jsonl"
    result = predictions_from_completions(
        instruct_mini_manifest,
        completions,
        repr_name="octuple",
        output_predictions=preds,
        split="train",
    )
    assert result.n_written == 1
    assert result.n_failed == 0
    assert preds.read_text(encoding="utf-8").strip()


def test_encode_midi_file_octuple_uses_compound_format(sample_midi, miditok_available):
    if not miditok_available:
        pytest.skip("miditok not installed")

    payload = encode_midi_file("octuple", sample_midi)
    assert "|" in payload
    deserialize_midi_payload("octuple", payload)
