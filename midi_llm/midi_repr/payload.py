"""Representation-aware serialization for LLM training and evaluation."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass

from .base import EncodeResult

COMPOUND_SEP = "|"
FIELD_SEP = ","

# Llama chat terminators that may appear in generated completion text.
_CHAT_TERMINATOR_RE = re.compile(
    r"<\|(?:eot_id|end_of_text|end_of_turn|end_of_message)\|>",
    re.IGNORECASE,
)

# Legacy broken payloads used Python list repr, e.g. "[43, 4, 4] [44, 12, 4]"
_LEGACY_COMPOUND_RE = re.compile(r"\[[^\]]+\]")


def sanitize_midi_completion_text(payload: str) -> str:
    """Strip chat control tokens and surrounding whitespace from model output."""
    cleaned = _CHAT_TERMINATOR_RE.sub("", payload)
    return cleaned.strip()


@dataclass(frozen=True)
class DecodedPayload:
    token_ids: list[int] | None = None
    compound_token_ids: list[list[int]] | None = None
    text: str | None = None


def extract_compound_token_ids(encoded) -> list[list[int]]:
    """Read compound Octuple tokens from a MidiTok encode result."""
    if hasattr(encoded, "ids"):
        raw = list(encoded.ids)
    elif isinstance(encoded, list):
        raw = encoded
    else:
        raise TypeError(f"unexpected encoded token type: {type(encoded)!r}")

    if not raw:
        return []
    if not isinstance(raw[0], list):
        raise TypeError("expected compound token rows from Octuple encoder")
    return [[int(value) for value in row] for row in raw]


def extract_flat_token_ids(encoded) -> list[int]:
    """Read flat integer tokens from a MidiTok encode result."""
    if hasattr(encoded, "ids"):
        raw = list(encoded.ids)
    elif isinstance(encoded, list):
        raw = encoded
    else:
        raise TypeError(f"unexpected encoded token type: {type(encoded)!r}")

    if not raw:
        return []
    if isinstance(raw[0], list):
        flat: list[int] = []
        for row in raw:
            flat.extend(int(value) for value in row)
        return flat
    return [int(value) for value in raw]


def serialize_midi_payload(repr_name: str, encoded: EncodeResult) -> str:
    """Serialize an encode result to the text form consumed by the LLM."""
    if encoded.text is not None:
        return encoded.text.strip()

    if repr_name == "octuple":
        compounds = encoded.compound_token_ids
        if compounds is None:
            raise ValueError("octuple encode result missing compound_token_ids")
        return COMPOUND_SEP.join(
            FIELD_SEP.join(str(value) for value in compound) for compound in compounds
        )

    if encoded.token_ids is not None:
        return " ".join(str(token_id) for token_id in encoded.token_ids)

    raise ValueError(f"{repr_name} produced empty encoding")


def _parse_legacy_compounds(text: str) -> list[list[int]]:
    compounds: list[list[int]] = []
    for match in _LEGACY_COMPOUND_RE.finditer(text):
        parsed = ast.literal_eval(match.group(0))
        if not isinstance(parsed, list):
            raise TypeError(f"expected list literal, got {type(parsed)!r}")
        compounds.append([int(value) for value in parsed])
    return compounds


def deserialize_midi_payload(repr_name: str, payload: str) -> DecodedPayload:
    """Parse LLM completion text back into representation-specific token structures."""
    text = sanitize_midi_completion_text(payload)
    if not text:
        raise ValueError("empty MIDI payload")

    if repr_name == "octuple":
        if COMPOUND_SEP in text or FIELD_SEP in text:
            compounds: list[list[int]] = []
            for segment in text.split(COMPOUND_SEP):
                segment = segment.strip()
                if not segment:
                    continue
                if segment.startswith("["):
                    compounds.extend(_parse_legacy_compounds(segment))
                    continue
                compounds.append(
                    [int(value.strip()) for value in segment.split(FIELD_SEP) if value.strip()]
                )
            if not compounds:
                raise ValueError("octuple payload contained no compound tokens")
            return DecodedPayload(compound_token_ids=compounds)

        if _LEGACY_COMPOUND_RE.search(text):
            compounds = _parse_legacy_compounds(text)
            if compounds:
                return DecodedPayload(compound_token_ids=compounds)

        raise ValueError(
            "octuple payload must use compound format 'a,b,c|d,e,f' "
            "(legacy '[a,b,c] [d,e,f]' also accepted)"
        )

    if repr_name == "abc":
        return DecodedPayload(text=text)

    try:
        token_ids = [int(part) for part in text.split()]
    except ValueError as exc:
        raise ValueError(f"invalid flat token payload for {repr_name!r}") from exc
    if not token_ids:
        raise ValueError(f"empty flat token payload for {repr_name!r}")
    return DecodedPayload(token_ids=token_ids)
