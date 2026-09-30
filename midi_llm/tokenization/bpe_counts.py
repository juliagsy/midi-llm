"""Count Llama BPE tokens for serialized MIDI payloads."""

from __future__ import annotations

from functools import lru_cache

from midi_llm.midi_repr.base import EncodeResult


def serialized_payload(encoded: EncodeResult) -> str:
    """Text form fed to the LLM (space-separated MIDI token IDs or ABC text)."""
    if encoded.text is not None:
        return encoded.text.strip()
    if encoded.token_ids is not None:
        return " ".join(str(token_id) for token_id in encoded.token_ids)
    return ""


@lru_cache(maxsize=4)
def _load_tokenizer(model_name: str):
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "BPE token counting requires transformers: pip install -e '.[train]'"
        ) from exc
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def count_llama_bpe_tokens(text: str, model_name: str) -> int:
    """Return Llama BPE length for a serialized MIDI payload string."""
    if not text:
        return 0
    tokenizer = _load_tokenizer(model_name)
    return len(tokenizer.encode(text, add_special_tokens=False))
