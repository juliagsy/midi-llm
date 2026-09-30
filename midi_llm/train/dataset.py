"""JSONL datasets for supervised fine-tuning."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from midi_llm.data.jsonl_io import SFT_FIELDS, iter_jsonl

try:
    from torch.utils.data import Dataset
except ImportError:  # pragma: no cover - train extra not installed
    class Dataset:  # type: ignore[no-redef]
        """Fallback when torch is unavailable outside training."""


def format_example(prompt: str, completion: str) -> str:
    """Plain causal-LM text: prompt immediately followed by completion."""
    return f"{prompt}{completion}"


def format_llama_instruct(prompt: str, completion: str) -> dict[str, str]:
    """Chat-style fields for Llama 3.2 Instruct training."""
    user_block = prompt.strip() or "Continue the sequence."
    sh, eh = "start_header_id", "end_header_id"
    user_hdr = f"<|begin_of_text|><|{sh}|>user<|{eh}|>\n\n"
    asst_hdr = f"<|{sh}|>assistant<|{eh}|>\n\n"
    return {
        "prompt": f"{user_hdr}{user_block}<|eot_id|>{asst_hdr}",
        "completion": f"{completion.strip()}<|eot_id|>",
    }


class SFTJsonlDataset(Dataset):
    """Load midi-llm JSONL shards (prompt/completion records)."""

    def __init__(
        self,
        shard_path: str | Path,
        *,
        max_samples: int | None = None,
        chat_template: bool = True,
    ) -> None:
        self.records: list[dict[str, Any]] = []
        path = Path(shard_path)
        for _line_no, record in iter_jsonl(path, required_fields=SFT_FIELDS, label="SFT shard"):
            self.records.append(record)
            if max_samples is not None and len(self.records) >= max_samples:
                break
        if not self.records:
            raise RuntimeError(f"SFT shard is empty or invalid: {path}")
        self.chat_template = chat_template

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, str]:
        row = self.records[index]
        prompt = row.get("prompt", "")
        completion = row.get("completion", "")
        if self.chat_template:
            parts = format_llama_instruct(prompt, completion)
            return {
                "text": parts["prompt"] + parts["completion"],
                "prompt": parts["prompt"],
                "completion": parts["completion"],
            }
        return {"text": format_example(prompt, completion), "prompt": prompt, "completion": completion}
