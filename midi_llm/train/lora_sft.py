"""LoRA supervised fine-tuning for Stage S3 (and small pilot runs)."""

from __future__ import annotations

import inspect
import json
import logging
from pathlib import Path
from typing import Any

from midi_llm.config import load_config
from midi_llm.config_helpers import resolve_grad_accum_steps
from midi_llm.reproducibility import set_global_seed
from midi_llm.train.dataset import SFTJsonlDataset, format_llama_instruct, load_sft_records
from midi_llm.train.sft_tokenize import SFTExampleUnfit, tokenize_sft_example
from midi_llm.train.shard_meta import resolve_repr_name

logger = logging.getLogger(__name__)


def _warmup_scheduler_kwargs(
    training_args_cls: type,
    *,
    warmup_ratio: float,
    max_steps: int,
) -> dict[str, float | int]:
    """Support transformers v4 (warmup_ratio) and v5 (warmup_steps float ratio)."""
    params = inspect.signature(training_args_cls.__init__).parameters
    if "warmup_ratio" in params:
        return {"warmup_ratio": warmup_ratio}
    if "warmup_steps" in params:
        return {"warmup_steps": warmup_ratio}
    return {"warmup_steps": max(0, int(max_steps * warmup_ratio))}


def _require_train_deps():
    try:
        import torch
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
    except ImportError as exc:
        raise ImportError("Training requires: pip install -e '.[train]'") from exc
    return torch, LoraConfig, TaskType, get_peft_model, AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments


def filter_sft_records_for_seq_len(
    records: list[dict[str, Any]],
    tokenizer,
    *,
    max_seq_len: int,
    chat_template: bool = True,
) -> tuple[list[dict[str, Any]], int]:
    """Drop rows that cannot fit ``max_seq_len`` with the edit instruction preserved."""
    kept: list[dict[str, Any]] = []
    skipped = 0
    for record in records:
        prompt = record.get("prompt", "")
        completion = record.get("completion", "")
        if chat_template:
            parts = format_llama_instruct(prompt, completion)
            prompt, completion = parts["prompt"], parts["completion"]
        try:
            tokenize_sft_example(
                tokenizer,
                prompt=prompt,
                completion=completion,
                max_seq_len=max_seq_len,
            )
        except (SFTExampleUnfit, ValueError):
            skipped += 1
            continue
        kept.append(record)
    return kept, skipped


def _tokenize_batch(examples: dict[str, list[str]], tokenizer, max_seq_len: int) -> dict[str, Any]:
    torch, *_ = _require_train_deps()
    input_ids = []
    labels = []
    for prompt, completion in zip(examples["prompt"], examples["completion"], strict=True):
        try:
            ids, label, _meta = tokenize_sft_example(
                tokenizer,
                prompt=prompt,
                completion=completion,
                max_seq_len=max_seq_len,
            )
        except (SFTExampleUnfit, ValueError):
            continue
        input_ids.append(ids)
        labels.append(label)

    if not input_ids:
        raise RuntimeError(
            f"No examples in batch fit max_seq_len={max_seq_len}. "
            "Increase MAX_SEQ_LEN (real v0.2: 2048) or re-run setup after updating midi-llm."
        )

    max_len = max(len(row) for row in input_ids)
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    batch_input = []
    batch_labels = []
    batch_mask = []
    for ids, label in zip(input_ids, labels, strict=True):
        pad = max_len - len(ids)
        batch_input.append(ids + [pad_id] * pad)
        batch_labels.append(label + [-100] * pad)
        batch_mask.append([1] * len(ids) + [0] * pad)

    return {
        "input_ids": torch.tensor(batch_input, dtype=torch.long),
        "attention_mask": torch.tensor(batch_mask, dtype=torch.long),
        "labels": torch.tensor(batch_labels, dtype=torch.long),
    }


class _SFTCollator:
    def __init__(self, tokenizer, max_seq_len: int) -> None:
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __call__(self, batch: list[dict[str, str]]) -> dict[str, Any]:
        merged = {
            "prompt": [row["prompt"] for row in batch],
            "completion": [row["completion"] for row in batch],
        }
        return _tokenize_batch(merged, self.tokenizer, self.max_seq_len)


def setup_lora_trainer(
    shard_path: str | Path,
    output_dir: str | Path,
    *,
    repr_name: str | None = None,
    max_steps: int | None = None,
    max_seq_len: int | None = None,
    max_samples: int | None = None,
    per_device_batch_size: int = 1,
    gradient_accumulation_steps: int | None = None,
    learning_rate: float | None = None,
    precision: str | None = None,
    seed: int | None = None,
    logging_steps: int = 10,
    save_steps: int = 200,
):
    """Build a HuggingFace Trainer for LoRA SFT (call ``save_lora_artifacts`` after train)."""
    (
        torch,
        LoraConfig,
        TaskType,
        get_peft_model,
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    ) = _require_train_deps()

    resolved_repr = resolve_repr_name(shard_path, repr_name)
    cfg = load_config(resolved_repr)
    train_seed = 42 if seed is None else seed
    set_global_seed(train_seed)

    model_name = cfg["model"]["backbone"]
    seq_len = max_seq_len or cfg["model"]["max_seq_len"]
    lr = learning_rate or cfg["training"]["learning_rate"]
    steps = max_steps or cfg["training"]["max_steps"]["s3_edit_lora"]
    lora_cfg = cfg["lora"]
    grad_accum = resolve_grad_accum_steps(
        cfg,
        seq_len=seq_len,
        batch_size=per_device_batch_size,
        override=gradient_accumulation_steps,
    )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    raw_records = load_sft_records(shard_path, max_samples=max_samples)
    filtered_records, n_skipped = filter_sft_records_for_seq_len(
        raw_records,
        tokenizer,
        max_seq_len=seq_len,
        chat_template=True,
    )
    if n_skipped:
        logger.warning(
            "Dropped %s/%s SFT rows that exceed max_seq_len=%s (instruction + completion must fit)",
            n_skipped,
            len(raw_records),
            seq_len,
        )
    if not filtered_records:
        raise RuntimeError(
            f"No SFT rows fit max_seq_len={seq_len} in {shard_path}. "
            "Increase max_seq_len (protocol default 2048) or rebuild shards with shorter MIDI."
        )
    dataset = SFTJsonlDataset(shard_path, chat_template=True, records=filtered_records)

    prec = precision or cfg["model"].get("precision", "bf16")
    use_bf16 = prec == "bf16"
    dtype = torch.bfloat16 if use_bf16 else torch.float16
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
    model.config.use_cache = False

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_cfg["rank"],
        lora_alpha=lora_cfg["alpha"],
        target_modules=lora_cfg["target_modules"],
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(model, peft_config)

    warmup_kwargs = _warmup_scheduler_kwargs(
        TrainingArguments,
        warmup_ratio=float(cfg["training"]["warmup_ratio"]),
        max_steps=steps,
    )
    args = TrainingArguments(
        output_dir=str(out),
        max_steps=steps,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        **warmup_kwargs,
        lr_scheduler_type=cfg["training"]["lr_schedule"],
        logging_steps=logging_steps,
        save_steps=save_steps,
        save_total_limit=2,
        bf16=use_bf16,
        fp16=not use_bf16,
        seed=train_seed,
        data_seed=train_seed,
        report_to=[],
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=_SFTCollator(tokenizer, seq_len),
    )
    meta = {
        "repr_name": resolved_repr,
        "shard_path": str(shard_path),
        "model": model_name,
        "seed": train_seed,
        "max_steps": steps,
        "max_seq_len": seq_len,
        "precision": prec,
        "gradient_accumulation_steps": grad_accum,
        "sft_rows_kept": len(filtered_records),
        "sft_rows_skipped_seq_len": n_skipped,
    }
    return trainer, out, meta


def save_lora_artifacts(trainer, output_dir: str | Path, meta: dict[str, Any]) -> Path:
    """Write ``lora_adapter/``, tokenizer, and ``train_meta.json`` from a Trainer."""
    out = Path(output_dir)
    adapter_dir = out / "lora_adapter"
    trainer.save_model(str(adapter_dir))
    trainer.tokenizer.save_pretrained(out / "tokenizer")
    payload = {
        **meta,
        "saved_global_step": int(getattr(trainer.state, "global_step", 0)),
    }
    (out / "train_meta.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return adapter_dir


def train_lora_sft(
    shard_path: str | Path,
    output_dir: str | Path,
    *,
    repr_name: str | None = None,
    max_steps: int | None = None,
    max_seq_len: int | None = None,
    max_samples: int | None = None,
    per_device_batch_size: int = 1,
    gradient_accumulation_steps: int | None = None,
    learning_rate: float | None = None,
    precision: str | None = None,
    seed: int | None = None,
    logging_steps: int = 10,
    save_steps: int = 200,
) -> Path:
    """Fine-tune Llama with LoRA on a midi-llm JSONL shard."""
    trainer, out, meta = setup_lora_trainer(
        shard_path,
        output_dir,
        repr_name=repr_name,
        max_steps=max_steps,
        max_seq_len=max_seq_len,
        max_samples=max_samples,
        per_device_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        precision=precision,
        seed=seed,
        logging_steps=logging_steps,
        save_steps=save_steps,
    )
    trainer.train()
    save_lora_artifacts(trainer, out, meta)
    return out
