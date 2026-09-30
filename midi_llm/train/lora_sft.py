"""LoRA supervised fine-tuning for Stage S3 (and small pilot runs)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from midi_llm.config import load_config
from midi_llm.config_helpers import resolve_grad_accum_steps
from midi_llm.reproducibility import set_global_seed
from midi_llm.train.shard_meta import resolve_repr_name


def _require_train_deps():
    try:
        import torch
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
    except ImportError as exc:
        raise ImportError("Training requires: pip install -e '.[train]'") from exc
    return torch, LoraConfig, TaskType, get_peft_model, AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments


MIN_COMPLETION_TOKENS = 8


def _tokenize_sft_example(
    tokenizer,
    *,
    prompt: str,
    completion: str,
    max_seq_len: int,
) -> tuple[list[int], list[int]]:
    """Tokenize one SFT row, reserving completion tokens when truncating."""
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    completion_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]

    max_prompt_len = max(0, max_seq_len - MIN_COMPLETION_TOKENS)
    if len(prompt_ids) > max_prompt_len:
        prompt_ids = prompt_ids[-max_prompt_len:]

    ids = prompt_ids + completion_ids
    if len(ids) > max_seq_len:
        overflow = len(ids) - max_seq_len
        if overflow >= len(completion_ids):
            completion_ids = []
            ids = prompt_ids[-max_seq_len:]
        else:
            completion_ids = completion_ids[:-overflow]
            ids = prompt_ids + completion_ids

    prompt_len = len(ids) - len(completion_ids)
    label = ids.copy()
    label[:prompt_len] = [-100] * prompt_len
    if not any(token != -100 for token in label):
        raise ValueError("SFT example produced all-masked labels after truncation")
    return ids, label


def _tokenize_batch(examples: dict[str, list[str]], tokenizer, max_seq_len: int) -> dict[str, Any]:
    torch, *_ = _require_train_deps()
    input_ids = []
    labels = []
    for prompt, completion in zip(examples["prompt"], examples["completion"], strict=True):
        ids, label = _tokenize_sft_example(
            tokenizer,
            prompt=prompt,
            completion=completion,
            max_seq_len=max_seq_len,
        )
        input_ids.append(ids)
        labels.append(label)

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

    from midi_llm.train.dataset import SFTJsonlDataset

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

    dataset = SFTJsonlDataset(shard_path, max_samples=max_samples, chat_template=True)

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

    args = TrainingArguments(
        output_dir=str(out),
        max_steps=steps,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        warmup_ratio=cfg["training"]["warmup_ratio"],
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
    trainer.train()
    trainer.save_model(str(out / "lora_adapter"))
    tokenizer.save_pretrained(out / "tokenizer")
    meta = {
        "repr_name": resolved_repr,
        "shard_path": str(shard_path),
        "model": model_name,
        "seed": train_seed,
        "max_steps": steps,
        "max_seq_len": seq_len,
        "precision": prec,
        "gradient_accumulation_steps": grad_accum,
    }
    (out / "train_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return out
