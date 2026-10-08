"""LoRA supervised fine-tuning for Stage S3 (and small pilot runs)."""

from __future__ import annotations

import inspect
import json
import logging
from pathlib import Path
from typing import Any

from midi_llm.config import load_config
from midi_llm.config_helpers import cap_s3_max_steps, resolve_s3_grad_accum_steps
from midi_llm.reproducibility import set_global_seed
from midi_llm.train.dataset import SFTJsonlDataset, format_llama_instruct, load_sft_records
from midi_llm.train.sft_tokenize import SFTExampleUnfit, tokenize_sft_example
from midi_llm.train.shard_meta import resolve_repr_name
from midi_llm.train.shard_paths import infer_validation_shard

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
    require_full_fit: bool = True,
) -> tuple[list[dict[str, Any]], int, int]:
    """Drop rows that cannot fit ``max_seq_len`` with the edit instruction preserved."""
    kept: list[dict[str, Any]] = []
    skipped_unfit = 0
    skipped_truncated = 0
    for record in records:
        prompt = record.get("prompt", "")
        completion = record.get("completion", "")
        if chat_template:
            parts = format_llama_instruct(prompt, completion)
            prompt, completion = parts["prompt"], parts["completion"]
        try:
            _ids, _label, meta = tokenize_sft_example(
                tokenizer,
                prompt=prompt,
                completion=completion,
                max_seq_len=max_seq_len,
            )
        except (SFTExampleUnfit, ValueError):
            skipped_unfit += 1
            continue
        if require_full_fit and (meta.prompt_truncated or meta.completion_truncated):
            skipped_truncated += 1
            continue
        kept.append(record)
    return kept, skipped_unfit, skipped_truncated


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
    max_epochs: float | None = None,
    max_seq_len: int | None = None,
    max_samples: int | None = None,
    val_shard_path: str | Path | None = None,
    per_device_batch_size: int = 1,
    gradient_accumulation_steps: int | None = None,
    learning_rate: float | None = None,
    weight_decay: float | None = None,
    precision: str | None = None,
    seed: int | None = None,
    logging_steps: int = 10,
    save_steps: int = 200,
    early_stopping_patience: int | None = None,
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
    train_cfg = cfg["training"]
    lr = learning_rate or train_cfg["learning_rate"]
    requested_steps = max_steps if max_steps is not None else train_cfg["max_steps"]["s3_edit_lora"]
    epochs = float(max_epochs if max_epochs is not None else train_cfg.get("s3_max_epochs", 3))
    wd = weight_decay if weight_decay is not None else float(train_cfg.get("weight_decay", 0.0))
    lora_cfg = cfg["lora"]
    grad_accum = resolve_s3_grad_accum_steps(
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
    filtered_records, n_skipped_unfit, n_skipped_truncated = filter_sft_records_for_seq_len(
        raw_records,
        tokenizer,
        max_seq_len=seq_len,
        chat_template=True,
        require_full_fit=True,
    )
    if n_skipped_unfit:
        logger.warning(
            "Dropped %s/%s SFT rows (unfit at max_seq_len=%s)",
            n_skipped_unfit,
            len(raw_records),
            seq_len,
        )
    if n_skipped_truncated:
        logger.warning(
            "Dropped %s/%s SFT rows (prompt/completion truncated at max_seq_len=%s)",
            n_skipped_truncated,
            len(raw_records),
            seq_len,
        )
    if not filtered_records:
        raise RuntimeError(
            f"No SFT rows fit max_seq_len={seq_len} in {shard_path}. "
            "Increase max_seq_len (protocol default 2048) or rebuild shards with shorter MIDI."
        )
    n_train = len(filtered_records)
    steps, steps_per_epoch = cap_s3_max_steps(
        n_train,
        batch_size=per_device_batch_size,
        grad_accum=grad_accum,
        max_epochs=epochs,
        requested_steps=requested_steps,
    )
    if steps < requested_steps:
        logger.warning(
            "Capped max_steps %s → %s (~%.1f epochs over %s train rows; s3_max_epochs=%s)",
            requested_steps,
            steps,
            steps / steps_per_epoch,
            n_train,
            epochs,
        )

    dataset = SFTJsonlDataset(shard_path, chat_template=True, records=filtered_records)

    val_records: list[dict[str, Any]] | None = None
    val_path = Path(val_shard_path) if val_shard_path else infer_validation_shard(shard_path)
    if val_path is not None and val_path.is_file():
        raw_val = load_sft_records(val_path)
        val_records, val_unfit, val_trunc = filter_sft_records_for_seq_len(
            raw_val,
            tokenizer,
            max_seq_len=seq_len,
            chat_template=True,
            require_full_fit=True,
        )
        if val_unfit or val_trunc:
            logger.warning(
                "Validation shard dropped unfit=%s truncated=%s (kept %s)",
                val_unfit,
                val_trunc,
                len(val_records),
            )
        if not val_records:
            val_records = None
            val_path = None

    eval_dataset = None
    if val_records:
        eval_dataset = SFTJsonlDataset(val_path, chat_template=True, records=val_records)

    prec = precision or cfg["model"].get("precision", "bf16")
    use_bf16 = prec == "bf16"
    dtype = torch.bfloat16 if use_bf16 else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
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
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )

    warmup_kwargs = _warmup_scheduler_kwargs(
        TrainingArguments,
        warmup_ratio=float(cfg["training"]["warmup_ratio"]),
        max_steps=steps,
    )
    ta_params = inspect.signature(TrainingArguments.__init__).parameters
    extra_ta: dict[str, Any] = {}
    if "gradient_checkpointing" in ta_params:
        extra_ta["gradient_checkpointing"] = True
    if "dataloader_pin_memory" in ta_params:
        extra_ta["dataloader_pin_memory"] = False
    if "weight_decay" in ta_params:
        extra_ta["weight_decay"] = wd

    eval_steps = max(5, min(save_steps, max(1, steps // 10)))
    if eval_dataset is not None:
        if "eval_strategy" in ta_params:
            extra_ta["eval_strategy"] = "steps"
            extra_ta["eval_steps"] = eval_steps
        elif "evaluation_strategy" in ta_params:
            extra_ta["evaluation_strategy"] = "steps"
            extra_ta["eval_steps"] = eval_steps
        if "load_best_model_at_end" in ta_params:
            extra_ta["load_best_model_at_end"] = True
        if "metric_for_best_model" in ta_params:
            extra_ta["metric_for_best_model"] = "eval_loss"
        if "greater_is_better" in ta_params:
            extra_ta["greater_is_better"] = False

    args = TrainingArguments(
        output_dir=str(out),
        max_steps=steps,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        **warmup_kwargs,
        lr_scheduler_type=train_cfg["lr_schedule"],
        logging_steps=logging_steps,
        save_steps=save_steps,
        save_total_limit=2,
        bf16=use_bf16,
        fp16=not use_bf16,
        seed=train_seed,
        data_seed=train_seed,
        report_to=[],
        remove_unused_columns=False,
        **extra_ta,
    )

    logger.info(
        "LoRA SFT: seq_len=%s batch=%s grad_accum=%s steps=%s (~%.2f epochs) "
        "train_rows=%s (from %s shard rows)%s",
        seq_len,
        per_device_batch_size,
        grad_accum,
        steps,
        steps / steps_per_epoch,
        n_train,
        len(raw_records),
        f" val_rows={len(val_records)}" if val_records else "",
    )

    callbacks = []
    patience = early_stopping_patience
    if patience is None:
        patience = int(train_cfg.get("s3_early_stopping_patience", 0))
    if eval_dataset is not None and patience > 0:
        from transformers import EarlyStoppingCallback

        callbacks.append(EarlyStoppingCallback(early_stopping_patience=patience))

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        data_collator=_SFTCollator(tokenizer, seq_len),
        callbacks=callbacks or None,
    )
    meta = {
        "repr_name": resolved_repr,
        "shard_path": str(shard_path),
        "val_shard_path": str(val_path) if val_records and val_path else None,
        "model": model_name,
        "seed": train_seed,
        "max_steps": steps,
        "max_steps_requested": requested_steps,
        "s3_max_epochs": epochs,
        "s3_steps_per_epoch": steps_per_epoch,
        "max_seq_len": seq_len,
        "precision": prec,
        "weight_decay": wd,
        "gradient_accumulation_steps": grad_accum,
        "sft_rows_kept": n_train,
        "sft_rows_skipped_unfit": n_skipped_unfit,
        "sft_rows_skipped_truncated": n_skipped_truncated,
        "sft_val_rows_kept": len(val_records) if val_records else 0,
        "early_stopping_patience": patience if eval_dataset is not None else 0,
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
    max_epochs: float | None = None,
    max_seq_len: int | None = None,
    max_samples: int | None = None,
    val_shard_path: str | Path | None = None,
    per_device_batch_size: int = 1,
    gradient_accumulation_steps: int | None = None,
    learning_rate: float | None = None,
    weight_decay: float | None = None,
    precision: str | None = None,
    seed: int | None = None,
    logging_steps: int = 10,
    save_steps: int = 200,
    early_stopping_patience: int | None = None,
) -> Path:
    """Fine-tune Llama with LoRA on a midi-llm JSONL shard."""
    trainer, out, meta = setup_lora_trainer(
        shard_path,
        output_dir,
        repr_name=repr_name,
        max_steps=max_steps,
        max_epochs=max_epochs,
        max_seq_len=max_seq_len,
        max_samples=max_samples,
        val_shard_path=val_shard_path,
        per_device_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        precision=precision,
        seed=seed,
        logging_steps=logging_steps,
        save_steps=save_steps,
        early_stopping_patience=early_stopping_patience,
    )
    trainer.train()
    save_lora_artifacts(trainer, out, meta)
    return out
