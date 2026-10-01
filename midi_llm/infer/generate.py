"""Model loading and text generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GenerateResult:
    text: str
    n_new_tokens: int
    hit_max_new_tokens: bool


def _require_infer_deps():
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise ImportError("Inference requires: pip install -e '.[train]'") from exc
    return torch, PeftModel, AutoModelForCausalLM, AutoTokenizer


def resolve_device(torch) -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_causal_lm(
    model_name: str,
    *,
    adapter_path: str | Path | None = None,
    torch_dtype: str = "auto",
):
    torch, PeftModel, AutoModelForCausalLM, AutoTokenizer = _require_infer_deps()

    dtype_map = {
        "auto": "auto",
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    dtype = dtype_map.get(torch_dtype, "auto")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = resolve_device(torch)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    if adapter_path is not None:
        model = PeftModel.from_pretrained(model, str(adapter_path))

    if device == "cuda":
        model = model.to("cuda")
    elif device == "mps":
        model = model.to("mps")

    model.eval()
    return model, tokenizer, device


def generate_completion(
    model,
    tokenizer,
    prompt: str,
    *,
    max_new_tokens: int = 512,
    temperature: float = 0.2,
    top_p: float = 0.95,
    seed: int | None = None,
) -> GenerateResult:
    torch, *_ = _require_infer_deps()
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    generator = None
    if seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            top_p=top_p if temperature > 0 else None,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            generator=generator,
        )

    new_tokens = output[0, inputs["input_ids"].shape[1] :]
    n_new = int(new_tokens.shape[0])
    hit_max = n_new >= max_new_tokens
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return GenerateResult(text=text, n_new_tokens=n_new, hit_max_new_tokens=hit_max)
