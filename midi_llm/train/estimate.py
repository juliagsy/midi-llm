"""Estimate training duration from hardware profile and run optional micro-benchmark."""

from __future__ import annotations

import json
import platform
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from midi_llm.config import load_config


@dataclass
class HardwareProfile:
    machine: str
    cpu: str
    ram_gb: float | None
    gpu: str | None
    torch_device: str | None
    vram_gb: float | None


@dataclass
class TrainingEstimate:
    stage: str
    steps: int
    samples: int
    seq_len: int
    batch_size: int
    grad_accum: int
    effective_batch: int
    seconds_per_step: float | None
    estimated_hours: float
    notes: str


def _ram_gb() -> float | None:
    try:
        page_size = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"]).strip())
        return round(page_size / (1024**3), 1)
    except Exception:
        return None


def detect_hardware() -> HardwareProfile:
    machine = platform.machine()
    cpu = platform.processor() or machine
    ram = _ram_gb()
    gpu = None
    torch_device = None
    vram = None

    try:
        import torch

        if torch.cuda.is_available():
            torch_device = "cuda"
            gpu = torch.cuda.get_device_name(0)
            vram = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 1)
        elif torch.backends.mps.is_available():
            torch_device = "mps"
            gpu = "Apple Metal (MPS)"
            if ram is not None:
                vram = max(0.0, ram - 2.0)  # unified memory heuristic
        else:
            torch_device = "cpu"
    except ImportError:
        pass

    if gpu is None:
        try:
            out = subprocess.check_output(["system_profiler", "SPDisplaysDataType"], text=True)
            if "Apple M" in out:
                gpu = "Apple M-series (integrated)"
        except Exception:
            pass

    return HardwareProfile(
        machine=f"{platform.system()} {platform.release()}",
        cpu=cpu,
        ram_gb=ram,
        gpu=gpu,
        torch_device=torch_device,
        vram_gb=vram,
    )


def _fallback_secs_per_step(profile: HardwareProfile, model_params_b: float = 1.0) -> float:
    """Conservative heuristics when micro-benchmark is unavailable."""
    if profile.torch_device == "cuda":
        if profile.vram_gb and profile.vram_gb >= 40:
            return 0.8
        if profile.vram_gb and profile.vram_gb >= 16:
            return 1.5
        return 3.0
    if profile.torch_device == "mps":
        if profile.ram_gb and profile.ram_gb <= 8:
            return 12.0
        return 6.0
    return 25.0


def micro_benchmark_step(
    model_name: str,
    *,
    seq_len: int = 256,
    batch_size: int = 1,
) -> float | None:
    """Run one training-style forward+backward step; return seconds or None on failure."""
    try:
        import torch
        from transformers import AutoModelForCausalLM
    except ImportError:
        return None

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device != "cpu" else torch.float32,
            low_cpu_mem_usage=True,
        ).to(device)
        model.train()
        input_ids = torch.randint(0, 1000, (batch_size, seq_len), device=device)
        labels = input_ids.clone()

        if device == "mps":
            torch.mps.empty_cache()
        t0 = time.perf_counter()
        out = model(input_ids=input_ids, labels=labels)
        out.loss.backward()
        if device == "cuda":
            torch.cuda.synchronize()
        elif device == "mps":
            torch.mps.synchronize()
        elapsed = time.perf_counter() - t0
        del model
        if device == "mps":
            torch.mps.empty_cache()
        return elapsed
    except Exception:
        return None


def estimate_training(
    *,
    repr_name: str | None = None,
    stage: str = "s3_edit_lora",
    steps: int | None = None,
    samples: int = 224,
    seq_len: int = 512,
    batch_size: int = 1,
    grad_accum: int = 8,
    run_benchmark: bool = False,
    model_name: str | None = None,
) -> dict[str, Any]:
    cfg = load_config(repr_name)
    backbone = model_name or cfg["model"]["backbone"]
    step_key = stage if stage.startswith("s") else f"s{stage}"
    if steps is None:
        steps = cfg["training"]["max_steps"].get(step_key, cfg["training"]["max_steps"]["s3_edit_lora"])

    profile = detect_hardware()
    secs = micro_benchmark_step(backbone, seq_len=seq_len, batch_size=batch_size) if run_benchmark else None
    if secs is None:
        secs = _fallback_secs_per_step(profile)
        bench_note = "heuristic (run with --benchmark for measured)"
    else:
        bench_note = "measured micro-benchmark"

    effective_batch = batch_size * grad_accum
    hours = (steps * secs) / 3600.0

    est = TrainingEstimate(
        stage=step_key,
        steps=steps,
        samples=samples,
        seq_len=seq_len,
        batch_size=batch_size,
        grad_accum=grad_accum,
        effective_batch=effective_batch,
        seconds_per_step=round(secs, 3),
        estimated_hours=round(hours, 2),
        notes=bench_note,
    )

    warnings: list[str] = []
    if profile.torch_device == "mps" and profile.ram_gb and profile.ram_gb <= 8:
        warnings.append(
            "8 GB unified memory is tight for Llama-3.2-1B LoRA at seq_len>=1024; "
            "use max_seq_len=512, batch_size=1, and expect swapping."
        )
    if profile.torch_device in (None, "cpu"):
        warnings.append("No GPU/MPS detected; training will be impractically slow on CPU.")

    return {
        "hardware": asdict(profile),
        "model": backbone,
        "estimate": asdict(est),
        "warnings": warnings,
        "scenarios": [
            asdict(
                TrainingEstimate(
                    stage="pilot",
                    steps=50,
                    samples=samples,
                    seq_len=seq_len,
                    batch_size=batch_size,
                    grad_accum=grad_accum,
                    effective_batch=effective_batch,
                    seconds_per_step=round(secs, 3),
                    estimated_hours=round((50 * secs) / 3600.0, 2),
                    notes=bench_note,
                )
            ),
            asdict(est),
        ],
    }
