"""TrainingArguments warmup kwarg compatibility."""

from midi_llm.train.lora_sft import _warmup_scheduler_kwargs


def test_warmup_ratio_when_supported() -> None:
    class V4TrainingArgs:
        def __init__(self, *, warmup_ratio: float = 0, max_steps: int = 0) -> None:
            pass

    assert _warmup_scheduler_kwargs(V4TrainingArgs, warmup_ratio=0.03, max_steps=100) == {
        "warmup_ratio": 0.03
    }


def test_warmup_steps_float_for_transformers_v5() -> None:
    class V5TrainingArgs:
        def __init__(
            self,
            *,
            warmup_steps: float = 0,
            max_steps: int = 0,
        ) -> None:
            pass

    assert _warmup_scheduler_kwargs(V5TrainingArgs, warmup_ratio=0.03, max_steps=100) == {
        "warmup_steps": 0.03
    }


def test_warmup_steps_int_fallback() -> None:
    class LegacyArgs:
        def __init__(self, *, max_steps: int = 0) -> None:
            pass

    assert _warmup_scheduler_kwargs(LegacyArgs, warmup_ratio=0.03, max_steps=100) == {
        "warmup_steps": 3
    }
