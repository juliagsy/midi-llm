"""Deterministic seeds for training and inference."""

from __future__ import annotations


def set_global_seed(seed: int) -> None:
    """Set Python, NumPy, and PyTorch RNG seeds."""
    import random

    random.seed(seed)

    try:
        import numpy as np
    except ImportError:
        pass
    else:
        np.random.seed(seed)

    try:
        import torch
    except ImportError:
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
