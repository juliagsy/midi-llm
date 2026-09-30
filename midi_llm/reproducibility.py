"""Deterministic seeds for training and inference."""

from __future__ import annotations

import hashlib


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
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def item_seed(base_seed: int, item_id: str) -> int:
    """Derive a stable per-item seed from a run seed and benchmark item id."""
    digest = hashlib.sha256(f"{base_seed}:{item_id}".encode()).hexdigest()
    return int(digest[:8], 16)
