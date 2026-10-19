"""fast_hadamard_transform.py — pure-PyTorch stand-in for the CUDA extension.

The reference `rotate_activation` (model.py:253-257) does:
    hadamard_transform(x, scale=x.size(-1) ** -0.5)
i.e. multiply the last dim by a (normalized) Walsh-Hadamard matrix. The reference asserts the
last-dim size is a power of two (indexer/compressor head_dims 128/256/512 all are). Golden reference
only — correctness over speed.
"""
import torch


def _hadamard_matrix(n: int, device, dtype=torch.float32) -> torch.Tensor:
    assert (n & (n - 1)) == 0, f"Hadamard size must be a power of two, got {n}"
    H = torch.ones((1, 1), device=device, dtype=dtype)
    while H.size(0) < n:
        H = torch.cat([torch.cat([H, H], dim=1),
                       torch.cat([H, -H], dim=1)], dim=0)
    return H


def hadamard_transform(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    n = x.size(-1)
    H = _hadamard_matrix(n, x.device, torch.float32)
    y = (x.float() @ H) * scale
    return y.to(x.dtype)
