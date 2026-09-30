from __future__ import annotations

import torch


def _haar_kernels(channels: int, dims: int, device, dtype):
    lo = torch.tensor([1.0, 1.0], device=device, dtype=dtype) / (2.0 ** 0.5)
    hi = torch.tensor([1.0, -1.0], device=device, dtype=dtype) / (2.0 ** 0.5)
    banks = [lo, hi]
    if dims == 2:
        filt = [torch.outer(a, b) for a in banks for b in banks]
        k = torch.stack(filt, 0).view(4, 1, 2, 2)
        return k.repeat(channels, 1, 1, 1)
    filt = [torch.einsum("i,j,k->ijk", a, b, c) for a in banks for b in banks for c in banks]
    k = torch.stack(filt, 0).view(8, 1, 2, 2, 2)
    return k.repeat(channels, 1, 1, 1, 1)
