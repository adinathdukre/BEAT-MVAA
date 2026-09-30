from __future__ import annotations

import copy
from typing import Iterator

import torch
import torch.nn as nn


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def fsdp_wrap(self, device, min_num_params: int = 10_000_000) -> None:
        from common.dist import wrap_fsdp

        self.ema = wrap_fsdp(self.ema, device, min_num_params=min_num_params, mixed_precision=True)

    def full_state_dict(self):
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        if isinstance(self.ema, FSDP):
            from common.dist import fsdp_full_state_dict
            return fsdp_full_state_dict(self.ema)
        return self.ema.state_dict()

    @torch.no_grad()
    def update(self, model: nn.Module, step: int | None = None) -> None:
        d = self.decay
        if step is not None:
            d = min(self.decay, (1 + step) / (10 + step))
        for ema_p, p in zip(self.ema.parameters(), model.parameters()):
            ema_p.mul_(d).add_(p.detach(), alpha=1 - d)
        for ema_b, b in zip(self.ema.buffers(), model.buffers()):
            ema_b.copy_(b)

    def parameters(self) -> Iterator[torch.nn.Parameter]:
        return self.ema.parameters()

    def __call__(self, *args, **kwargs):
        return self.ema(*args, **kwargs)

    def state_dict(self):
        return self.ema.state_dict()
