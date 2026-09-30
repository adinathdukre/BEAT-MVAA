from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GINGroupConv(nn.Module):
    def __init__(self, in_ch: int = 3, n_layer: int = 4, interm_ch: int = 2, kernels=(1, 3)):
        super().__init__()
        self.in_ch, self.n_layer, self.interm_ch = in_ch, n_layer, interm_ch
        self.kernels = tuple(kernels)
        self.act = nn.LeakyReLU(0.2)

    @torch.no_grad()
    def _one(self, xb: torch.Tensor) -> torch.Tensor:
        chs = [self.in_ch] + [self.interm_ch] * (self.n_layer - 1) + [self.in_ch]
        out = xb
        for i in range(self.n_layer):
            k = int(self.kernels[int(torch.randint(len(self.kernels), ()))])
            w = torch.randn(chs[i + 1], chs[i], k, k, device=xb.device, dtype=xb.dtype)
            w = w * (2.0 / (chs[i] * k * k)) ** 0.5
            out = F.conv2d(out, w, padding=k // 2)
            if i < self.n_layer - 1:
                out = self.act(out)
        return out

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outs = []
        for b in range(x.shape[0]):
            xb = x[b:b + 1]
            g = self._one(xb)
            g = g * (xb.norm() / (g.norm() + 1e-6))
            a = float(torch.rand(()))
            outs.append(a * g + (1.0 - a) * xb)
        return torch.cat(outs, 0).clamp(0.0, 1.0)
