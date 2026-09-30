from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ShapePriorAE(nn.Module):
    def __init__(self, dims: int = 3, base: int = 16, depth: int = 3, in_ch: int = 1):
        super().__init__()
        self.dims = dims
        Conv = nn.Conv3d if dims == 3 else nn.Conv2d
        ConvT = nn.ConvTranspose3d if dims == 3 else nn.ConvTranspose2d
        Norm = nn.InstanceNorm3d if dims == 3 else nn.InstanceNorm2d

        def cbr(ci, co, stride=1, transpose=False):
            conv = (ConvT(ci, co, 2, stride=2) if transpose
                    else Conv(ci, co, 3, stride=stride, padding=1))
            return nn.Sequential(conv, Norm(co, affine=True), nn.LeakyReLU(0.1, inplace=True))

        chs = [base * (2 ** i) for i in range(depth)]
        enc, c = [], in_ch
        for co in chs:
            enc += [cbr(c, co), cbr(co, co, stride=2)]
            c = co
        self.enc = nn.Sequential(*enc)
        self.mid = cbr(c, c)
        dec = []
        for co in reversed(chs[:-1] + [base]):
            dec += [cbr(c, co, transpose=True), cbr(co, co)]
            c = co
        self.dec = nn.Sequential(*dec)
        self.head = Conv(c, in_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[2:]
        h = self.dec(self.mid(self.enc(x)))
        h = self.head(h)
        if h.shape[2:] != size:
            mode = "trilinear" if self.dims == 3 else "bilinear"
            h = F.interpolate(h, size=size, mode=mode, align_corners=False)
        return h


class ShapeConsistencyLoss(nn.Module):
    def __init__(self, prior: ShapePriorAE):
        super().__init__()
        self.prior = prior.eval()
        for p in self.prior.parameters():
            p.requires_grad_(False)

    def forward(self, logits: torch.Tensor, target: Optional[torch.Tensor] = None) -> torch.Tensor:
        p = torch.softmax(logits, dim=1)[:, 1:].sum(1, keepdim=True).clamp(0, 1)
        with torch.no_grad():
            recon = torch.sigmoid(self.prior(p))
        return F.l1_loss(p, recon)


def load_shape_prior(ckpt: Optional[str], dims: int, device) -> Optional[ShapePriorAE]:
    if not ckpt or not Path(str(ckpt)).exists():
        return None
    from common.io import load_checkpoint
    from common.log import report_load

    state = load_checkpoint(ckpt, map_location=device)
    cfg = state.get("ae", {}) if isinstance(state, dict) else {}
    prior = ShapePriorAE(dims=dims, base=cfg.get("base", 16), depth=cfg.get("depth", 3))
    sd = state["model"] if isinstance(state, dict) and "model" in state else state
    report_load(prior, sd, name=f"shape_prior_{dims}d")
    prior.to(device).eval()
    for p in prior.parameters():
        p.requires_grad_(False)
    return prior


def load_for_predict(cfg, dims: int, device):
    pred = cfg.get("predict", {}) if hasattr(cfg, "get") else {}
    sp = pred.get("shape_prior", {}) if hasattr(pred, "get") else {}
    if not sp or not sp.get("enabled", False):
        return None, None
    prior = load_shape_prior(sp.get("ckpt"), dims, device)
    size = tuple(sp.get("size")) if sp.get("size") else None
    return prior, size


@torch.inference_mode()
def refine(label: np.ndarray, num_classes: int, prior: Optional[ShapePriorAE], device,
           size=None, fallback_lcc: bool = True) -> np.ndarray:
    if prior is None:
        if fallback_lcc:
            from common.postprocess import keep_largest_per_class

            return keep_largest_per_class(label, num_classes)
        return label

    dims = label.ndim
    mode = "trilinear" if dims == 3 else "bilinear"
    out = np.zeros_like(label)
    for c in range(1, max(num_classes, 2)):
        m = (label == c).astype(np.float32)
        if m.sum() == 0:
            continue
        t = torch.from_numpy(m)[None, None].to(device)
        canon = tuple(size) if size is not None else label.shape
        if tuple(canon) != tuple(label.shape):
            t = F.interpolate(t, size=canon, mode="nearest")
        r = torch.sigmoid(prior(t.float()))
        if r.shape[2:] != label.shape:
            r = F.interpolate(r, size=label.shape, mode=mode, align_corners=False)
        out[r[0, 0].detach().cpu().numpy() > 0.5] = c
    return out


def corrupt_mask(mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    from scipy.ndimage import binary_dilation, binary_erosion, generate_binary_structure

    m = mask.astype(bool).copy()
    st = generate_binary_structure(m.ndim, 1)
    r = rng.integers(0, 3)
    if r == 1:
        m = binary_dilation(m, st, iterations=int(rng.integers(1, 3)))
    elif r == 2:
        m = binary_erosion(m, st, iterations=int(rng.integers(1, 3)))
    if rng.random() < 0.5 and m.any():
        idx = np.argwhere(m)
        lo = idx.min(0); hi = idx.max(0)
        c = [int(rng.integers(lo[d], hi[d] + 1)) for d in range(m.ndim)]
        sl = tuple(slice(c[d], c[d] + max(1, (hi[d] - lo[d]) // 4)) for d in range(m.ndim))
        m[sl] = False
    if rng.random() < 0.5:
        c = [int(rng.integers(0, s)) for s in m.shape]
        sl = tuple(slice(max(0, c[d] - 2), c[d] + 3) for d in range(m.ndim))
        m[sl] = True
    return m.astype(np.float32)
