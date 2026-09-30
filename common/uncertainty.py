from __future__ import annotations

import numpy as np


def boundary_ring_np(fg: np.ndarray, band: int = 2) -> np.ndarray:
    from scipy.ndimage import binary_dilation, binary_erosion, generate_binary_structure

    fg = fg.astype(bool)
    if not fg.any() or band <= 0:
        return np.zeros_like(fg)
    st = generate_binary_structure(fg.ndim, 1)
    return binary_dilation(fg, st, iterations=band) & ~binary_erosion(fg, st, iterations=band)


def reliable_keep_np(conf: np.ndarray, label: np.ndarray, tau: float,
                     boundary_tau: float, band: int = 2) -> np.ndarray:
    keep = conf >= tau
    ring = boundary_ring_np(label > 0, band)
    return keep & (~ring | (conf >= boundary_tau))


def boundary_ring_torch(fg, band: int = 2):
    import torch.nn.functional as F

    if band <= 0:
        return fg.bool() & ~fg.bool()
    k = 2 * band + 1
    if fg.ndim == 4:
        dil = F.max_pool2d(fg, k, 1, band)
        ero = -F.max_pool2d(-fg, k, 1, band)
    else:
        dil = F.max_pool3d(fg, k, 1, band)
        ero = -F.max_pool3d(-fg, k, 1, band)
    return (dil > 0.5) & (ero < 0.5)


def reliable_keep_torch(conf, pseudo, tau: float, boundary_tau: float, band: int = 2):
    fg = (pseudo > 0).float().unsqueeze(1)
    ring = boundary_ring_torch(fg, band)[:, 0]
    return (conf >= tau) & (~ring | (conf >= boundary_tau))
