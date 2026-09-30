from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import torch


def _edt(binary: np.ndarray, sampling: Optional[Sequence[float]]) -> np.ndarray:
    from scipy.ndimage import distance_transform_edt

    return distance_transform_edt(binary, sampling=sampling)


def mask_to_sdf(
    mask: np.ndarray,
    sampling: Optional[Sequence[float]] = None,
    normalize: bool = True,
) -> np.ndarray:
    mask = mask.astype(bool)
    if not mask.any():
        return np.ones_like(mask, dtype=np.float32)
    if mask.all():
        return -np.ones_like(mask, dtype=np.float32)

    pos = _edt(~mask, sampling)
    neg = _edt(mask, sampling)
    sdf = pos - neg

    if normalize:
        out = sdf.copy().astype(np.float32)
        pmax = sdf[sdf > 0].max() if (sdf > 0).any() else 1.0
        nmin = sdf[sdf < 0].min() if (sdf < 0).any() else -1.0
        out[sdf > 0] = sdf[sdf > 0] / (pmax + 1e-8)
        out[sdf < 0] = sdf[sdf < 0] / (abs(nmin) + 1e-8)
        return out
    return sdf.astype(np.float32)


def labels_to_sdf(
    labels: np.ndarray,
    num_classes: int,
    sampling: Optional[Sequence[float]] = None,
    normalize: bool = True,
) -> np.ndarray:
    chans = []
    for c in range(1, max(num_classes, 2)):
        chans.append(mask_to_sdf(labels == c, sampling, normalize))
    return np.stack(chans, axis=0).astype(np.float32)


def batch_labels_to_sdf(
    labels: torch.Tensor,
    num_classes: int,
    sampling: Optional[Sequence[float]] = None,
    normalize: bool = True,
) -> torch.Tensor:
    arr = labels.detach().cpu().numpy()
    out = np.stack(
        [labels_to_sdf(arr[b], num_classes, sampling, normalize) for b in range(arr.shape[0])],
        axis=0,
    )
    return torch.from_numpy(out).to(labels.device, dtype=torch.float32)
