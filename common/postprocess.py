from __future__ import annotations

from typing import Optional

import numpy as np


def _cc_labels(mask: np.ndarray, connectivity: Optional[int] = None) -> np.ndarray:
    if mask.ndim == 3:
        import cc3d

        return cc3d.connected_components(mask.astype(np.uint8), connectivity=connectivity or 26)
    from scipy.ndimage import label

    return label(mask, structure=np.ones((3, 3)))[0]


def keep_large_components(mask: np.ndarray, min_ratio: float = 0.0, min_voxels: int = 0,
                          connectivity: Optional[int] = None) -> np.ndarray:
    mask = mask.astype(bool)
    if not mask.any():
        return mask
    labels = _cc_labels(mask, connectivity)
    if labels.max() == 0:
        return mask
    counts = np.bincount(labels.ravel())
    counts[0] = 0
    if min_ratio <= 0 and min_voxels <= 0:
        return labels == counts.argmax()
    thresh = max(min_ratio * counts.max(), float(min_voxels), 1.0)
    keep_ids = np.nonzero(counts >= thresh)[0]
    return np.isin(labels, keep_ids)


def largest_connected_component(mask: np.ndarray, connectivity: Optional[int] = None) -> np.ndarray:
    return keep_large_components(mask, connectivity=connectivity)


def keep_largest_per_class(labels: np.ndarray, num_classes: int, min_ratio: float = 0.0,
                           min_voxels: int = 0) -> np.ndarray:
    out = np.zeros_like(labels)
    for c in range(1, max(num_classes, 2)):
        comp = keep_large_components(labels == c, min_ratio=min_ratio, min_voxels=min_voxels)
        out[comp] = c
    return out


def morphological_closing(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    if radius <= 0:
        return mask.astype(bool)
    from scipy.ndimage import binary_closing, generate_binary_structure, iterate_structure

    st = generate_binary_structure(mask.ndim, 1)
    st = iterate_structure(st, radius)
    return binary_closing(mask.astype(bool), structure=st)


def closing_per_class(labels: np.ndarray, num_classes: int, radius: int = 1) -> np.ndarray:
    out = labels.copy()
    bg = labels == 0
    for c in range(1, max(num_classes, 2)):
        comp = morphological_closing(labels == c, radius)
        out[comp & bg] = c
    return out


def postprocess_volume(
    labels: np.ndarray,
    num_classes: int,
    lcc: bool = True,
    closing_radius: int = 1,
    lcc_min_ratio: float = 0.0,
    lcc_min_voxels: int = 0,
) -> np.ndarray:
    out = labels.copy()
    if lcc:
        out = keep_largest_per_class(out, num_classes, min_ratio=lcc_min_ratio, min_voxels=lcc_min_voxels)
    if closing_radius > 0:
        out = closing_per_class(out, num_classes, closing_radius)
    return out


class TemporalEMA:
    def __init__(self, alpha: float = 0.6):
        self.alpha = alpha
        self._state: Optional[np.ndarray] = None

    def reset(self) -> None:
        self._state = None

    def update(self, prob: np.ndarray) -> np.ndarray:
        if self._state is None or self._state.shape != prob.shape:
            self._state = prob.astype(np.float32)
        else:
            self._state = self.alpha * self._state + (1.0 - self.alpha) * prob.astype(np.float32)
        return self._state
