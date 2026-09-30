from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Iterable, Optional, Sequence

import numpy as np

HD_REF_DEFAULT = 20.0
ASD_REF_DEFAULT = 3.0


def q_dsc(dice: float) -> float:
    return float(np.clip(dice, 0.0, 1.0))


def q_hd(hd95: float, hd_ref: float = HD_REF_DEFAULT) -> float:
    return float(1.0 / (1.0 + hd95 / hd_ref))


def q_asd(asd: float, asd_ref: float = ASD_REF_DEFAULT) -> float:
    return float(1.0 / (1.0 + asd / asd_ref))


def composite_score(
    dice: float,
    hd95: float,
    asd: float,
    hd_ref: float = HD_REF_DEFAULT,
    asd_ref: float = ASD_REF_DEFAULT,
) -> float:
    return (q_dsc(dice) + q_hd(hd95, hd_ref) + q_asd(asd, asd_ref)) / 3.0


def dice_coefficient(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    denom = pred.sum() + gt.sum()
    if denom == 0:
        return 1.0
    return float(2.0 * inter / denom)


def _surface_distances(pred: np.ndarray, gt: np.ndarray, spacing: Sequence[float]) -> np.ndarray:
    from scipy.ndimage import binary_erosion, distance_transform_edt

    pred = pred.astype(bool)
    gt = gt.astype(bool)
    if not pred.any() or not gt.any():
        return np.array([])

    conn = np.ones((3,) * pred.ndim, dtype=bool)
    pred_border = pred ^ binary_erosion(pred, structure=conn, iterations=1)
    gt_border = gt ^ binary_erosion(gt, structure=conn, iterations=1)
    if not pred_border.any() or not gt_border.any():
        return np.array([])

    dt_gt = distance_transform_edt(~gt_border, sampling=spacing)
    dt_pred = distance_transform_edt(~pred_border, sampling=spacing)
    d_pred_to_gt = dt_gt[pred_border]
    d_gt_to_pred = dt_pred[gt_border]
    return np.concatenate([d_pred_to_gt, d_gt_to_pred])


def _surface_metrics(pred: np.ndarray, gt: np.ndarray, spacing: Sequence[float]):
    try:
        import torch
        from monai.metrics import compute_average_surface_distance, compute_hausdorff_distance

        yp = torch.from_numpy(np.ascontiguousarray(pred.astype(np.uint8)))[None, None]
        yg = torch.from_numpy(np.ascontiguousarray(gt.astype(np.uint8)))[None, None]
        sp = tuple(float(s) for s in spacing)
        hd = compute_hausdorff_distance(yp, yg, include_background=True, percentile=None, spacing=sp)
        ad = compute_average_surface_distance(yp, yg, include_background=True, symmetric=True, spacing=sp)
        return float(hd.reshape(-1)[0]), float(ad.reshape(-1)[0])
    except Exception:
        d = _surface_distances(pred, gt, spacing)
        if d.size == 0:
            return float("nan"), float("nan")
        return float(d.max()), float(d.mean())


def _diag_mm(shape: Sequence[int], spacing: Sequence[float]) -> float:
    return float(np.sqrt(sum((float(s) * float(sp)) ** 2 for s, sp in zip(shape, spacing))))


def hd95(pred: np.ndarray, gt: np.ndarray, spacing: Sequence[float], fallback: float = HD_REF_DEFAULT) -> float:
    p, g = pred.astype(bool), gt.astype(bool)
    if not p.any() and not g.any():
        return 0.0
    if p.any() != g.any():
        return _diag_mm(pred.shape, spacing)
    h, _ = _surface_metrics(p, g, spacing)
    return float(h) if np.isfinite(h) else _diag_mm(pred.shape, spacing)


def asd(pred: np.ndarray, gt: np.ndarray, spacing: Sequence[float], fallback: float = ASD_REF_DEFAULT) -> float:
    p, g = pred.astype(bool), gt.astype(bool)
    if not p.any() and not g.any():
        return 0.0
    if p.any() != g.any():
        return _diag_mm(pred.shape, spacing)
    _, a = _surface_metrics(p, g, spacing)
    return float(a) if np.isfinite(a) else _diag_mm(pred.shape, spacing)


@dataclass
class CaseMetrics:
    dice: float
    hd95: float
    asd: float
    score: float

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)


def evaluate_case(
    pred: np.ndarray,
    gt: np.ndarray,
    num_classes: int,
    spacing: Optional[Sequence[float]] = None,
    hd_ref: float = HD_REF_DEFAULT,
    asd_ref: float = ASD_REF_DEFAULT,
) -> CaseMetrics:
    if spacing is None:
        spacing = (1.0,) * pred.ndim
    fg = range(1, max(num_classes, 2))
    dices, hds, asds = [], [], []
    for c in fg:
        pc = pred == c
        gc = gt == c
        dices.append(dice_coefficient(pc, gc))
        hds.append(hd95(pc, gc, spacing, fallback=hd_ref))
        asds.append(asd(pc, gc, spacing, fallback=asd_ref))
    d, h, a = float(np.mean(dices)), float(np.mean(hds)), float(np.mean(asds))
    return CaseMetrics(d, h, a, composite_score(d, h, a, hd_ref, asd_ref))


def aggregate(cases: Iterable[CaseMetrics]) -> Dict[str, float]:
    cases = list(cases)
    if not cases:
        return {"dice": 0.0, "hd95": 0.0, "asd": 0.0, "score": 0.0, "n": 0}
    return {
        "dice": float(np.mean([c.dice for c in cases])),
        "hd95": float(np.mean([c.hd95 for c in cases])),
        "asd": float(np.mean([c.asd for c in cases])),
        "score": float(np.mean([c.score for c in cases])),
        "n": len(cases),
    }
