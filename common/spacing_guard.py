from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np

IN_PLANE_OK: Tuple[float, float] = (0.28, 0.52)
SLICE_OK: Tuple[float, float] = (0.40, 0.70)
TARGET_SPACING: Tuple[float, float, float] = (0.357, 0.357, 0.5)

GT_VOLUME_MM3: Tuple[float, float] = (1854.0, 4639.0)

MAX_VOXELS: int = 12_000_000
ROI_MM: Tuple[float, float, float] = (80.0, 80.0, 72.0)


def needs_resample(spacing: Sequence[float],
                   in_plane_ok: Tuple[float, float] = IN_PLANE_OK,
                   slice_ok: Tuple[float, float] = SLICE_OK) -> bool:
    sp = [float(s) for s in spacing[:3]]
    if not all(np.isfinite(sp)) or min(sp) <= 0:
        return False
    lo, hi = in_plane_ok
    if not (lo <= sp[0] <= hi) or not (lo <= sp[1] <= hi):
        return True
    lo, hi = slice_ok
    return not (lo <= sp[2] <= hi)


def resample_factor(shape: Sequence[int], spacing: Sequence[float],
                    target: Sequence[float] = TARGET_SPACING,
                    max_voxels: int = MAX_VOXELS) -> np.ndarray:
    factor = np.asarray(spacing[:3], dtype=float) / np.asarray(target, dtype=float)
    n = float(np.prod(np.asarray(shape, dtype=float) * factor))
    if n > max_voxels:
        factor = factor * (max_voxels / n) ** (1.0 / 3.0)
    return factor


def resampled_shape(shape: Sequence[int], spacing: Sequence[float],
                    target: Sequence[float] = TARGET_SPACING,
                    minimum: int = 16) -> Tuple[int, int, int]:
    return tuple(
        max(int(round(int(size) * float(source) / float(destination))), int(minimum))
        for size, source, destination in zip(shape[:3], spacing[:3], target[:3])
    )


def resample_volume(vol: np.ndarray, spacing: Sequence[float],
                    target: Sequence[float] = TARGET_SPACING,
                    max_voxels: int = MAX_VOXELS) -> np.ndarray:
    from scipy.ndimage import zoom

    factor = resample_factor(vol.shape, spacing, target, max_voxels)
    return zoom(vol.astype(np.float32), factor, order=1, prefilter=False)


def resample_mask_back(mask: np.ndarray, out_shape: Sequence[int]) -> np.ndarray:
    from scipy.ndimage import zoom

    out_shape = tuple(int(s) for s in out_shape)
    factor = np.asarray(out_shape, dtype=float) / np.asarray(mask.shape, dtype=float)
    out = np.zeros(out_shape, dtype=mask.dtype)
    for c in [c for c in np.unique(mask) if c != 0]:
        up = zoom((mask == c).astype(np.float32), factor, order=1, prefilter=False) > 0.5
        up = up[:out_shape[0], :out_shape[1], :out_shape[2]]
        pad = [(0, max(0, out_shape[i] - up.shape[i])) for i in range(3)]
        out[np.pad(up, pad)] = c
    return out


def central_crop_bounds(shape: Sequence[int], spacing: Sequence[float],
                        roi_mm: Sequence[float] = ROI_MM):
    shape = np.asarray(shape[:3], dtype=int)
    spacing = np.asarray(spacing[:3], dtype=float)
    size = np.minimum(shape, np.ceil(np.asarray(roi_mm, dtype=float) / spacing).astype(int))
    start = np.maximum((shape - size) // 2, 0)
    stop = start + size
    return tuple(slice(int(a), int(b)) for a, b in zip(start, stop))


def predict_volume_ensemble_guarded(models, image_path, cfg, device):
    import torch

    from common.io import read_nifti
    from common.postprocess import postprocess_volume
    from common.sliding_window import sliding_window_predict
    from common.transforms import ct_window

    arr, affine, spacing = read_nifti(image_path)
    options = cfg.predict.get("spacing_guard", {})
    in_plane_ok = tuple(options.get("in_plane_ok", IN_PLANE_OK))
    slice_ok = tuple(options.get("slice_ok", SLICE_OK))
    target = tuple(options.get("target_spacing", TARGET_SPACING))
    roi_mm = tuple(options.get("roi_mm", ROI_MM))
    max_voxels = int(options.get("max_voxels", MAX_VOXELS))
    fire = needs_resample(spacing, in_plane_ok, slice_ok)
    crop = central_crop_bounds(arr.shape, spacing, roi_mm) if fire else tuple(slice(None) for _ in range(3))
    native = arr[crop]
    work = resample_volume(native, spacing, target, max_voxels) if fire else native
    if fire:
        print(f"[t1.spacing_guard] {image_path}: spacing {tuple(round(float(s), 4) for s in spacing[:3])} "
              f"out of trained range -> cropped {tuple(arr.shape)} -> {tuple(native.shape)} "
              f"-> resampled {tuple(work.shape)}")

    vol = ct_window(work, *cfg.data.hu_clip)
    with torch.inference_mode():
        x = torch.from_numpy(vol)[None, None].float().to(device)
        prob = None
        for m in models:
            p = sliding_window_predict(m, x, cfg.data.patch_size, overlap=cfg.predict.sw_overlap,
                                       mode=cfg.predict.sw_mode, amp=cfg.amp,
                                       sw_batch_size=cfg.predict.get("sw_batch_size", 1),
                                       flip_axes=cfg.predict.get("tta_flip_axes", []),
                                       flip_combos=cfg.predict.get("tta_full_combos", False))
            prob = p if prob is None else prob + p
        pred = prob.argmax(1)[0].cpu().numpy()

    if fire:
        native_pred = resample_mask_back(pred.astype(np.uint8), native.shape)
        pred = np.zeros(arr.shape, dtype=np.uint8)
        pred[crop] = native_pred
    pred = postprocess_volume(pred, cfg.num_classes, lcc=cfg.predict.lcc,
                              closing_radius=cfg.predict.closing_radius,
                              lcc_min_ratio=cfg.predict.get("lcc_min_ratio", 0.0))
    return pred, affine


PLAUSIBLE_MM3: Tuple[float, float] = (1600.0, 5000.0)
AGREEMENT_MAX: float = 0.5
CATASTROPHIC_MM3: float = 15000.0


def volume_mm3(mask, spacing: Sequence[float]) -> float:
    return float((np.asarray(mask) > 0).sum() * float(np.prod(np.asarray(spacing[:3], dtype=float))))


def largest_component_centroid(mask):
    from scipy.ndimage import label as _cc

    mask = np.asarray(mask) > 0
    if not mask.any():
        return None
    lab, n = _cc(mask)
    if n == 0:
        return None
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    return np.argwhere(lab == int(sizes.argmax())).mean(0)


def centered_crop_bounds(center, shape: Sequence[int], spacing: Sequence[float],
                         roi_mm: Sequence[float] = ROI_MM):
    shape = np.asarray(shape[:3], dtype=int)
    size = np.minimum(shape, np.ceil(np.asarray(roi_mm, dtype=float)
                                     / np.asarray(spacing[:3], dtype=float)).astype(int))
    start = np.clip((np.asarray(center, dtype=float) - size / 2.0).astype(int), 0, shape - size)
    return tuple(slice(int(a), int(a + b)) for a, b in zip(start, size))


def _ensemble_mask(models, vol: np.ndarray, cfg, device, tta: bool = True) -> np.ndarray:
    import torch

    from common.sliding_window import sliding_window_predict
    from common.transforms import ct_window

    with torch.inference_mode():
        x = torch.from_numpy(ct_window(vol, *cfg.data.hu_clip))[None, None].float().to(device)
        prob = None
        for m in models:
            p = sliding_window_predict(m, x, cfg.data.patch_size, overlap=cfg.predict.sw_overlap,
                                       mode=cfg.predict.sw_mode, amp=cfg.amp,
                                       sw_batch_size=cfg.predict.get("sw_batch_size", 1),
                                       flip_axes=cfg.predict.get("tta_flip_axes", []) if tta else [],
                                       flip_combos=cfg.predict.get("tta_full_combos", False) if tta else False)
            prob = p if prob is None else prob + p
        return prob.argmax(1)[0].cpu().numpy().astype(np.uint8)


def predict_volume_ensemble_rescued(models, image_path, cfg, device):
    from common.io import read_nifti
    from common.postprocess import postprocess_volume
    from scripts.predict_ensemble_t1 import predict_volume_ensemble

    pred, affine = predict_volume_ensemble(models, image_path, cfg, device)

    options = cfg.predict.get("spacing_rescue", {})
    if not bool(options.get("enabled", False)):
        return pred, affine
    in_plane_ok = tuple(options.get("in_plane_ok", IN_PLANE_OK))
    slice_ok = tuple(options.get("slice_ok", SLICE_OK))
    target = tuple(options.get("target_spacing", TARGET_SPACING))
    roi_mm = tuple(options.get("roi_mm", ROI_MM))
    max_voxels = int(options.get("max_voxels", MAX_VOXELS))
    lo, hi = tuple(options.get("plausible_mm3", PLAUSIBLE_MM3))

    arr, _, spacing = read_nifti(image_path)
    if not needs_resample(spacing, in_plane_ok, slice_ok):
        return pred, affine
    v0 = volume_mm3(pred, spacing)
    if lo <= v0 <= hi:
        print(f"[t1.rescue] {image_path}: out-of-band spacing but prediction is plausible "
              f"({v0:.0f} mm^3) -> keeping unguarded result")
        return pred, affine

    coarse = _ensemble_mask(models, resample_volume(arr, spacing, target, max_voxels), cfg, device, tta=False)
    center = largest_component_centroid(coarse)
    if center is None:
        print(f"[t1.rescue] {image_path}: coarse pass found no foreground -> keeping unguarded result")
        return pred, affine
    center = np.asarray(center, dtype=float) * (np.asarray(arr.shape[:3], dtype=float)
                                                / np.asarray(coarse.shape[:3], dtype=float))
    rescued = None
    for _ in range(2):
        crop = centered_crop_bounds(center, arr.shape, spacing, roi_mm)
        native = arr[crop]
        sub = _ensemble_mask(models, resample_volume(native, spacing, target, max_voxels), cfg, device)
        full = np.zeros(arr.shape, dtype=np.uint8)
        full[crop] = resample_mask_back(sub, native.shape)
        rescued = full
        nxt = largest_component_centroid(full)
        if nxt is None:
            break
        center = np.asarray(nxt, dtype=float)
    if rescued is None or not rescued.any():
        return pred, affine
    rescued = postprocess_volume(rescued, cfg.num_classes, lcc=cfg.predict.lcc,
                                 closing_radius=cfg.predict.closing_radius,
                                 lcc_min_ratio=cfg.predict.get("lcc_min_ratio", 0.0))
    v1 = volume_mm3(rescued, spacing)
    catastrophic = float(options.get("catastrophic_mm3", CATASTROPHIC_MM3))
    salvage = v0 > catastrophic and v1 < 0.5 * v0
    if not (lo <= v1 <= hi) and not salvage:
        print(f"[t1.rescue] {image_path}: rescue implausible ({v1:.0f} mm^3) -> keeping unguarded result")
        return pred, affine
    if salvage and not (lo <= v1 <= hi):
        print(f"[t1.rescue] {image_path}: unguarded mask is catastrophic ({v0:.0f} mm^3); accepting the "
              f"smaller rescue ({v1:.0f} mm^3) even though it is outside the anatomical band")
    a, b = np.asarray(pred) > 0, np.asarray(rescued) > 0
    agree = (2.0 * (a & b).sum() / (a.sum() + b.sum())) if (a.any() and b.any()) else 0.0
    if agree >= float(options.get("agreement_max", AGREEMENT_MAX)):
        print(f"[t1.rescue] {image_path}: rescue agrees with unguarded (Dice {agree:.3f}) -> keeping "
              f"unguarded result (an unusually sized but correctly placed valve)")
        return pred, affine
    print(f"[t1.rescue] {image_path}: RESCUED misplaced prediction {v0:.0f} -> {v1:.0f} mm^3 "
          f"(Dice vs unguarded {agree:.3f})")
    return rescued, affine
