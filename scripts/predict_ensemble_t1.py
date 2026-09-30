from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from common.io import load_config, read_nifti, write_nifti, write_predictions_json
from common.postprocess import postprocess_volume
from common.spacing_guard import needs_resample, resampled_shape
from common.sliding_window import sliding_window_predict
from common.transforms import ct_window
from task1.dataset import list_images
from task1.predict import load_student


def _fusion_options(cfg):
    options = cfg.predict.get("canonical_fusion", {})
    enabled = bool(options.get("enabled", False))
    minimum = float(options.get("min_slice_spacing", 0.55))
    target = tuple(float(value) for value in options.get("target_spacing", (0.357, 0.357, 0.5)))
    roi = tuple(int(value) for value in options.get("roi", cfg.data.patch_size))
    threshold = float(options.get("threshold", 0.525))
    if not np.isfinite(minimum) or minimum <= 0:
        raise ValueError("predict.canonical_fusion.min_slice_spacing must be finite and positive")
    if len(target) != 3 or not all(np.isfinite(target)) or min(target) <= 0:
        raise ValueError("predict.canonical_fusion.target_spacing must contain three positive finite values")
    if len(roi) != 3 or min(roi) <= 0:
        raise ValueError("predict.canonical_fusion.roi must contain three positive integers")
    if not np.isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError("predict.canonical_fusion.threshold must be between zero and one")
    return enabled, minimum, target, roi, threshold


def _valid_spacing(spacing):
    values = tuple(float(value) for value in spacing[:3])
    return len(values) == 3 and all(np.isfinite(values)) and min(values) > 0


def _thick_slice(spacing, minimum):
    return _valid_spacing(spacing) and float(spacing[2]) > float(minimum) + 1e-6


def _member_probability(model, x, cfg, roi):
    return sliding_window_predict(
        model,
        x,
        roi,
        overlap=cfg.predict.sw_overlap,
        mode=cfg.predict.sw_mode,
        amp=cfg.amp,
        sw_batch_size=cfg.predict.get("sw_batch_size", 1),
        flip_axes=cfg.predict.get("tta_flip_axes", []),
        flip_combos=cfg.predict.get("tta_full_combos", False),
    )


def _postprocess(pred, cfg):
    return postprocess_volume(
        pred,
        cfg.num_classes,
        lcc=cfg.predict.lcc,
        closing_radius=cfg.predict.closing_radius,
        lcc_min_ratio=cfg.predict.get("lcc_min_ratio", 0.0),
    )


@torch.inference_mode()
def predict_volume_ensemble(models, image_path, cfg, device):
    fusion_enabled, min_slice, target_spacing, canonical_roi, threshold = _fusion_options(cfg)
    arr = affine = spacing = None
    guard_options = cfg.predict.get("spacing_guard", {})
    guard_enabled = bool(guard_options.get("enabled", False))
    if guard_enabled or fusion_enabled:
        arr, affine, spacing = read_nifti(image_path)
    out_of_range = (guard_enabled or fusion_enabled) and needs_resample(
        spacing,
        tuple(guard_options.get("in_plane_ok", (0.28, 0.52))),
        tuple(guard_options.get("slice_ok", (0.40, 0.70))),
    )
    if guard_enabled and out_of_range:
        from common.spacing_guard import predict_volume_ensemble_guarded

        return predict_volume_ensemble_guarded(models, image_path, cfg, device)
    if arr is None:
        arr, affine, spacing = read_nifti(image_path)
    vol = ct_window(arr, *cfg.data.hu_clip)
    x = torch.from_numpy(vol)[None, None].float().to(device)
    use_fusion = fusion_enabled and not out_of_range and _thick_slice(spacing, min_slice)
    if use_fusion:
        work_shape = resampled_shape(arr.shape, spacing, target_spacing)
        canonical_x = F.interpolate(x, size=work_shape, mode="trilinear", align_corners=False)
        foreground = None
        for model in models:
            native_prob = _member_probability(model, x, cfg, cfg.data.patch_size)[:, 1:2]
            canonical_prob = _member_probability(model, canonical_x, cfg, canonical_roi)[:, 1:2]
            canonical_prob = F.interpolate(
                canonical_prob, size=arr.shape, mode="trilinear", align_corners=False
            )
            member_sum = native_prob + canonical_prob
            foreground = member_sum if foreground is None else foreground + member_sum
        foreground = foreground / (2 * len(models))
        pred = (foreground[0, 0] >= threshold).cpu().numpy().astype(np.uint8)
        return _postprocess(pred, cfg), affine
    prob = None
    for m in models:
        p = _member_probability(m, x, cfg, cfg.data.patch_size)
        prob = p if prob is None else prob + p
    pred = prob.argmax(1)[0].cpu().numpy()
    return _postprocess(pred, cfg), affine


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", required=True, help="repeat for each ensemble member")
    ap.add_argument("--split", default="val")
    ap.add_argument("--out", default=None)
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()

    cfg = load_config("task1", args.overrides)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out or cfg.paths.submission_dir) / "t1_ct"
    out_dir.mkdir(parents=True, exist_ok=True)

    models = [load_student(cfg, c, device).eval() for c in args.ckpt]
    print(f"[ensemble] {len(models)} members; tta_flip_axes={list(cfg.predict.get('tta_flip_axes', []))} "
          f"full_combos={cfg.predict.get('tta_full_combos', False)}")
    items = list_images(cfg.paths.t1_root, args.split)

    cases, times = [], []
    for item in items:
        case_id = Path(item["image"]).name.replace(".nii.gz", "").replace(".nii", "")
        t0 = time.time()
        pred, affine = predict_volume_ensemble(models, item["image"], cfg, device)
        rel = f"{case_id}.nii.gz"
        write_nifti(out_dir / rel, pred, affine)
        dt = time.time() - t0
        times.append(dt)
        cases.append({"case_id": case_id, "segmentation": rel})
        print(f"{case_id}: {dt:.2f}s")
    write_predictions_json(out_dir, cases, task_num=1)
    if times:
        print(f"p50={np.percentile(times,50):.2f}s p95={np.percentile(times,95):.2f}s (budget 10s)")


if __name__ == "__main__":
    main()
