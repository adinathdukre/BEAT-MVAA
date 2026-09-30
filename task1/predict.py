from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from common.io import load_checkpoint, load_config, read_nifti, write_nifti, write_predictions_json
from common.postprocess import postprocess_volume
from common.registry import build_student3d, export_student, has_aux_heads
from common.sliding_window import sliding_window_predict
from common.log import report_load
from common.transforms import ct_window
from task1.dataset import list_images


@torch.inference_mode()
def predict_volume(model, image_path, cfg, device, prior=None, prior_size=None):
    arr, affine, _ = read_nifti(image_path)
    vol = ct_window(arr, *cfg.data.hu_clip)
    x = torch.from_numpy(vol)[None, None].float().to(device)
    prob = sliding_window_predict(model, x, cfg.data.patch_size, overlap=cfg.predict.sw_overlap,
                                  mode=cfg.predict.sw_mode, amp=cfg.amp,
                                  sw_batch_size=cfg.predict.get("sw_batch_size", 1),
                                  flip_axes=cfg.predict.get("tta_flip_axes", []))
    pred = prob.argmax(1)[0].cpu().numpy()
    if prior is not None:
        from common.postprocess import closing_per_class
        from common.shape_prior import refine
        pred = refine(pred, cfg.num_classes, prior, device, size=prior_size)
        if cfg.predict.closing_radius > 0:
            pred = closing_per_class(pred, cfg.num_classes, cfg.predict.closing_radius)
    else:
        pred = postprocess_volume(pred, cfg.num_classes, lcc=cfg.predict.lcc,
                                  closing_radius=cfg.predict.closing_radius,
                                  lcc_min_ratio=cfg.predict.get("lcc_min_ratio", 0.0))
    return pred, affine


def load_student(cfg, ckpt, device):
    model = build_student3d(cfg, load_pretrained=not ckpt).to(device)
    if ckpt:
        state = load_checkpoint(ckpt, map_location=device)
        report_load(model, state["model"], name="task1.student", require=0.95)
    model = export_student(model)
    assert not has_aux_heads(model), "deployed student must have no aux heads"
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--split", default="val")
    ap.add_argument("--out", default=None)
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()

    cfg = load_config("task1", args.overrides)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out or cfg.paths.submission_dir) / "t1_ct"
    out_dir.mkdir(parents=True, exist_ok=True)

    model = load_student(cfg, args.ckpt, device)
    from common.shape_prior import load_for_predict
    prior, prior_size = load_for_predict(cfg, 3, device)
    items = list_images(cfg.paths.t1_root, args.split)

    cases, times = [], []
    for item in items:
        case_id = Path(item["image"]).name.replace(".nii.gz", "").replace(".nii", "")
        t0 = time.time()
        pred, affine = predict_volume(model, item["image"], cfg, device, prior, prior_size)
        rel = f"{case_id}.nii.gz"
        write_nifti(out_dir / rel, pred, affine)
        dt = time.time() - t0
        times.append(dt)
        cases.append({"case_id": case_id, "segmentation": rel})
        print(f"{case_id}: {dt:.2f}s")
    write_predictions_json(out_dir, cases, task_num=1)
    if times:
        import numpy as np
        print(f"p50={np.percentile(times,50):.2f}s p95={np.percentile(times,95):.2f}s (budget 10s)")


if __name__ == "__main__":
    main()
