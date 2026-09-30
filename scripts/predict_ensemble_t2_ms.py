from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from common.io import load_config, read_nifti, write_nifti, write_predictions_json
from common.postprocess import postprocess_volume
from common.sliding_window import sliding_window_predict
from common.transforms import us_zscore
from task2.dataset import list_images
from task2.predict import load_student


def parse_geom(spec: str, cfg):
    if spec in ("native", "nat", ""):
        return None, list(cfg.data.patch_size), cfg.predict.sw_mode
    parts = spec.split(":")
    sp = [float(v) for v in parts[0].split(",")]
    roi = int(parts[1]) if len(parts) > 1 and parts[1] else int(cfg.data.patch_size[0])
    mode = parts[2] if len(parts) > 2 and parts[2] else cfg.predict.sw_mode
    return sp, [roi, roi, roi], mode


@torch.inference_mode()
def predict_volume_multiscale(models, image_path, cfg, device, geoms, flip_axes, flip_combos):
    arr, affine, native_sp = read_nifti(image_path)
    x0 = torch.from_numpy(us_zscore(arr))[None, None].float().to(device)
    native_shape = tuple(x0.shape[2:])
    prob, n = None, 0
    for tgt_sp, roi, mode in geoms:
        x = x0
        if tgt_sp is not None:
            new_shape = [max(int(round(s * float(o) / float(t))), 16)
                         for s, o, t in zip(native_shape, native_sp, tgt_sp)]
            x = torch.nn.functional.interpolate(x0, size=new_shape, mode="trilinear",
                                                align_corners=False)
        acc = None
        for m in models:
            p = sliding_window_predict(m, x, roi, overlap=cfg.predict.sw_overlap, mode=mode,
                                       amp=cfg.amp,
                                       sw_batch_size=cfg.predict.get("sw_batch_size", 1),
                                       flip_axes=flip_axes, flip_combos=flip_combos)
            acc = p if acc is None else acc + p
        acc /= float(len(models))
        if tgt_sp is not None:
            acc = torch.nn.functional.interpolate(acc, size=native_shape, mode="trilinear",
                                                  align_corners=False)
        prob = acc if prob is None else prob + acc
        n += 1
        del acc, x
    prob /= float(n)
    cw = cfg.predict.get("class_weights", None)
    if cw:
        w = torch.tensor(list(cw), device=prob.device, dtype=prob.dtype)
        prob = prob * w.view(1, -1, *([1] * (prob.ndim - 2)))
    pred = prob.argmax(1)[0].cpu().numpy()
    pred = postprocess_volume(pred, cfg.num_classes, lcc=cfg.predict.lcc,
                              closing_radius=cfg.predict.closing_radius,
                              lcc_min_ratio=cfg.predict.get("lcc_min_ratio", 0.0))
    return pred, affine


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", required=True, help="repeat per ensemble member")
    ap.add_argument("--geom", action="append", default=[],
                    help="repeat per inference geometry; default: native + median-spacing roi160")
    ap.add_argument("--flip", action="store_true", help="8-way flip TTA")
    ap.add_argument("--split", default="val")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()

    cfg = load_config("task2", args.overrides)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out or cfg.paths.submission_dir) / "t2_tee"
    out_dir.mkdir(parents=True, exist_ok=True)

    specs = args.geom or ["native", "0.3727,0.5396,0.2322:160:constant"]
    geoms = [parse_geom(s, cfg) for s in specs]
    models = [load_student(cfg, c, device).eval() for c in args.ckpt]
    flip_axes = [0, 1, 2] if args.flip else []
    print(f"[ens-t2-ms] {len(models)} members x {len(geoms)} geometries {specs}; flip={args.flip}")
    items = list_images(cfg.paths.t2_root, args.split)
    if args.limit:
        items = items[: args.limit]

    cases, times = [], []
    for item in items:
        case_id = Path(item["image"]).name.replace(".nii.gz", "").replace(".nii", "")
        if case_id.endswith("-US"):
            case_id = case_id[:-3]
        t0 = time.time()
        pred, affine = predict_volume_multiscale(models, item["image"], cfg, device, geoms,
                                                 flip_axes, args.flip)
        rel = f"{case_id}.nii.gz"
        write_nifti(out_dir / rel, pred, affine)
        dt = time.time() - t0
        times.append(dt)
        cases.append({"case_id": case_id, "segmentation": rel})
        print(f"{case_id}: {dt:.2f}s", flush=True)
    write_predictions_json(out_dir, cases, task_num=2)
    if times:
        print(f"p50={np.percentile(times,50):.2f}s p95={np.percentile(times,95):.2f}s "
              f"total={np.sum(times):.0f}s")


if __name__ == "__main__":
    main()
