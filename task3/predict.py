from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from common.io import load_checkpoint, load_config, read_png, write_png, write_predictions_json
from common.log import report_load
from common.postprocess import TemporalEMA, largest_connected_component
from task3.dataset import list_unlabeled_videos
from task3.model import build_model, export_student


def probability_to_mask(prob: np.ndarray, native_hw, threshold: float,
                        resize_probability: bool = False):
    oh, ow = (int(native_hw[0]), int(native_hw[1]))
    if resize_probability:
        prob = cv2.resize(prob, (ow, oh), interpolation=cv2.INTER_LINEAR)
        return (prob >= threshold).astype(np.uint8)
    mask = (prob >= threshold).astype(np.uint8)
    if mask.shape != (oh, ow):
        mask = cv2.resize(mask, (ow, oh), interpolation=cv2.INTER_NEAREST)
    return mask


@torch.inference_mode()
def predict_frame(model, img_path, cfg, device, ema: TemporalEMA, orig_size, prior=None, prior_size=None):
    H, W = cfg.data.img_size
    img = read_png(img_path, gray=False)
    oh, ow = img.shape[:2]
    x = cv2.resize(img, (W, H)).transpose(2, 0, 1)[None]
    x = torch.from_numpy(x.copy()).float().to(device) / 255.0
    flips = cfg.predict.get("tta_flips", None) or [[]]
    _ps = []
    for fdims in flips:
        fdims = list(fdims)
        xf = torch.flip(x, dims=fdims) if fdims else x
        p = torch.softmax(model(xf), 1)
        if fdims:
            p = torch.flip(p, dims=fdims)
        _ps.append(p)
    prob = (torch.stack(_ps).mean(0) if len(_ps) > 1 else _ps[0])[0, 1].cpu().numpy()
    prob = ema.update(prob)
    mask = probability_to_mask(
        prob, (oh, ow), float(cfg.predict.get("threshold", 0.5)),
        bool(cfg.predict.get("resize_probability", False)))
    if prior is not None:
        from common.shape_prior import refine
        mask = refine(mask.astype(np.int64), cfg.num_classes, prior, device, size=prior_size).astype(np.uint8)
    else:
        min_area = int(cfg.predict.get("min_area", 0))
        if min_area and int(mask.sum()) < min_area:
            mask = np.zeros_like(mask)
        else:
            min_comp = int(cfg.predict.get("min_component", 0))
            if min_comp:
                import cc3d
                lbl = cc3d.connected_components(mask, connectivity=8)
                if lbl.max() > 0:
                    sizes = np.bincount(lbl.ravel()); sizes[0] = 0
                    mask = np.isin(lbl, np.where(sizes >= min_comp)[0]).astype(np.uint8)
    return mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--split", default="val")
    ap.add_argument("--out", default=None)
    ap.add_argument("--source", choices=["labeled", "pool"], default="labeled",
                    help="labeled = reference_data val frames; pool = images/REC_* clips")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()

    cfg = load_config("task3", args.overrides)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_root = Path(args.out or cfg.paths.submission_dir) / "t3_vid"
    out_root.mkdir(parents=True, exist_ok=True)

    model = build_model(cfg).to(device)
    if args.ckpt:
        state = load_checkpoint(args.ckpt, map_location=device)
        report_load(model, state["model"], name="task3.student", require=0.95)
    model = export_student(model)
    from common.shape_prior import load_for_predict
    prior, prior_size = load_for_predict(cfg, 2, device)

    if args.source == "pool":
        images_root = cfg.paths.images_dir
    else:
        images_root = Path(cfg.paths.t3_root) / args.split / "images"
        if not images_root.exists():
            images_root = Path(cfg.paths.t3_root) / args.split
    videos = list_unlabeled_videos(images_root)
    videos = {v: [f for f in fr if "_label" not in Path(f).name.lower()] for v, fr in videos.items()}

    from common.io import parse_frame_name
    max_gap = int(cfg.predict.get("ema_max_gap", 2))
    cases, times, resets = [], [], 0
    for vid, frames in videos.items():
        ema = TemporalEMA(alpha=cfg.predict.temporal_ema)
        prev_idx = None
        for fpath in frames:
            parsed = parse_frame_name(Path(fpath).name)
            idx = parsed[1] if parsed else None
            if prev_idx is None or idx is None or (idx - prev_idx) > max_gap:
                ema.reset()
                resets += 1
            prev_idx = idx
            t0 = time.time()
            mask = predict_frame(model, fpath, cfg, device, ema, None, prior, prior_size)
            stem = Path(fpath).name[:-4]
            rel = f"{vid}/{stem}_label_bin.png"
            write_png(out_root / rel, mask * 255)
            times.append(time.time() - t0)
            cases.append({"case_id": stem, "segmentation": rel})
    write_predictions_json(out_root, cases, task_num=3)
    print(f"[predict] {len(cases)} frames, {resets} EMA resets (=frames means EMA off for sparse keyframes)")
    if times:
        print(f"per-frame p50={np.percentile(times,50)*1000:.1f}ms p95={np.percentile(times,95)*1000:.1f}ms")


if __name__ == "__main__":
    main()
