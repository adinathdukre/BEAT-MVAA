from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from common.io import load_checkpoint, load_config, read_png, write_png, write_predictions_json
from common.log import report_load
from common.postprocess import largest_connected_component
from task3.dataset import list_unlabeled_videos
from task3.model import build_model, export_student
from task3.predict import probability_to_mask


def load_one(cfg, ckpt, device):
    m = build_model(cfg).to(device)
    state = load_checkpoint(ckpt, map_location=device)
    report_load(m, state["model"], name="task3.student", require=0.95)
    return export_student(m).eval()


@torch.inference_mode()
def predict_member_probabilities(models, img_path, cfg, device, hflip=False):
    H, W = cfg.data.img_size
    img = read_png(img_path, gray=False)
    x = torch.from_numpy(cv2.resize(img, (W, H)).transpose(2, 0, 1)[None].copy()).float().to(device) / 255.0
    probabilities = []
    for model in models:
        probability = torch.softmax(model(x), 1)[0, 1]
        if hflip:
            flipped = torch.softmax(model(torch.flip(x, dims=[3])), 1)[0, 1]
            probability = (probability, torch.flip(flipped, dims=[1]))
        probabilities.append(probability)
    return probabilities, img.shape[:2], 2 if hflip else 1


def blend_probabilities(probabilities, weights=None, view_count=1):
    weights = weights if weights is not None else [1.0] * len(probabilities)
    if len(weights) != len(probabilities) or not probabilities:
        raise ValueError("probabilities and weights must have equal nonzero length")
    total, weight_sum = None, 0.0
    for probability, weight in zip(probabilities, weights):
        if view_count == 1:
            weighted = weight * probability
        else:
            weighted = weight * probability[0]
            weighted = weighted + weight * probability[1]
        total = weighted if total is None else total + weighted
        weight_sum += weight
    return total.cpu().numpy() / (weight_sum * view_count)


def finalize_probability(probability, native_hw, cfg, threshold=None):
    threshold = (
        float(cfg.predict.get("threshold", 0.5))
        if threshold is None
        else float(threshold)
    )
    mask = probability_to_mask(
        probability,
        native_hw,
        threshold,
        bool(cfg.predict.get("resize_probability", False)),
    )
    if cfg.predict.get("lcc", False):
        mask = largest_connected_component(mask).astype(np.uint8)
    return mask


def layered_mask(incumbent, primary, recovery):
    if primary.any():
        return primary, "primary"
    if incumbent.any():
        return incumbent, "incumbent"
    if recovery.any():
        return recovery, "recovery"
    return recovery, "blank"


@torch.inference_mode()
def predict_frame(models, img_path, cfg, device, hflip, weights=None):
    probabilities, native_hw, view_count = predict_member_probabilities(
        models, img_path, cfg, device, hflip
    )
    probability = blend_probabilities(probabilities, weights, view_count)
    return finalize_probability(probability, native_hw, cfg)


@torch.inference_mode()
def predict_layered_frame(s42, s44, guard_e37, img_path, cfg, device, return_branch=False):
    thr = float(cfg.predict.get("layered_threshold", 0.45))
    thr_rec = float(cfg.predict.get("layered_recovery_threshold", 0.10))
    probabilities, native_hw, view_count = predict_member_probabilities(
        [s42, s44, guard_e37], img_path, cfg, device, False
    )
    p42, p44, p37 = probabilities
    primary = finalize_probability(
        blend_probabilities([p37, p44], view_count=view_count),
        native_hw,
        cfg,
        thr,
    )
    if primary.any():
        weight = float(cfg.predict.get("primary_render_weight", 0.5))
        if abs(weight - 0.5) > 1e-9:
            weighted = finalize_probability(
                blend_probabilities([p37, p44], weights=[weight, 1.0 - weight], view_count=view_count),
                native_hw,
                cfg,
                thr,
            )
            if weighted.any():
                primary = weighted
        result = (primary, "primary")
    else:
        incumbent = finalize_probability(
            blend_probabilities([p42, p44], view_count=view_count),
            native_hw,
            cfg,
            thr,
        )
        if incumbent.any():
            result = (incumbent, "incumbent")
        else:
            recovery = finalize_probability(
                blend_probabilities([p37], view_count=view_count),
                native_hw,
                cfg,
                thr_rec,
            )
            result = layered_mask(incumbent, primary, recovery)

    if bool(cfg.predict.get("tta_render", False)) and result[0].any():
        spec = {"primary": ([2, 1], thr), "incumbent": ([0, 1], thr), "recovery": ([2], thr_rec)}.get(result[1])
        if spec is not None:
            members, branch_thr = spec
            roster = [s42, s44, guard_e37]
            tta_probs, _, tta_views = predict_member_probabilities(
                [roster[i] for i in members], img_path, cfg, device, True
            )
            rendered = finalize_probability(
                blend_probabilities(tta_probs, view_count=tta_views),
                native_hw,
                cfg,
                branch_thr,
            )
            if rendered.any():
                result = (rendered, result[1])
    return result if return_branch else result[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", required=True, help="repeat per ensemble member")
    ap.add_argument("--hflip", action="store_true")
    ap.add_argument("--split", default="val")
    ap.add_argument("--out", default=None)
    ap.add_argument("--source", choices=["labeled", "pool"], default="labeled")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()

    cfg = load_config("task3", args.overrides)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_root = Path(args.out or cfg.paths.submission_dir) / "t3_vid"
    out_root.mkdir(parents=True, exist_ok=True)

    models = [load_one(cfg, c, device) for c in args.ckpt]
    print(f"[ens-t3] {len(models)} members; hflip={args.hflip}")
    if args.source == "pool":
        images_root = cfg.paths.images_dir
    else:
        images_root = Path(cfg.paths.t3_root) / args.split / "images"
        if not images_root.exists():
            images_root = Path(cfg.paths.t3_root) / args.split
    videos = list_unlabeled_videos(images_root)
    videos = {v: [f for f in fr if "_label" not in Path(f).name.lower()] for v, fr in videos.items()}

    cases, times = [], []
    for vid, frames in videos.items():
        for fpath in frames:
            t0 = time.time()
            mask = predict_frame(models, fpath, cfg, device, args.hflip)
            stem = Path(fpath).name[:-4]
            rel = f"{vid}/{stem}_label_bin.png"
            write_png(out_root / rel, mask * 255)
            times.append(time.time() - t0)
            cases.append({"case_id": stem, "segmentation": rel})
    write_predictions_json(out_root, cases, task_num=3)
    print(f"[ens-t3] {len(cases)} frames; per-frame p95={np.percentile(times,95)*1000:.0f}ms")


if __name__ == "__main__":
    main()
