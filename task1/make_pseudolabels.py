from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from common.io import load_checkpoint, load_config, read_nifti
from common.log import report_load
from common.postprocess import postprocess_volume
from common.registry import build_teacher3d
from common.sliding_window import sliding_window_predict
from common.transforms import ct_window
from task1.dataset import list_unlabeled


def _safe_affine(affine, spacing=None):
    a = np.asarray(affine, dtype=np.float64)
    if a.shape == (4, 4) and np.isfinite(a).all() and abs(np.linalg.det(a[:3, :3])) > 1e-8:
        return a
    a = np.eye(4)
    if spacing is not None and len(spacing) >= 3:
        a[0, 0], a[1, 1], a[2, 2] = float(spacing[0]), float(spacing[1]), float(spacing[2])
    return a


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", required=True,
                    help="teacher checkpoint; repeat for a softmax-averaged ENSEMBLE (e.g. 3 CV folds)")
    ap.add_argument("--unlabeled_dir", default=None)
    ap.add_argument("--out_dir", default=None, help="override cfg.pseudolabel.out_dir (write a NEW dir; "
                    "do not clobber the existing pseudo-labels until the new set is validated)")
    ap.add_argument("--overwrite", action="store_true",
                    help="recompute pseudo-labels even if the output file already exists (default: skip "
                         "existing -> idempotent resume after a timeout/crash)")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()

    cfg = load_config("task1", args.overrides)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir or cfg.pseudolabel.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    models = []
    for i, ck in enumerate(args.ckpt):
        m = build_teacher3d(cfg, load_pretrained=False).to(device)
        state = load_checkpoint(ck, map_location=device)
        report_load(m, state["model"], name=f"task1.teacher[{i}]", require=0.95)
        m.eval()
        models.append(m)
    print(f"[pseudo] ensemble of {len(models)} teacher(s) -> {out_dir}")

    unl = list_unlabeled(args.unlabeled_dir or cfg.pseudolabel.unlabeled_dir)
    tau = cfg.pseudolabel.conf_threshold
    roi = list(cfg.teacher.get("patch_size", None) or cfg.data.patch_size)
    from common.uncertainty import reliable_keep_np

    suffix = cfg.get("label_suffix", "-seg")
    btau = cfg.pseudolabel.get("boundary_conf_threshold", tau)
    band = int(cfg.pseudolabel.get("boundary_band", 0))
    ignore_index = cfg.loss.get("ignore_index", None)
    min_fg = int(cfg.pseudolabel.get("min_fg_voxels", 0))

    kept = skipped_empty = skipped_exist = 0
    for item in unl:
        stem = Path(item["image"]).name.replace(".nii.gz", "").replace(".nii", "")
        out_path = out_dir / f"{stem}{suffix}.nii.gz"
        if out_path.exists() and not args.overwrite:
            skipped_exist += 1
            continue
        arr, affine, _ = read_nifti(item["image"])
        vol = ct_window(arr, *cfg.data.hu_clip)
        x = torch.from_numpy(vol)[None, None].float().to(device)
        prob = None
        for m in models:
            p = sliding_window_predict(m, x, roi, overlap=cfg.predict.sw_overlap,
                                       mode=cfg.predict.sw_mode, amp=cfg.amp)
            prob = p if prob is None else prob + p
        prob = prob / len(models)
        conf, label = prob.max(1)
        label = label[0].cpu().numpy()
        conf = conf[0].cpu().numpy()
        keep = reliable_keep_np(conf, label, tau, btau, band)
        label = postprocess_volume(label, cfg.num_classes, lcc=True, closing_radius=1)
        fg = int((label > 0).sum())
        if fg < min_fg:
            skipped_empty += 1
            continue
        label = label.astype(np.int16)
        if ignore_index is not None:
            label[~keep] = int(ignore_index)
        nib.save(nib.Nifti1Image(label, _safe_affine(affine, cfg.data.get("spacing"))), str(out_path))
        kept += 1
    print(f"wrote {kept} pseudo-labels -> {out_dir}  "
          f"(skipped {skipped_empty} empty/<{min_fg}fg, {skipped_exist} pre-existing)")


if __name__ == "__main__":
    main()
