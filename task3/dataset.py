from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from common.io import parse_frame_name, read_label_from_tar, read_png
from common.transforms import build_2d_train_transforms


def video_id_of(path: Path) -> str:
    parsed = parse_frame_name(path.name)
    if parsed:
        return parsed[0]
    return path.parent.name


def list_labeled_frames(root: str | Path, split: str, target_label: int = 10) -> List[Dict[str, str]]:
    base = Path(root) / split
    images_dir = base / "images"
    labels_dir = base / "labels"
    items: List[Dict[str, str]] = []
    img_paths = list(images_dir.rglob("*.png")) if images_dir.exists() else list(base.rglob("*.png"))
    img_paths = [p for p in img_paths if "_label" not in p.name.lower()]
    for img in sorted(img_paths):
        stem = img.name[:-4]
        cand_dirs = [labels_dir, img.parent, base]
        mask = None
        mtype = None
        for d in cand_dirs:
            if d is None or not d.exists():
                continue
            bin_png = next(iter(d.rglob(f"{stem}*_label_bin.png")), None) or \
                (d / f"{stem}_label_bin.png" if (d / f"{stem}_label_bin.png").exists() else None)
            if bin_png and Path(bin_png).exists():
                mask, mtype = str(bin_png), "png"
                break
            tar = next(iter(d.rglob(f"{stem}*_png_Label.tar")), None)
            if tar and Path(tar).exists():
                mask, mtype = str(tar), "tar"
                break
        if mask is None:
            continue
        items.append({"image": str(img), "mask": mask, "mask_type": mtype, "video": video_id_of(img)})
    return items


def exclude_frames(items: List[Dict[str, str]], patterns: Sequence[str] | None):
    patterns = tuple(str(p) for p in (patterns or []) if str(p))
    if not patterns:
        return list(items)
    return [it for it in items if not any(p in Path(it["image"]).stem for p in patterns)]


def _frame_key(f):
    p = parse_frame_name(Path(f).name)
    return (0, p[1]) if p else (1, 0, Path(f).name)


def list_unlabeled_videos(images_root: str | Path) -> Dict[str, List[str]]:
    images_root = Path(images_root)
    videos: Dict[str, List[str]] = {}
    if not images_root.exists():
        return videos
    for sub in sorted(p for p in images_root.iterdir() if p.is_dir()):
        frames = sorted((str(f) for f in sub.glob("*.png") if "_label" not in f.name.lower()),
                        key=_frame_key)
        if frames:
            videos[sub.name] = frames
    if not videos:
        for f in images_root.glob("*.png"):
            if "_label" in f.name.lower():
                continue
            p = parse_frame_name(f.name)
            vid = p[0] if p else f.stem
            videos.setdefault(vid, []).append(str(f))
        videos = {v: sorted(fr, key=_frame_key) for v, fr in videos.items()}
    return videos


def load_mask(item: Dict[str, str], target_label: int = 10) -> np.ndarray:
    if item["mask_type"] == "tar":
        m = np.squeeze(read_label_from_tar(item["mask"], target_label))
        if m.ndim == 3:
            m = m.max(axis=int(np.argmin(m.shape)))
        return np.ascontiguousarray((m > 0).astype(np.uint8).T)
    return (read_png(item["mask"], gray=True) > 0).astype(np.uint8)


def list_pseudolabeled_frames(pseudo_dir: str | Path, images_root: str | Path) -> List[Dict[str, str]]:
    pseudo_dir, images_root = Path(pseudo_dir), Path(images_root)
    if not pseudo_dir.exists():
        return []
    items = []
    for vdir in sorted(p for p in pseudo_dir.iterdir() if p.is_dir()):
        for m in sorted(vdir.glob("*_pseudo.png")):
            stem = m.name[: -len("_pseudo.png")]
            img = images_root / vdir.name / f"{stem}.png"
            if img.exists():
                items.append({"image": str(img), "mask": str(m), "mask_type": "png", "video": vdir.name})
    return items


class T3FrameDataset:
    def __init__(self, items: List[Dict[str, str]], cfg, train: bool = True, native_label: bool = False):
        self.items = items
        self.cfg = cfg
        self.train = train
        self.native_label = native_label
        self.tf = build_2d_train_transforms(
            cfg.data.img_size, heavy=cfg.train.get("heavy_aug", False),
            surg_nuisance=cfg.train.get("surg_nuisance", False),
            nuisance_p=float(cfg.train.get("surg_nuisance_p", 0.3))) if train else None
        self.target_label = cfg.get("target_label", 10)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        import cv2
        import torch

        it = self.items[i]
        img = read_png(it["image"], gray=False)
        mask = load_mask(it, self.target_label)
        H, W = self.cfg.data.img_size
        if self.train and self.tf is not None:
            img, mask = self.tf(img, mask)
        else:
            img = cv2.resize(img, (W, H))
            if not self.native_label:
                mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
        img = torch.from_numpy(img.transpose(2, 0, 1).copy()).float() / 255.0
        mask = torch.from_numpy(mask.astype(np.int64))
        return {"image": img, "label": mask, "video": it["video"], "path": it["image"]}


def presence_sample_weights(items: List[Dict[str, str]], target_positive_fraction: float,
                            target_label: int = 10):
    import torch

    target = float(target_positive_fraction)
    if not 0.0 < target < 1.0:
        raise ValueError("target_positive_fraction must be between 0 and 1")
    positive = np.asarray([bool(load_mask(it, target_label).any()) for it in items])
    n_pos = int(positive.sum())
    n_neg = len(items) - n_pos
    if n_pos == 0 or n_neg == 0:
        return torch.ones(len(items), dtype=torch.double)
    pos_weight = target / n_pos
    neg_weight = (1.0 - target) / n_neg
    return torch.as_tensor(np.where(positive, pos_weight, neg_weight), dtype=torch.double)


def split_by_video(items: List[Dict[str, str]], val_frac: float = 0.2, seed: int = 1337, holdout=None):
    vids = sorted({it["video"] for it in items})
    if holdout:
        hs = [holdout] if isinstance(holdout, str) else list(holdout)
        val_vids = {v for v in vids if any(h in v for h in hs)}
        if not val_vids:
            raise SystemExit(f"holdout {holdout!r} matched no video in {[v[-5:] for v in vids]}")
    else:
        rng = np.random.default_rng(seed)
        rng.shuffle(vids)
        n_val = max(1, int(len(vids) * val_frac))
        val_vids = set(vids[:n_val])
    train = [it for it in items if it["video"] not in val_vids]
    val = [it for it in items if it["video"] in val_vids]
    return train, val
