from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from common.transforms import build_3d_train_transforms, build_3d_val_transforms

IMG_SUFFIX = "-US"
LABEL_SUFFIX = "-label"


def _stem(p: Path) -> str:
    n = p.name
    for ext in (".nii.gz", ".nii"):
        if n.endswith(ext):
            return n[: -len(ext)]
    return p.stem


def _images_dir(root: Path, split: str) -> Path:
    cand = root / split / "images"
    return cand if cand.exists() else root / split


def list_pairs(root: str | Path, split: str) -> List[Dict[str, str]]:
    root = Path(root)
    img_dir = _images_dir(root, split)
    lbl_dir = root / split / "labels"
    if not lbl_dir.exists():
        lbl_dir = img_dir
    if not img_dir.exists():
        raise FileNotFoundError(f"{img_dir} not found - inspect dataset layout first")
    pairs = []
    for img in sorted(img_dir.glob(f"*{IMG_SUFFIX}.nii*")):
        base = _stem(img)[: -len(IMG_SUFFIX)] if _stem(img).endswith(IMG_SUFFIX) else _stem(img)
        lbl = None
        for ext in (".nii.gz", ".nii"):
            cand = lbl_dir / f"{base}{LABEL_SUFFIX}{ext}"
            if cand.exists():
                lbl = cand
                break
        if lbl is not None:
            pairs.append({"image": str(img), "label": str(lbl)})
    return pairs


def list_images(root: str | Path, split: str) -> List[Dict[str, str]]:
    root = Path(root)
    for cand in (root / split / "images", root / split, root):
        if cand.exists():
            files = sorted(cand.glob(f"*{IMG_SUFFIX}.nii*")) or sorted(cand.glob("*.nii*"))
            if files:
                return [{"image": str(p)} for p in files]
    raise FileNotFoundError(f"no .nii.gz images found under {root} (split={split})")


def build_datasets(cfg, cache: bool = True):
    from monai.data import CacheDataset, Dataset

    pairs = list_pairs(cfg.paths.t2_root, "train")
    if not pairs:
        raise RuntimeError(f"no t2 train pairs under {cfg.paths.t2_root}/train")
    n_val = min(int(cfg.data.get("val_count", 20)), len(pairs) - 1)
    off = int(cfg.data.get("val_offset", 0)) % len(pairs)
    if off:
        pairs = pairs[off:] + pairs[:off]
    train_list = pairs[:-n_val] if n_val > 0 else pairs
    val_list = pairs[-n_val:] if n_val > 0 else []
    print(f"[task2.dataset] {len(train_list)} train + {len(val_list)} held-out val")
    tr_tf = build_3d_train_transforms(cfg.data.patch_size, modality="us",
                                      oversample_fg=cfg.train.oversample_fg,
                                      speckle=cfg.train.get("speckle_aug", True),
                                      boundary_sdf_classes=cfg.num_classes,
                                      sampling=cfg.data.get("spacing"))
    va_tf = build_3d_val_transforms(modality="us")
    rate = cfg.data.get("cache_rate", 0.0)
    if cache and rate > 0:
        train_ds = CacheDataset(train_list, tr_tf, cache_rate=rate)
        val_ds = CacheDataset(val_list, va_tf, cache_rate=rate)
    else:
        train_ds = Dataset(train_list, tr_tf)
        val_ds = Dataset(val_list, va_tf)
    n_te = int(cfg.data.get("train_eval_count", 0))
    if n_te > 0:
        train_eval_ds = Dataset(train_list[:n_te], va_tf) if train_list else None
        return train_ds, val_ds, train_eval_ds
    return train_ds, val_ds
