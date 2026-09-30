from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from common.transforms import build_3d_train_transforms, build_3d_val_transforms


def _stem(p: Path) -> str:
    name = p.name
    for ext in (".nii.gz", ".nii"):
        if name.endswith(ext):
            return name[: -len(ext)]
    return p.stem


def _labeled_dirs(root: Path, split: str):
    for imgs, lbls in (
        (root / split / "labeled" / "images", root / split / "labeled" / "labels"),
        (root / split / "images", root / split / "labels"),
    ):
        if imgs.exists():
            return imgs, lbls
    return root / split / "labeled" / "images", root / split / "labeled" / "labels"


def list_labeled_pairs(root: str | Path, split: str, label_suffix: str = "-seg") -> List[Dict[str, str]]:
    images_dir, labels_dir = _labeled_dirs(Path(root), split)
    if not images_dir.exists():
        raise FileNotFoundError(f"{images_dir} not found - inspect the dataset layout first")
    label_files = {f.name: f for f in labels_dir.glob("*.nii*")} if labels_dir.exists() else {}
    pairs = []
    for img in sorted(images_dir.glob("*.nii*")):
        st = _stem(img)
        candidates = [f"{st}{label_suffix}.nii.gz", f"{st}{label_suffix}.nii", f"{st}.nii.gz", f"{st}.nii"]
        lbl = next((label_files[c] for c in candidates if c in label_files), None)
        if lbl is not None:
            pairs.append({"image": str(img), "label": str(lbl)})
    return pairs


def list_images(root: str | Path, split: str) -> List[Dict[str, str]]:
    root = Path(root)
    for cand in (root / split / "labeled" / "images", root / split / "images", root / split, root):
        if cand.exists():
            files = sorted(cand.glob("*.nii*"))
            if files:
                return [{"image": str(p)} for p in files]
    raise FileNotFoundError(f"no .nii.gz images found under {root} (split={split})")


def list_unlabeled(root: str | Path, labeled_stems: Optional[set] = None,
                   label_suffix: str = "-seg") -> List[Dict[str, str]]:
    root = Path(root)
    if not root.exists():
        return []
    out = []
    for p in sorted(root.glob("*.nii*")):
        st = _stem(p)
        if st.endswith(label_suffix):
            continue
        if labeled_stems and st in labeled_stems:
            continue
        out.append({"image": str(p)})
    return out


def list_pseudolabeled_pairs(images_dir: str | Path, pseudo_dir: str | Path,
                             label_suffix: str = "-seg") -> List[Dict[str, str]]:
    images_dir, pseudo_dir = Path(images_dir), Path(pseudo_dir)
    if not pseudo_dir.exists():
        return []
    images = {_stem(p): p for p in images_dir.glob("*.nii*")} if images_dir.exists() else {}
    pairs = []
    for lbl in sorted(pseudo_dir.glob(f"*{label_suffix}.nii*")):
        st = _stem(lbl)[: -len(label_suffix)] if _stem(lbl).endswith(label_suffix) else _stem(lbl)
        img = images.get(st)
        if img is not None:
            pairs.append({"image": str(img), "label": str(lbl)})
    return pairs


def build_datasets(cfg, cache: bool = True, return_flags: bool = False):
    import random

    from monai.data import CacheDataset, Dataset

    suffix = cfg.get("label_suffix", "-seg")
    labeled = list_labeled_pairs(cfg.paths.t1_root, "train", suffix)
    if not labeled:
        raise RuntimeError(f"no labeled t1 train pairs under {cfg.paths.t1_root}/train")
    shuffled = list(labeled)
    random.Random(cfg.seed).shuffle(shuffled)
    n_folds = int(cfg.data.get("n_folds", 0))
    if n_folds > 1:
        fold = int(cfg.data.get("fold", 0))
        chunks = [shuffled[i::n_folds] for i in range(n_folds)]
        val_list = chunks[fold]
        train_list = [p for i, c in enumerate(chunks) if i != fold for p in c]
        print(f"[task1.dataset] CV fold {fold}/{n_folds}")
    else:
        n_val = int(cfg.data.get("val_count", 0)) or max(1, round(len(labeled) * cfg.data.get("val_ratio", 0.1)))
        n_val = min(n_val, len(labeled) - 1) if len(labeled) > 1 else 0
        val_list = shuffled[:n_val]
        train_list = shuffled[n_val:]
    labeled_train = list(train_list)
    flags = [True] * len(train_list)
    print(f"[task1.dataset] {len(train_list)} train + {len(val_list)} held-out val (labeled)")
    if cfg.data.get("use_pseudolabels", False):
        unl_dir = cfg.pseudolabel.get("unlabeled_dir", f"{cfg.paths.t1_root}/train/unlabeled")
        pseudo = list_pseudolabeled_pairs(unl_dir, cfg.data.get("pseudolabel_dir", cfg.pseudolabel.out_dir), suffix)
        val_paths = {str(Path(v["image"]).resolve()) for v in val_list}
        leaked = [p for p in pseudo if str(Path(p["image"]).resolve()) in val_paths]
        if leaked:
            raise RuntimeError(f"val leak: {len(leaked)} held-out val case(s) present in the pseudo-label pool "
                               f"({[p['image'] for p in leaked][:3]}...) - check unlabeled_dir")
        print(f"[task1.dataset] {len(train_list)} labeled + {len(pseudo)} pseudo-labeled cases")
        train_list = train_list + pseudo
        flags = flags + [False] * len(pseudo)
    tr_tf = build_3d_train_transforms(cfg.data.patch_size, modality="ct",
                                      hu_clip=tuple(cfg.data.hu_clip),
                                      oversample_fg=cfg.train.oversample_fg,
                                      boundary_sdf_classes=cfg.num_classes,
                                      sampling=cfg.data.get("spacing"),
                                      heavy=cfg.train.get("heavy_aug", False),
                                      spacing_aug=cfg.train.get("spacing_aug", False))
    va_tf = build_3d_val_transforms(modality="ct", hu_clip=tuple(cfg.data.hu_clip))
    rate = cfg.data.get("cache_rate", 0.0)
    if cache and rate > 0:
        train_ds = CacheDataset(train_list, tr_tf, cache_rate=rate)
        val_ds = CacheDataset(val_list, va_tf, cache_rate=rate)
    else:
        train_ds = Dataset(train_list, tr_tf)
        val_ds = Dataset(val_list, va_tf)
    n_te = int(cfg.data.get("train_eval_count", 0))
    train_eval_list = labeled_train[:n_te] if n_te > 0 else []
    train_eval_ds = Dataset(train_eval_list, va_tf) if train_eval_list else None
    if return_flags:
        return train_ds, val_ds, flags, train_eval_ds
    return train_ds, val_ds
