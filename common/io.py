from __future__ import annotations

import json
import os
import re
import tarfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def load_dotenv(repo_root: Optional[Path] = None) -> None:
    root = repo_root or Path(__file__).resolve().parent.parent
    env_file = root / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key, val = key.strip(), val.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, val)


def load_config(task: str, overrides: Optional[List[str]] = None):
    from omegaconf import OmegaConf

    cfg_dir = Path(__file__).resolve().parent.parent / "configs"
    load_dotenv(cfg_dir.parent)
    paths = OmegaConf.load(cfg_dir / "paths.yaml")
    common = OmegaConf.load(cfg_dir / "common.yaml")
    task_cfg = OmegaConf.load(cfg_dir / f"{task}.yaml")
    for c in (task_cfg,):
        if "defaults" in c:
            del c["defaults"]
    cfg = OmegaConf.merge({"paths": paths, "metric": common.metric}, common, task_cfg)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    OmegaConf.resolve(cfg)
    return cfg


def cfg_to_plain(cfg):
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(cfg):
            return OmegaConf.to_container(cfg, resolve=True)
    except Exception:
        pass
    return dict(cfg)


def load_checkpoint(path, map_location="cpu"):
    import torch

    return torch.load(str(path), map_location=map_location, weights_only=False)


def _affine_ok(a) -> bool:
    a = np.asarray(a, dtype=np.float64)
    return a.shape == (4, 4) and np.isfinite(a).all() and abs(np.linalg.det(a[:3, :3])) > 1e-8


def read_nifti(path: str | Path) -> Tuple[np.ndarray, np.ndarray, Tuple[float, ...]]:
    import nibabel as nib

    img = nib.load(str(path))
    arr = np.asanyarray(img.dataobj)
    affine = img.affine
    if not _affine_ok(affine):
        for getter in (img.header.get_qform, img.header.get_sform):
            try:
                m, code = getter(coded=True)
            except Exception:
                m, code = None, 0
            if m is not None and code and _affine_ok(m):
                affine = np.asarray(m, dtype=np.float64)
                break
    spacing = tuple(float(z) for z in img.header.get_zooms()[:3])
    if not _affine_ok(affine):
        sp = ([s if (s and np.isfinite(s)) else 1.0 for s in spacing] + [1.0, 1.0, 1.0])[:3]
        affine = np.diag([sp[0], sp[1], sp[2], 1.0]).astype(np.float64)
    return arr, affine, spacing


def nifti_spacing(path: str | Path) -> Tuple[float, ...]:
    import nibabel as nib

    return tuple(float(z) for z in nib.load(str(path)).header.get_zooms()[:3])


def write_nifti(path: str | Path, array: np.ndarray, affine: np.ndarray) -> None:
    import nibabel as nib

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(array.astype(np.uint8), affine), str(path))


def write_nifti_like(path: str | Path, array: np.ndarray, src_path: str | Path,
                     affine: np.ndarray | None = None) -> None:
    import nibabel as nib

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        src = nib.load(str(src_path))
        img = nib.Nifti1Image(array.astype(np.uint8), src.affine, header=src.header.copy())
        img.set_data_dtype(np.uint8)
    except Exception:
        img = nib.Nifti1Image(array.astype(np.uint8), affine if affine is not None else np.eye(4))
    nib.save(img, str(path))


def read_png(path: str | Path, gray: bool = False) -> np.ndarray:
    import cv2

    flag = cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR
    img = cv2.imread(str(path), flag)
    if img is None:
        raise FileNotFoundError(path)
    if not gray:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def write_png(path: str | Path, mask: np.ndarray) -> None:
    import cv2

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), mask.astype(np.uint8))


def read_label_from_tar(tar_path: str | Path, target_label: int = 10) -> np.ndarray:
    import nibabel as nib

    import tempfile

    with tarfile.open(str(tar_path)) as tf:
        member = next((m for m in tf.getmembers() if m.name.endswith((".nii.gz", ".nii"))), None)
        if member is None:
            raise ValueError(f"no nii in {tar_path}")
        data = tf.extractfile(member).read()
    suffix = ".nii.gz" if member.name.endswith(".gz") else ".nii"
    with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
        tmp.write(data)
        tmp.flush()
        arr = np.asanyarray(nib.load(tmp.name).dataobj)
    return (arr == target_label).astype(np.uint8)


FRAME_RE = re.compile(r"^(?P<prefix>.+)_(?P<idx>\d{6})\.png$")


def parse_frame_name(name: str) -> Optional[Tuple[str, int]]:
    m = FRAME_RE.match(os.path.basename(name))
    if not m:
        return None
    return m.group("prefix"), int(m.group("idx"))


def setup_run_dir(
    runs_dir: str | Path,
    task: str,
    exp_name: str,
    cfg=None,
    preserve_existing: bool = False,
) -> Path:
    import subprocess
    import sys

    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        sha = "nogit"
    run = Path(runs_dir) / task / f"{exp_name}_{sha}"
    (run / "checkpoints").mkdir(parents=True, exist_ok=True)
    config_path = run / "config.yaml"
    if cfg is not None and not (preserve_existing and config_path.exists()):
        from omegaconf import OmegaConf

        OmegaConf.save(cfg, config_path)
    freeze_path = run / "pip_freeze.txt"
    if not (preserve_existing and freeze_path.exists()):
        try:
            freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"]).decode()
            freeze_path.write_text(freeze)
        except Exception:
            pass
    return run


def write_predictions_json(task_dir: str | Path, cases: List[Dict[str, str]], task_num: int) -> Path:
    task_dir = Path(task_dir)
    task_dir.mkdir(parents=True, exist_ok=True)
    out = task_dir / f"task{task_num}_predictions.json"
    tmp = out.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump({"cases": cases}, f, indent=2)
    os.replace(tmp, out)
    return out
