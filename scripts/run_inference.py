from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

APP = Path(os.environ.get("MVAA_APP_DIR", "/app"))
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

WEIGHTS = Path(os.environ.get("MVAA_WEIGHTS_DIR", str(APP / "weights")))
INPUT = Path(os.environ.get("MVAA_INPUT_DIR", "/input"))
OUTPUT = Path(os.environ.get("MVAA_OUTPUT_DIR", "/output"))
_JOB_START = time.time()

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def log(m: str) -> None:
    print(f"[mvaa] {m}", flush=True)


def env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def strip_nii_gz(name: str) -> str:
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return Path(name).stem


def safe_stem(cid: str) -> str:
    s = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in str(cid).strip())
    return s.strip("._") or "case"


def is_label_like(p: Path) -> bool:
    n = p.name.lower()
    return (n.endswith("-seg.nii.gz") or n.endswith("-label.nii.gz") or n.endswith("-pred.nii.gz")
            or n.endswith("_label_bin.png") or "_png_label_vis" in n or n.endswith("_png_label.tar")
            or n.endswith("_pred.png"))


def existing_dirs(cs) -> List[Path]:
    return [c for c in cs if c.exists() and c.is_dir()]


def _normalize_task(key: str) -> Optional[str]:
    s = str(key).strip().lower()
    if s in {"t1", "task1", "task_1", "task1_ct", "t1_ct", "ct"}:
        return "t1_ct"
    if s in {"t2", "task2", "task_2", "task2_tee", "t2_tee", "tee"}:
        return "t2_tee"
    if s in {"t3", "task3", "task_3", "task3_vid", "t3_vid", "video", "vid"}:
        return "t3_vid"
    return None


def _infer_task(item: Dict[str, Any], default: Optional[str]) -> Optional[str]:
    if default:
        return default
    for k in ("task", "task_id", "task_name", "folder", "modality"):
        t = _normalize_task(item.get(k, ""))
        if t:
            return t
    text = " ".join(str(item.get(k, "")) for k in ("case_id", "image", "image_path", "path", "filename", "file")).lower()
    if "t1_ct" in text or "task1" in text:
        return "t1_ct"
    if "t2_tee" in text or "task2" in text or "-us.nii" in text:
        return "t2_tee"
    if "t3_vid" in text or "task3" in text or text.endswith(".png") or "t3_" in text:
        return "t3_vid"
    return None


def _item_image(item: Dict[str, Any]) -> str:
    for k in ("image", "image_path", "image_rel_path", "path", "filename", "file", "input", "input_path"):
        v = item.get(k)
        if v:
            return str(v)
    return ""


def _extract_entries(raw: Any) -> List[Tuple[Optional[str], Dict[str, Any]]]:
    if isinstance(raw, list):
        return [(None, x) for x in raw if isinstance(x, dict)]
    if not isinstance(raw, dict):
        return []
    out: List[Tuple[Optional[str], Dict[str, Any]]] = []
    for key in ("cases", "test_cases", "data", "samples", "inputs"):
        v = raw.get(key)
        if isinstance(v, list):
            out.extend((None, x) for x in v if isinstance(x, dict))
    for key, v in raw.items():
        t = _normalize_task(key)
        if t and isinstance(v, list):
            out.extend((t, x) for x in v if isinstance(x, dict))
        elif t and isinstance(v, dict):
            out.extend((t, item) for _, item in _extract_entries(v))
    return out


def _task_roots(d: Path, task: str) -> List[Path]:
    return existing_dirs([d / task / "images", d / task])


def _resolve_image(d: Path, task: str, item: Dict[str, Any]) -> Optional[Path]:
    val = _item_image(item).strip()
    cands: List[Path] = []
    if val:
        raw = Path(val)
        if raw.is_absolute():
            cands.append(raw)
        else:
            cands += [d / raw, d / task / raw, d / task / "images" / raw]
            for r in _task_roots(d, task):
                cands += [r / raw, r / Path(val).name]
    for p in cands:
        if p.exists() and p.is_file():
            return p.resolve()
    base = Path(val).name if val else ""
    roots = _task_roots(d, task) or [d / task, d]
    if base:
        for r in roots:
            hits = sorted(p for p in r.rglob(base) if p.is_file()) if r.exists() else []
            if hits:
                return hits[0].resolve()
    cid = str(item.get("case_id") or item.get("id") or "").strip()
    if cid:
        for r in roots:
            if not r.exists():
                continue
            if task == "t1_ct":
                hits = sorted(p for p in r.rglob(f"{cid}*.nii.gz") if "-US" not in p.name and not is_label_like(p))
            elif task == "t2_tee":
                hits = sorted(p for p in r.rglob(f"{cid}*-US.nii.gz") if not is_label_like(p))
            else:
                hits = sorted(p for p in r.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS
                              and not is_label_like(p) and (p.stem == cid or p.name == cid))
            if hits:
                return hits[0].resolve()
    return None


def load_manifest(d: Path) -> Optional[Dict[str, List[Dict[str, Any]]]]:
    path = d / "test_cases.json"
    if not path.exists():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    out: Dict[str, List[Dict[str, Any]]] = {"t1_ct": [], "t2_tee": [], "t3_vid": []}
    for default_task, item in _extract_entries(raw):
        task = _infer_task(item, default_task)
        if task not in out:
            continue
        cid = str(item.get("case_id") or item.get("id") or "").strip() or strip_nii_gz(Path(_item_image(item)).name)
        if task == "t2_tee" and cid.endswith("-US"):
            cid = cid[:-3]
        img = _resolve_image(d, task, item)
        if img is None:
            log(f"WARN manifest: cannot resolve image for {task} {cid} ({item}); skipping")
            continue
        out[task].append({"case_id": cid, "image_path": img})
    log(f"manifest test_cases.json: " + ", ".join(f"{k}={len(v)}" for k, v in out.items()))
    return out


def discover_t1(d: Path) -> List[Dict[str, Any]]:
    for c in existing_dirs([d / "t1_ct" / "images", d / "t1_ct", d / "task1" / "images"]):
        f = sorted(p for p in c.glob("*.nii.gz") if "-US" not in p.name and not is_label_like(p))
        if f:
            return [{"case_id": strip_nii_gz(p.name), "image_path": p} for p in f]
    f = sorted(p for p in d.rglob("*.nii.gz") if "-US" not in p.name and not is_label_like(p)
               and "t2_tee" not in {x.lower() for x in p.parts})
    return [{"case_id": strip_nii_gz(p.name), "image_path": p} for p in f]


def discover_t2(d: Path) -> List[Dict[str, Any]]:
    for c in existing_dirs([d / "t2_tee" / "images", d / "t2_tee", d / "task2" / "images"]):
        f = sorted(p for p in c.glob("*-US.nii.gz") if not is_label_like(p))
        if f:
            return [{"case_id": strip_nii_gz(p.name.replace("-US.nii.gz", ".nii.gz")), "image_path": p} for p in f]
    f = sorted(p for p in d.rglob("*-US.nii.gz") if not is_label_like(p))
    return [{"case_id": strip_nii_gz(p.name.replace("-US.nii.gz", ".nii.gz")), "image_path": p} for p in f]


def discover_t3(d: Path) -> List[Dict[str, Any]]:
    roots = existing_dirs([d / "t3_vid" / "images", d / "t3_vid", d / "task3" / "images", d / "images"]) or [d]
    for c in roots:
        f = sorted(p for p in c.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS and not is_label_like(p))
        if f:
            return [{"case_id": p.stem, "image_path": p} for p in f]
    return []


def _video_id(fpath: Path) -> str:
    stem = fpath.stem
    parts = stem.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return fpath.parent.name


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _empty_nifti(path: Path, src) -> None:
    import numpy as np, nibabel as nib
    try:
        img = nib.load(str(src)); arr = np.zeros(img.shape[:3], dtype=np.uint8); aff = img.affine
    except Exception:
        arr = np.zeros((16, 16, 16), dtype=np.uint8); aff = np.eye(4)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(arr, aff), str(path))


def _empty_png(path: Path, src) -> None:
    import numpy as np, cv2
    try:
        h, w = cv2.imread(str(src)).shape[:2]
    except Exception:
        h, w = 16, 16
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.zeros((h, w), np.uint8))


def t3_rel(case_id: str) -> str:
    return f"{safe_stem(case_id)}_pred.png"


def t3_member_weight(stem: str) -> float:
    tags = [
        part
        for part in stem.split("_")
        if len(part) > 1 and part[0] == "w" and part[1:].isdigit()
    ]
    if not tags:
        return 1.0
    if len(tags) != 1:
        raise ValueError(f"T3 checkpoint has multiple weight tags: {stem}")
    weight = int(tags[0][1:]) / 100.0
    if weight <= 0:
        raise ValueError(f"T3 checkpoint has nonpositive weight: {stem}")
    return weight


def t3_checkpoint_plan(
    root: Path,
    layered: bool = True,
    cv_ensemble: bool = False,
):
    if cv_ensemble:
        checkpoints = sorted(root.glob("*_cv.pt"))
        if not checkpoints:
            raise FileNotFoundError(f"CV T3 ensemble has no *_cv.pt checkpoints under {root}")
        missing_weights = [
            path.name
            for path in checkpoints
            if not any(
                len(part) > 1 and part[0] == "w" and part[1:].isdigit()
                for part in path.stem.split("_")
            )
        ]
        if missing_weights:
            raise ValueError(f"CV T3 checkpoints missing weight tags: {missing_weights}")
        for path in checkpoints:
            t3_member_weight(path.stem)
        return "cv", checkpoints
    if not layered:
        checkpoints = [root / "s42.pt", root / "s44.pt"]
        missing = [path.name for path in checkpoints if not path.exists()]
        if missing:
            raise FileNotFoundError(f"conservative T3 missing checkpoints: {missing}")
        return "legacy", checkpoints
    guard = root / "guard_e37.pt"
    if not guard.exists():
        return "legacy", sorted(root.glob("*.pt"))
    checkpoints = [root / "s42.pt", root / "s44.pt", guard]
    missing = [path.name for path in checkpoints if not path.exists()]
    if missing:
        raise FileNotFoundError(f"layered T3 missing checkpoints: {missing}")
    return "layered", checkpoints


def _dedup_entries(entries: List[Dict[str, Any]], out_dir: Path) -> List[Dict[str, Any]]:
    seen, fixed = {}, []
    for e in entries:
        rel = e["segmentation"]
        if rel in seen:
            stem, _, ext = rel.rpartition(".")
            n = seen[rel] = seen[rel] + 1
            rel = f"{stem}__{n}.{ext}"
            log(f"WARNING duplicate output path for case_id={e['case_id']!r} -> writing {rel} instead")
        else:
            seen[rel] = 0
        fixed.append({**e, "segmentation": rel})
    return fixed


def _fallback_task(cases, folder: str, kind: str, task_num: int) -> None:
    from common.io import write_predictions_json
    out_dir = OUTPUT / folder
    out_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for cs in cases:
        if kind == "nii":
            rel = f"{safe_stem(cs['case_id'])}.nii.gz"
            if not (out_dir / rel).exists():
                _empty_nifti(out_dir / rel, cs["image_path"])
        else:
            rel = t3_rel(cs["case_id"])
            if not (out_dir / rel).exists():
                _empty_png(out_dir / rel, cs["image_path"])
        entries.append({"case_id": cs["case_id"], "segmentation": rel})
    write_predictions_json(out_dir, _dedup_entries(entries, out_dir), task_num=task_num)


def run_t1(cases: List[Dict[str, Any]], device) -> None:
    from common.io import load_config, write_nifti_like, write_predictions_json
    from task1.predict import load_student
    from scripts.predict_ensemble_t1 import predict_volume_ensemble

    def _t1_predict(models, path, cfg, device):
        if cfg.predict.get("spacing_rescue", {}).get("enabled", False):
            try:
                from common.spacing_guard import predict_volume_ensemble_rescued
                return predict_volume_ensemble_rescued(models, path, cfg, device)
            except Exception as exc:                       # noqa: BLE001
                log(f"T1 rescue failed ({type(exc).__name__}: {exc}) -> unguarded prediction")
        return predict_volume_ensemble(models, path, cfg, device)
    cfg = load_config("task1", ["predict.tta_flip_axes=[0,1,2]", "predict.tta_full_combos=true",
                                "predict.sw_batch_size=2"])
    if "spacing_guard" not in cfg.predict:
        cfg.predict.spacing_guard = {}
    if "canonical_fusion" not in cfg.predict:
        cfg.predict.canonical_fusion = {}
    cfg.predict.spacing_guard.enabled = env_flag(
        "MVAA_T1_SPACING_GUARD", bool(cfg.predict.spacing_guard.get("enabled", False)))
    cfg.predict.canonical_fusion.enabled = env_flag(
        "MVAA_T1_CANONICAL_FUSION", bool(cfg.predict.canonical_fusion.get("enabled", False)))
    if "spacing_rescue" not in cfg.predict:
        cfg.predict.spacing_rescue = {}
    cfg.predict.spacing_rescue.enabled = env_flag(
        "MVAA_T1_SPACING_RESCUE", bool(cfg.predict.spacing_rescue.get("enabled", False)))
    if cfg.predict.spacing_rescue.enabled:
        log("T1 spacing RESCUE enabled (replaces only implausible+misplaced out-of-band predictions)")
    if cfg.predict.spacing_guard.enabled:
        log("T1 spacing guard enabled for out-of-range voxel spacing")
    if cfg.predict.canonical_fusion.enabled:
        log("T1 conditional native/canonical fusion enabled for in-range thick slices")
    cfgL = load_config("task1", ["model.stunet_size=L"])
    cfgB = load_config("task1", ["model.stunet_size=B"])
    wdir = WEIGHTS / "task1"
    ckH = sorted(wdir.glob("H_*.pt"))
    ckL, ckB = sorted(wdir.glob("L_*.pt")), sorted(wdir.glob("B_*.pt"))
    if not (ckL or ckB or ckH):
        raise FileNotFoundError(f"no T1 weights (H_*.pt / L_*.pt / B_*.pt) under {wdir}")
    cfgH = load_config("task1", ["model.stunet_size=H"]) if ckH else None
    models = [load_student(cfgL, str(c), device).eval() for c in ckL] \
           + [load_student(cfgB, str(c), device).eval() for c in ckB] \
           + [load_student(cfgH, str(c), device).eval() for c in ckH]
    log(f"T1: {len(ckL)} STU-Net-L + {len(ckB)} STU-Net-B + {len(ckH)} STU-Net-H members, "
        f"8-way TTA, {len(cases)} cases")
    out_dir = OUTPUT / "t1_ct"
    out_dir.mkdir(parents=True, exist_ok=True)
    deadline = float(os.environ.get("MVAA_DEADLINE_S", "39600"))
    entries = []
    for cs in cases:
        t0 = time.time()
        rel = f"{safe_stem(cs['case_id'])}.nii.gz"
        if time.time() - _JOB_START > deadline:
            log(f"T1 deadline {deadline:.0f}s reached -> empty fallback for {cs['case_id']} (and remaining)")
            _empty_nifti(out_dir / rel, cs["image_path"])
            entries.append({"case_id": cs["case_id"], "segmentation": rel})
            write_predictions_json(out_dir, entries, task_num=1)
            continue
        try:
            pred, affine = _t1_predict(models, str(cs["image_path"]), cfg, device)
            write_nifti_like(out_dir / rel, pred, cs["image_path"], affine)
            log(f"T1 {cs['case_id']}: {time.time()-t0:.1f}s")
        except Exception as e:
            log(f"T1 {cs['case_id']} FAILED ({type(e).__name__}: {e}) -> empty fallback")
            _empty_nifti(out_dir / rel, cs["image_path"])
        entries.append({"case_id": cs["case_id"], "segmentation": rel})
        write_predictions_json(out_dir, entries, task_num=1)


def run_t2(cases: List[Dict[str, Any]], device) -> None:
    from common.io import load_config, write_nifti_like, write_predictions_json
    from task2.predict import load_student
    from scripts.predict_ensemble_t2_ms import predict_volume_multiscale, parse_geom
    cfg = load_config("task2", ["model.stunet_size=B", "predict.sw_batch_size=2"])
    GEOMS = [parse_geom(s, cfg) for s in ("native", "0.3727,0.5396,0.2322:160:constant")]
    ckpts = sorted((WEIGHTS / "task2").glob("*.pt"))
    if not ckpts:
        raise FileNotFoundError(f"no T2 weights under {WEIGHTS/'task2'}")
    models = [load_student(cfg, str(c), device).eval() for c in ckpts]
    log(f"T2: {len(models)} STU-Net-B members x {len(GEOMS)} geometries, 8-way TTA, {len(cases)} cases")
    out_dir = OUTPUT / "t2_tee"
    out_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for cs in cases:
        t0 = time.time()
        rel = f"{safe_stem(cs['case_id'])}.nii.gz"
        try:
            pred, affine = predict_volume_multiscale(models, str(cs["image_path"]), cfg, device,
                                                     GEOMS, [0, 1, 2], True)
            write_nifti_like(out_dir / rel, pred, cs["image_path"], affine)
            log(f"T2 {cs['case_id']}: {time.time()-t0:.1f}s")
        except Exception as e:
            log(f"T2 {cs['case_id']} FAILED ({type(e).__name__}: {e}) -> empty fallback")
            _empty_nifti(out_dir / rel, cs["image_path"])
        entries.append({"case_id": cs["case_id"], "segmentation": rel})
        write_predictions_json(out_dir, entries, task_num=2)


def run_t3(cases: List[Dict[str, Any]], device) -> None:
    from common.io import load_config, write_png, write_predictions_json
    from scripts.predict_ensemble_t3 import load_one, predict_frame, predict_layered_frame
    _ARCH = {"unetpp": "unetplusplus", "dlv3": "deeplabv3plus", "manet": "manet", "unet": "unet", "fpn": "fpn"}
    _ENC = {"effb4": "efficientnet-b4", "effb5": "efficientnet-b5", "effb6": "efficientnet-b6",
            "effb3": "efficientnet-b3"}
    def _member_cfg(stem: str):
        parts = stem.split("_")
        arch = _ARCH.get(parts[0], "unetplusplus")
        enc = _ENC.get(parts[1] if len(parts) > 1 else "", "efficientnet-b4")
        return load_config(
            "task3",
            [
                "predict.lcc=false",
                "predict.resize_probability=false",
                "model.encoder_weights=null",
                f"model.arch={arch}",
                f"model.encoder={enc}",
            ],
        )
    mode, ckpts = t3_checkpoint_plan(
        WEIGHTS / "task3",
        env_flag("MVAA_T3_LAYERED", True),
        env_flag("MVAA_T3_CV_ENSEMBLE", False),
    )
    if not ckpts:
        raise FileNotFoundError(f"no T3 weights under {WEIGHTS/'task3'}")
    models, cfg = [], None
    for c in ckpts:
        mcfg = _member_cfg(c.stem)
        models.append(load_one(mcfg, str(c), device))
        cfg = cfg or mcfg
    weights = [t3_member_weight(c.stem) for c in ckpts]
    if mode == "layered":
        _lt = float(cfg.predict.get("layered_threshold", 0.45))
        _lr = float(cfg.predict.get("layered_recovery_threshold", 0.10))
        log(
            f"T3: layered guard_e37+s44@{_lt:.2f}, incumbent s42+s44@{_lt:.2f}, "
            f"recovery guard_e37@{_lr:.2f}, {len(cases)} frames"
        )
    else:
        log(f"T3: {len(models)} members [{', '.join(c.stem for c in ckpts)}] weights={weights}, "
            f"threshold={float(cfg.predict.get('threshold', 0.5)):.2f}, no-LCC/no-TTA, {len(cases)} frames")
    out_dir = OUTPUT / "t3_vid"
    out_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for i, cs in enumerate(cases):
        fpath = Path(cs["image_path"])
        rel = t3_rel(cs["case_id"])
        try:
            if mode == "layered":
                mask = predict_layered_frame(
                    models[0], models[1], models[2], str(fpath), cfg, device
                )
            else:
                mask = predict_frame(models, str(fpath), cfg, device, False, weights)
            write_png(out_dir / rel, mask * 255)
        except Exception as e:
            log(f"T3 {cs['case_id']} FAILED ({type(e).__name__}: {e}) -> empty fallback")
            _empty_png(out_dir / rel, cs["image_path"])
        entries.append({"case_id": cs["case_id"], "segmentation": rel})
        if i % 50 == 0 or i == len(cases) - 1:
            write_predictions_json(out_dir, _dedup_entries(entries, out_dir), task_num=3)
    entries = _dedup_entries(entries, out_dir)
    missing = [e for e in entries if not (out_dir / e["segmentation"]).exists()]
    if missing:
        log(f"T3 WARNING {len(missing)} declared masks absent on disk -> writing empty fallbacks")
        for e in missing:
            _empty_png(out_dir / e["segmentation"], cases[0]["image_path"])
    log(f"T3 wrote {len(entries)} masks, {len({e['segmentation'] for e in entries})} unique paths")
    write_predictions_json(out_dir, entries, task_num=3)


@torch.inference_mode()
def main() -> None:
    tasks = os.environ.get("MVAA_TASKS", "1,2,3").split(",")
    device = _device()
    log(f"input={INPUT} output={OUTPUT} weights={WEIGHTS} device={device} tasks={tasks}")
    from common.io import write_predictions_json as _wj
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for folder, num in [("t1_ct", 1), ("t2_tee", 2), ("t3_vid", 3)]:
        (OUTPUT / folder).mkdir(parents=True, exist_ok=True)
        try:
            _wj(OUTPUT / folder, [], num)
        except Exception:
            pass
    try:
        manifest = load_manifest(INPUT)
    except Exception as e:
        log(f"manifest parse failed ({type(e).__name__}: {e}) -> dir-glob discovery"); manifest = None

    def _disc(fn):
        try:
            return fn(INPUT)
        except Exception as e:
            log(f"{fn.__name__} failed ({type(e).__name__}: {e})"); return []
    t1 = manifest["t1_ct"] if manifest and manifest["t1_ct"] else _disc(discover_t1)
    t2 = manifest["t2_tee"] if manifest and manifest["t2_tee"] else _disc(discover_t2)
    t3 = manifest["t3_vid"] if manifest and manifest["t3_vid"] else _disc(discover_t3)
    log(f"discovered: t1={len(t1)} t2={len(t2)} t3={len(t3)}")

    if "3" in tasks and t3:
        try:
            run_t3(t3, device)
        except Exception as e:
            log(f"TASK3 FAILED ({type(e).__name__}: {e}) -> empty fallbacks for {len(t3)} frames")
            _fallback_task(t3, "t3_vid", "png", 3)
    if "2" in tasks and t2:
        try:
            run_t2(t2, device)
        except Exception as e:
            log(f"TASK2 FAILED ({type(e).__name__}: {e}) -> empty fallbacks for {len(t2)} cases")
            _fallback_task(t2, "t2_tee", "nii", 2)
    if "1" in tasks and t1:
        try:
            run_t1(t1, device)
        except Exception as e:
            log(f"TASK1 FAILED ({type(e).__name__}: {e}) -> empty fallbacks for {len(t1)} cases")
            _fallback_task(t1, "t1_ct", "nii", 1)
    log(f"done -> {OUTPUT}")


if __name__ == "__main__":
    main()
