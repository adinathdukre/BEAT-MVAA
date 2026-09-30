from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import shutil
import subprocess
from pathlib import Path

import numpy as np
import torch
import torch.distributed as tdist
import torch.nn.functional as F
from torch.utils.data import DataLoader

from common.dist import (FixedStepBatchSampler, TwoStreamBatchSampler, barrier, cleanup, is_dist,
                         is_main, make_loader, reduce_mean, setup_distributed, unwrap, wrap_ddp)
from common.distill import FeatureDistill, build_ct_distill_teachers
from common.io import cfg_to_plain, load_checkpoint, load_config, nifti_spacing, setup_run_dir
from common.losses import CompositeSegLoss
from common.metrics import aggregate, evaluate_case
from common.postprocess import postprocess_volume
from common.registry import build_student3d
from common.sliding_window import sliding_window_predict
from common.transforms import tensor_fbpaug
from task1.dataset import build_datasets


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_torch_save(payload: dict, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    torch.save(payload, temporary)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, target)


def _atomic_copy(source: str | Path, target: str | Path) -> None:
    destination = Path(target)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copyfile(source, temporary)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, destination)


def _atomic_json(payload: dict, path: str | Path) -> None:
    target = Path(path)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, target)


def _rng_state() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": (
            torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else []
        ),
    }


def _restore_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _sampler_state(sampler) -> dict | None:
    if sampler is None:
        return None
    if hasattr(sampler, "state_dict"):
        return sampler.state_dict()
    return {"epoch": int(getattr(sampler, "epoch", 0))}


def _restore_sampler_state(sampler, state: dict | None) -> None:
    if sampler is None:
        if state is not None:
            raise ValueError("checkpoint has sampler state but loader has no sampler")
        return
    if state is None:
        raise ValueError("checkpoint is missing sampler state")
    if hasattr(sampler, "load_state_dict"):
        sampler.load_state_dict(state)
    elif hasattr(sampler, "set_epoch"):
        sampler.set_epoch(int(state["epoch"]))


def _state_finite(state: dict) -> bool:
    for t in state.values():
        if torch.is_tensor(t) and t.is_floating_point() and not torch.isfinite(t).all():
            return False
    return True


def _optimizer_groups(model, adapters, lr: float, backbone_lr_scale: float):
    named = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    adapter_params = [p for adapter in adapters.values() for p in adapter.parameters() if p.requires_grad]
    if backbone_lr_scale == 1.0:
        return [p for _, p in named] + adapter_params
    heads = [p for name, p in named if "seg_outputs" in name]
    backbone = [p for name, p in named if "seg_outputs" not in name]
    if not heads:
        raise ValueError("backbone_lr_scale requires a model with seg_outputs")
    groups = [
        {"params": backbone, "lr": lr * backbone_lr_scale},
        {"params": heads, "lr": lr},
    ]
    if adapter_params:
        groups.append({"params": adapter_params, "lr": lr})
    return groups


def _freeze_backbone(model):
    trainable = 0
    for name, parameter in model.named_parameters():
        keep = "seg_outputs" in name
        parameter.requires_grad_(keep)
        trainable += parameter.numel() if keep else 0
    if trainable == 0:
        raise ValueError("freeze_backbone requires a model with seg_outputs")
    return trainable


def _build_optimizer(cfg, params):
    name = str(cfg.train.get("optimizer", "sgd")).lower()
    if name == "sgd":
        return torch.optim.SGD(
            params,
            lr=cfg.train.lr,
            momentum=cfg.train.momentum,
            weight_decay=cfg.train.weight_decay,
            nesterov=True,
        )
    if name == "adamw":
        return torch.optim.AdamW(params, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    raise ValueError(f"unsupported Task 1 optimizer: {name}")


def _all_ranks_finite(local_finite: bool, device: torch.device) -> bool:
    flag = torch.tensor([1.0 if local_finite else 0.0], device=device)
    if is_dist():
        tdist.all_reduce(flag, op=tdist.ReduceOp.MIN)
    return bool(flag.item() > 0.0)


def _synchronized_backoff_scale(scaler, device: torch.device) -> float:
    scale = torch.tensor([float(scaler.get_scale())], device=device)
    if is_dist():
        tdist.all_reduce(scale, op=tdist.ReduceOp.MIN)
    return float(scale.item()) * float(scaler.get_backoff_factor())


def _guarded_optimizer_step(
    loss: torch.Tensor,
    optimizer,
    scaler,
    use_scaler: bool,
    parameter_groups,
    max_norm: float,
    device: torch.device,
) -> bool:
    groups = [tuple(parameters) for parameters in parameter_groups]
    if use_scaler:
        scaler.unscale_(optimizer)
    local_finite = bool(torch.isfinite(loss.detach()).all().item())
    if local_finite:
        for parameters in groups:
            if not parameters:
                continue
            total_norm = torch.nn.utils.clip_grad_norm_(
                parameters,
                max_norm=max_norm,
            )
            local_finite = local_finite and bool(torch.isfinite(total_norm).item())
    should_step = _all_ranks_finite(local_finite, device)
    if should_step:
        scaler.step(optimizer)
        scaler.update()
        return True
    if use_scaler:
        scaler.update(new_scale=_synchronized_backoff_scale(scaler, device))
    else:
        scaler.update()
    optimizer.zero_grad(set_to_none=True)
    return False


def _step_metadata(loader_steps, successful_steps, skipped_steps, scheduler):
    return {
        "loader_steps_per_epoch": int(loader_steps),
        "global_step": int(successful_steps + skipped_steps),
        "successful_optimizer_steps": int(successful_steps),
        "skipped_optimizer_steps": int(skipped_steps),
        "scheduler_last_epoch": int(scheduler.last_epoch),
    }


def _selection_state_from_history(run_dir, checkpoint_epoch, cfg, start_epoch, window, keep_topk):
    history_path = run_dir / "history.csv"
    if not history_path.is_file():
        return None
    with history_path.open(newline="") as stream:
        rows = [
            row for row in csv.DictReader(stream)
            if int(row["epoch"]) <= int(checkpoint_epoch)
        ]
    if not rows or int(rows[-1]["epoch"]) != int(checkpoint_epoch):
        return None
    metric = "dice" if cfg.train.get("select_metric", "score") == "dice" else "score"
    values = []
    ranked = []
    best_value = -1.0
    no_improve = 0
    for row in rows:
        epoch = int(row["epoch"])
        values.append(float(row[metric]))
        smoothed = float(np.mean(values[-window:]))
        if epoch < start_epoch:
            continue
        path = run_dir / "checkpoints" / f"ckpt_e{epoch}.pt"
        ranked.append((smoothed, epoch, path))
        if smoothed > best_value:
            best_value = smoothed
            no_improve = 0
        else:
            no_improve += 1
    ranked.sort(key=lambda item: item[0], reverse=True)
    topk = ranked[:keep_topk]
    missing = [str(path) for _, _, path in topk if not path.is_file()]
    if missing:
        raise ValueError(f"resume selection checkpoints are missing: {missing}")
    return values, topk, no_improve, best_value


@torch.inference_mode()
def validate(model, val_loader, cfg, device) -> dict:
    model.eval()
    cases = []
    roi = cfg.data.patch_size
    items = val_loader.dataset.data
    for batch, item in zip(val_loader, items):
        img = batch["image"].to(device)
        prob = sliding_window_predict(model, img, roi, overlap=cfg.predict.sw_overlap,
                                      mode=cfg.predict.sw_mode, amp=cfg.amp,
                                      flip_axes=cfg.predict.get("tta_flip_axes", []))
        pred = prob.argmax(1).cpu().numpy()[0]
        pred = postprocess_volume(pred, cfg.num_classes, lcc=cfg.predict.lcc,
                                  closing_radius=cfg.predict.closing_radius,
                                  lcc_min_ratio=cfg.predict.get("lcc_min_ratio", 0.0))
        gt = batch["label"].cpu().numpy()[0]
        if gt.ndim == 4:
            gt = gt[0]
        cases.append(evaluate_case(pred, gt, cfg.num_classes, spacing=nifti_spacing(item["image"]),
                                   hd_ref=cfg.metric.hd_ref, asd_ref=cfg.metric.asd_ref))
    return aggregate(cases)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="student")
    ap.add_argument("--resume", default=None,
                    help="path to a last.pt to resume from (model+opt+sched+scaler+adapters+epoch+best); "
                         "'auto' = <run>/checkpoints/last.pt for this exp if it exists")
    ap.add_argument("--init", default=None, help="initialize model weights from a task checkpoint")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    if args.init and args.resume:
        ap.error("--init and --resume are mutually exclusive")

    cfg = load_config("task1", args.overrides)
    rank, world, device = setup_distributed()
    set_seed(cfg.seed + rank)
    run = setup_run_dir(
        cfg.paths.runs_dir,
        "task1",
        args.exp,
        cfg,
        preserve_existing=bool(args.resume),
    ) if is_main() else None
    if is_main() and args.resume:
        from omegaconf import OmegaConf

        existing_cfg = OmegaConf.load(run / "config.yaml")
        if cfg_to_plain(existing_cfg) != cfg_to_plain(cfg):
            raise ValueError("resume config differs from the preserved run config")
    try:
        _sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        _sha = "nogit"
    run_dir = Path(cfg.paths.runs_dir) / "task1" / f"{args.exp}_{_sha}"
    resume_path = (run_dir / "checkpoints" / "last.pt") if args.resume == "auto" \
        else (Path(args.resume) if args.resume else None)

    from monai.data import list_data_collate

    train_ds, val_ds, labeled_flags, train_eval_ds = build_datasets(cfg, return_flags=True)
    lab_frac = float(cfg.data.get("labeled_fraction", 0.0))
    two_stream = lab_frac > 0 and any(labeled_flags) and not all(labeled_flags)
    fixed_steps = int(cfg.train.get("steps_per_epoch", 0))
    exact_steps = bool(cfg.train.get("exact_steps", False))
    if two_stream:
        labeled_idx = [i for i, f in enumerate(labeled_flags) if f]
        unlabeled_idx = [i for i, f in enumerate(labeled_flags) if not f]
        bsampler = TwoStreamBatchSampler(labeled_idx, unlabeled_idx, cfg.train.batch_size,
                                         labeled_fraction=lab_frac, rank=rank, world_size=world,
                                         seed=cfg.seed, max_batches=fixed_steps)
        if is_main() and bsampler.singleton_mixed:
            print(f"[student] singleton two-stream schedule: {lab_frac:.1%} labeled batches, "
                  f"{1.0 - lab_frac:.1%} pseudo batches, {len(bsampler)} batches/epoch/rank")
        elif is_main():
            print(f"[student] two-stream batches: {bsampler.labeled_bs} labeled + "
                  f"{bsampler.unlabeled_bs} pseudo/batch, {len(bsampler)} batches/epoch/rank")
        train_loader, train_sampler = make_loader(
            train_ds, batch_size=cfg.train.batch_size, shuffle=True, num_workers=cfg.train.num_workers,
            collate_fn=list_data_collate, pin_memory=True, batch_sampler=bsampler)
    elif fixed_steps > 0 and labeled_flags and all(labeled_flags):
        bsampler = FixedStepBatchSampler(
            range(len(train_ds)), cfg.train.batch_size, fixed_steps,
            rank=rank, world_size=world, seed=cfg.seed,
        )
        if is_main():
            print(f"[student] labeled-only fixed-step batches: "
                  f"{len(bsampler)} batches/epoch/rank")
        train_loader, train_sampler = make_loader(
            train_ds, batch_size=cfg.train.batch_size, shuffle=True, num_workers=cfg.train.num_workers,
            collate_fn=list_data_collate, pin_memory=True, batch_sampler=bsampler)
    else:
        train_loader, train_sampler = make_loader(
            train_ds, batch_size=cfg.train.batch_size, shuffle=True, num_workers=cfg.train.num_workers,
            collate_fn=list_data_collate, drop_last=True, pin_memory=True)
    if exact_steps:
        if world != 1:
            raise RuntimeError("exact-step training requires one process")
        if two_stream or not labeled_flags or not all(labeled_flags):
            raise RuntimeError("exact-step training requires labeled-only data")
        if fixed_steps <= 0 or len(train_loader) != fixed_steps:
            raise RuntimeError(
                f"exact-step loader mismatch: configured={fixed_steps}, "
                f"observed={len(train_loader)}"
            )
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=2) if is_main() else None
    train_eval_loader = (DataLoader(train_eval_ds, batch_size=1, num_workers=2)
                         if is_main() and train_eval_ds is not None else None)

    model = build_student3d(cfg).to(device)

    try:
        ct_teachers = build_ct_distill_teachers(cfg)
    except Exception as e:
        print(f"[distill] CT teacher unavailable ({type(e).__name__}: {e}) - training WITHOUT CT distillation")
        ct_teachers = []
    ct_adapters = {}
    if ct_teachers:
        ok, s_ch = model.enable_distill_capture()
        if ok:
            for name, t, dim in ct_teachers:
                t.to(device)
                ct_adapters[name] = FeatureDistill(s_ch, dim, dims=3).to(device)
            print(f"[distill] CT teachers {[n for n,_,_ in ct_teachers]} -> student feat ch={s_ch}")
        else:
            print("[distill] could not hook student feature - CT distillation disabled")
            ct_teachers = []

    fcon = cfg.train.get("fconsist", {})
    fcon_on = bool(fcon.get("enabled", False))
    fcon_w = float(fcon.get("weight", 1.0))
    fcon_ramp = int(fcon.get("rampup_epochs", 40))
    fcon_alpha = tuple(fcon.get("fbp_alpha", [-0.6, 1.2]))
    fcon_beta = tuple(fcon.get("fbp_beta", [1.5, 3.0]))
    if fcon_on:
        ok, si = model.enable_fconsist_capture(int(fcon.get("stage", 1)))
        fcon_on = ok
        if is_main():
            print(f"[fconsist] {'ON' if ok else 'FAILED to hook'} stage={si} weight={fcon_w} rampup={fcon_ramp}")

    if args.init:
        state = load_checkpoint(args.init, map_location=device)
        if not _state_finite(state["model"]):
            raise ValueError(f"{args.init} contains non-finite weights")
        model.load_state_dict(state["model"], strict=True)
        if is_main():
            print(f"[student] initialized model from {args.init}")

    if cfg.train.get("freeze_backbone", False):
        trainable = _freeze_backbone(model)
        if is_main():
            print(f"[student] head-only warmup trainable_params={trainable}")

    if cfg.train.get("grad_checkpointing", False) and model.enable_grad_checkpointing():
        print("[student] activation checkpointing ON")
    model = wrap_ddp(model, device, find_unused_parameters=True)
    shape_prior = None
    if float(cfg.loss.get("shape_prior", 0)) > 0:
        from common.shape_prior import load_shape_prior
        shape_prior = load_shape_prior(cfg.predict.get("shape_prior", {}).get("ckpt"), dims=3, device=device)
    loss_fn = CompositeSegLoss(cfg.num_classes, cfg.loss, sampling=cfg.data.get("spacing"),
                               shape_prior=shape_prior)
    backbone_lr_scale = float(cfg.train.get("backbone_lr_scale", 1.0))
    params = _optimizer_groups(model, ct_adapters, float(cfg.train.lr), backbone_lr_scale)
    opt = _build_optimizer(cfg, params)
    if is_main() and backbone_lr_scale != 1.0:
        print(f"[student] backbone_lr={cfg.train.lr * backbone_lr_scale:g} head_lr={cfg.train.lr:g}")
    scheduler_epochs = int(cfg.train.get("scheduler_total_epochs", cfg.train.epochs))
    sched = torch.optim.lr_scheduler.PolynomialLR(opt, total_iters=scheduler_epochs, power=0.9)
    amp_dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) \
        if device.type == "cuda" else torch.float32
    use_scaler = cfg.amp and amp_dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    if is_main():
        print(f"[student] autocast={'off' if not cfg.amp else amp_dtype} grad_scaler={'on' if use_scaler else 'off'}")

    best = -1.0
    no_improve = 0
    patience = int(cfg.train.get("early_stop_patience", 0))
    min_stop_epoch = int(cfg.train.get("early_stop_min_epoch", 0))
    sel_min_epoch = int(cfg.train.get("ckpt_select_min_epoch", 20))
    smooth_k = max(1, int(cfg.train.get("ckpt_smooth_k", 3)))
    keep_topk = max(1, int(cfg.train.get("ckpt_keep_topk", 5)))
    val_start_epoch = int(cfg.logging.get("val_start_epoch", 0))
    val_scores: list[float] = []
    topk: list[tuple] = []
    start_epoch = 0
    distill_every = cfg.train.get("distill_every", 1)
    successful_optimizer_steps = 0
    skipped_optimizer_steps = 0
    exact_resume = bool(cfg.train.get("exact_resume", False))
    step = 0

    if resume_path is not None and Path(resume_path).exists():
        ck = load_checkpoint(resume_path, map_location=device)
        if not _state_finite(ck["model"]):
            if exact_resume:
                raise ValueError(f"{resume_path} contains non-finite weights")
            if is_main():
                print(f"[student] {resume_path} holds non-finite weights - IGNORING, starting fresh")
        else:
            if exact_resume:
                required = {
                    "schema_version",
                    "model",
                    "opt",
                    "sched",
                    "scaler",
                    "adapters",
                    "epoch",
                    "best",
                    "cfg",
                    "val_scores",
                    "topk",
                    "no_improve",
                    "global_step",
                    "successful_optimizer_steps",
                    "skipped_optimizer_steps",
                    "rng_state",
                    "sampler_state",
                    "terminal_complete",
                }
                missing = sorted(required.difference(ck))
                if missing:
                    raise ValueError(f"exact resume state is incomplete: {missing}")
                if ck["schema_version"] != "task1-train-resume-v2":
                    raise ValueError("exact resume schema mismatch")
                if ck["cfg"] != cfg_to_plain(cfg):
                    raise ValueError("exact resume config mismatch")
                if ck["terminal_complete"]:
                    raise ValueError("refusing to resume a completed training run")
            unwrap(model).load_state_dict(ck["model"])
            opt.load_state_dict(ck["opt"])
            sched.load_state_dict(ck["sched"])
            if ck.get("scaler"):
                scaler.load_state_dict(ck["scaler"])
            for nm, a in ct_adapters.items():
                if nm in ck.get("adapters", {}):
                    a.load_state_dict(ck["adapters"][nm])
            start_epoch = int(ck.get("epoch", -1)) + 1
            best = float(ck.get("best", -1.0))
            val_scores = [float(value) for value in ck.get("val_scores", [])]
            topk = [
                (
                    float(entry["smoothed"]),
                    int(entry["epoch"]),
                    Path(entry["path"]),
                )
                for entry in ck.get("topk", [])
            ]
            no_improve = int(ck.get("no_improve", 0))
            reconstructed = _selection_state_from_history(
                run_dir,
                int(ck.get("epoch", -1)),
                cfg,
                sel_min_epoch,
                smooth_k,
                keep_topk,
            )
            if reconstructed is not None and val_scores != reconstructed[0]:
                val_scores, topk, no_improve, best = reconstructed
                if is_main():
                    print(
                        f"[student] reconstructed resume selection state from "
                        f"{run_dir / 'history.csv'}"
                    )
            attempted_steps = start_epoch * len(train_loader)
            successful_optimizer_steps = int(
                ck.get("successful_optimizer_steps", attempted_steps)
            )
            skipped_optimizer_steps = int(ck.get("skipped_optimizer_steps", 0))
            step = int(
                ck.get(
                    "global_step",
                    successful_optimizer_steps + skipped_optimizer_steps,
                )
            )
            if exact_resume:
                if step != attempted_steps:
                    raise ValueError("exact resume global step mismatch")
                if successful_optimizer_steps != attempted_steps:
                    raise ValueError("exact resume optimizer update mismatch")
                if skipped_optimizer_steps != 0:
                    raise ValueError("exact resume contains skipped updates")
                if any(not path.is_file() for _, _, path in topk):
                    raise ValueError("exact resume top-k checkpoint is missing")
                _restore_sampler_state(train_sampler, ck["sampler_state"])
                if int(ck["sampler_state"]["epoch"]) != int(ck["epoch"]):
                    raise ValueError("exact resume sampler epoch mismatch")
                _restore_rng_state(ck["rng_state"])
            if is_main():
                print(f"[student] RESUMED from {resume_path} -> start epoch {start_epoch}, best={best:.4f}")
    elif args.resume:
        if exact_resume:
            raise FileNotFoundError(resume_path)
        if is_main():
            print(f"[student] --resume given but no checkpoint at {resume_path} - starting fresh")
    barrier()

    if step == 0:
        step = successful_optimizer_steps + skipped_optimizer_steps
    if is_main():
        hist = open(run / "history.csv", "a" if start_epoch else "w", newline="")
        writer = csv.writer(hist)
        if start_epoch == 0:
            writer.writerow(["epoch", "train_loss", "dice", "hd95", "asd", "score",
                             "train_dice", "train_score", "gap"])

    plain_cfg = cfg_to_plain(cfg)

    def resume_payload(epoch: int, terminal_complete: bool) -> dict:
        return {
            "schema_version": "task1-train-resume-v2",
            "model": unwrap(model).state_dict(),
            "opt": opt.state_dict(),
            "sched": sched.state_dict(),
            "scaler": scaler.state_dict(),
            "adapters": {
                name: adapter.state_dict()
                for name, adapter in ct_adapters.items()
            },
            "epoch": int(epoch),
            "best": float(best),
            "cfg": plain_cfg,
            "val_scores": list(val_scores),
            "topk": [
                {
                    "smoothed": float(score),
                    "epoch": int(saved_epoch),
                    "path": str(path),
                }
                for score, saved_epoch, path in topk
            ],
            "no_improve": int(no_improve),
            "global_step": int(step),
            "successful_optimizer_steps": int(successful_optimizer_steps),
            "skipped_optimizer_steps": int(skipped_optimizer_steps),
            "rng_state": _rng_state(),
            "sampler_state": _sampler_state(train_sampler),
            "terminal_complete": bool(terminal_complete),
        }

    last_completed_epoch = start_epoch - 1
    early_stopped = False
    for epoch in range(start_epoch, cfg.train.epochs):
        model.train()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        running = 0.0
        n_ok = n_bad = 0
        loader_steps = 0
        for batch in train_loader:
            img = batch["image"].to(device)
            lbl = batch["label"].squeeze(1).long().to(device)
            dm = batch.get("dist_maps")
            dm = dm.to(device) if dm is not None else None
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype,
                                enabled=cfg.amp and device.type == "cuda"):
                out = model(img)
                loss, logs = loss_fn(out, lbl, epoch=epoch, dist_maps=dm)
                if ct_teachers and step % distill_every == 0:
                    s_feat = unwrap(model)._distill_feat
                    if s_feat is not None:
                        w = cfg.train.get("distill_weight", 0.5)
                        for name, t, _ in ct_teachers:
                            loss = loss + w * ct_adapters[name](s_feat, t(img))
                if fcon_on:
                    feat_c = unwrap(model)._fcon_feat
                    if feat_c is not None:
                        _ = model(tensor_fbpaug(img, fcon_alpha, fcon_beta))
                        feat_a = unwrap(model)._fcon_feat
                        w = fcon_w * min(1.0, epoch / max(fcon_ramp, 1))
                        fcon_loss = F.mse_loss(feat_c.float(), feat_a.float())
                        loss = loss + w * fcon_loss
                        logs["fcon"] = float(fcon_loss.detach())
            scaler.scale(loss).backward()
            parameter_groups = [unwrap(model).parameters()]
            if ct_adapters:
                parameter_groups.append(
                    p for adapter in ct_adapters.values() for p in adapter.parameters()
                )
            did_step = _guarded_optimizer_step(
                loss,
                opt,
                scaler,
                use_scaler,
                parameter_groups,
                float(cfg.train.get("grad_clip", 12.0)),
                device,
            )
            if did_step:
                running += logs["total"]
                n_ok += 1
                successful_optimizer_steps += 1
            else:
                n_bad += 1
                skipped_optimizer_steps += 1
            step += 1
            loader_steps += 1
            if exact_steps and not did_step:
                raise RuntimeError(
                    f"exact-step optimizer update failed at epoch {epoch}, "
                    f"batch {loader_steps - 1}"
                )
        if exact_steps and loader_steps != fixed_steps:
            raise RuntimeError(
                f"exact-step epoch mismatch at epoch {epoch}: "
                f"expected={fixed_steps}, observed={loader_steps}"
            )
        sched.step()
        running = reduce_mean(running / max(n_ok, 1), device)

        stop = torch.zeros(1, device=device)
        validation_error = None
        if (
            epoch >= val_start_epoch
            and epoch % cfg.logging.val_every_n_epochs == 0
            and is_main()
        ):
            try:
                m = validate(unwrap(model), val_loader, cfg, device)
                tm = validate(unwrap(model), train_eval_loader, cfg, device) if train_eval_loader else None
                tr_dice = tm["dice"] if tm else float("nan")
                tr_score = tm["score"] if tm else float("nan")
                gap = (tr_score - m["score"]) if tm else float("nan")
                writer.writerow([epoch, running, m["dice"], m["hd95"], m["asd"], m["score"],
                                 tr_dice, tr_score, gap])
                hist.flush()
                skip = f" skipped={n_bad}" if n_bad else ""
                gapstr = f" train_dice={tr_dice:.4f} gap={gap:+.4f}" if tm else ""
                print(f"[e{epoch}] loss={running:.4f} dice={m['dice']:.4f} "
                      f"hd95={m['hd95']:.3f} asd={m['asd']:.3f} score={m['score']:.4f}{gapstr}{skip}")
                sel_val = float(m["dice"]) if cfg.train.get("select_metric", "score") == "dice" else float(m["score"])
                val_scores.append(sel_val)
                smoothed = float(np.mean(val_scores[-smooth_k:]))
                ckpt_dir = run / "checkpoints"
                step_metadata = _step_metadata(
                    loader_steps,
                    successful_optimizer_steps,
                    skipped_optimizer_steps,
                    sched,
                )
                if epoch >= sel_min_epoch:
                    ep_path = ckpt_dir / f"ckpt_e{epoch}.pt"
                    _atomic_torch_save(
                        {
                            "model": unwrap(model).state_dict(),
                            "cfg": plain_cfg,
                            "score": float(m["score"]),
                            "smoothed": smoothed,
                            "epoch": epoch,
                            **step_metadata,
                        },
                        ep_path,
                    )
                    topk.append((smoothed, epoch, ep_path))
                    topk.sort(key=lambda t: t[0], reverse=True)
                    for _, _, p in topk[keep_topk:]:
                        Path(p).unlink(missing_ok=True)
                    del topk[keep_topk:]
                    if smoothed > best:
                        best = smoothed
                        no_improve = 0
                        _atomic_copy(topk[0][2], ckpt_dir / "best.pt")
                    else:
                        no_improve += 1
                if patience > 0 and epoch >= min_stop_epoch and no_improve >= patience:
                    print(f"[student] early stop: no val-score gain in {no_improve} validations "
                          f"(best={best:.4f}) - stopping at epoch {epoch}")
                    stop[0] = 1.0
            except Exception as error:
                import traceback
                traceback.print_exc()
                validation_error = (
                    f"validation failed at epoch {epoch}: "
                    f"{type(error).__name__}: {error}"
                )
                stop[0] = 2.0
        if is_main() and validation_error is None:
            _atomic_torch_save(
                resume_payload(epoch, terminal_complete=False),
                run / "checkpoints" / "last.pt",
            )
        if is_dist():
            tdist.broadcast(stop, src=0)
        if stop.item() == 2:
            raise RuntimeError(
                validation_error or f"validation failed on rank 0 at epoch {epoch}"
            )
        last_completed_epoch = epoch
        if stop.item() == 1:
            early_stopped = True
            break
    if is_main():
        hist.close()
    if exact_steps:
        expected_updates = int(cfg.train.epochs) * fixed_steps
        if early_stopped or last_completed_epoch != int(cfg.train.epochs) - 1:
            raise RuntimeError(
                f"exact-step training ended at epoch {last_completed_epoch}, "
                f"expected {int(cfg.train.epochs) - 1}"
            )
        if step != expected_updates:
            raise RuntimeError(
                f"exact-step attempt count mismatch: {step} != {expected_updates}"
            )
        if successful_optimizer_steps != expected_updates:
            raise RuntimeError(
                "exact-step successful optimizer update count mismatch: "
                f"{successful_optimizer_steps} != {expected_updates}"
            )
        if skipped_optimizer_steps != 0:
            raise RuntimeError("exact-step training recorded skipped updates")
        if is_main():
            checkpoint_dir = run / "checkpoints"
            best_path = checkpoint_dir / "best.pt"
            if not best_path.is_file():
                raise RuntimeError("exact-step training did not produce best.pt")
            last_path = checkpoint_dir / "last.pt"
            _atomic_torch_save(
                resume_payload(last_completed_epoch, terminal_complete=True),
                last_path,
            )
            selection_paths = sorted(checkpoint_dir.glob("ckpt_e*.pt"))
            selection_paths.append(best_path)
            completion = {
                "schema_version": "task1-training-completion-v1",
                "status": "complete",
                "configured_epochs": int(cfg.train.epochs),
                "final_epoch": int(last_completed_epoch),
                "steps_per_epoch": int(fixed_steps),
                "expected_optimizer_updates": int(expected_updates),
                "attempted_steps": int(step),
                "successful_optimizer_steps": int(successful_optimizer_steps),
                "skipped_optimizer_steps": int(skipped_optimizer_steps),
                "scheduler_last_epoch": int(sched.last_epoch),
                "config": {
                    "path": str((run / "config.yaml").resolve()),
                    "sha256": _file_sha256(run / "config.yaml"),
                },
                "last_checkpoint": {
                    "path": str(last_path.resolve()),
                    "sha256": _file_sha256(last_path),
                },
                "selection_checkpoints": [
                    {
                        "path": str(path.resolve()),
                        "sha256": _file_sha256(path),
                    }
                    for path in selection_paths
                ],
            }
            completion_path = run / "training_complete.json"
            _atomic_json(completion, completion_path)
            os.chmod(completion_path, 0o444)
    if is_main():
        print(f"best composite score: {best:.4f}  (ckpt: {run/'checkpoints'/'best.pt'})")
    cleanup()


if __name__ == "__main__":
    main()
