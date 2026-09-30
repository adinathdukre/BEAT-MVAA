from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from common.ema import ModelEMA
from common.io import cfg_to_plain, load_checkpoint, load_config, setup_run_dir
from common.log import report_load
from common.losses import (BoundaryLoss, CBDiceLoss, DiceCELoss, EdgeConsistencyLoss, FarFPLoss,
                           FocalLoss, MbLSLoss, linear_rampup, sigmoid_rampup)
from common.metrics import aggregate, evaluate_case
from common.postprocess import keep_largest_per_class
from common.transforms import rand_conv, strong_augment_2d
from common.uncertainty import reliable_keep_torch
from task3.dataset import (T3FrameDataset, exclude_frames, list_labeled_frames,
                           list_unlabeled_videos,
                           presence_sample_weights, split_by_video)
from task3.model import build_model
from task3.predict import probability_to_mask


class UnlabeledFrames(Dataset):
    def __init__(self, frame_paths, img_size):
        self.frames = frame_paths
        self.img_size = img_size

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, i):
        import cv2

        from common.io import read_png

        H, W = self.img_size
        img = cv2.resize(read_png(self.frames[i], gray=False), (W, H))
        weak = torch.from_numpy(img.transpose(2, 0, 1).copy()).float() / 255.0
        strong = torch.from_numpy(strong_augment_2d(img).transpose(2, 0, 1).copy()).float() / 255.0
        return {"weak": weak, "strong": strong}


@torch.inference_mode()
def validate(model, loader, cfg, device):
    model.eval()
    cases = []
    positive_cases = []
    presence = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
    for batch in loader:
        out = model(batch["image"].to(device))
        if isinstance(out, dict) and "prob" in out:
            prob = out["prob"]
        else:
            seg = out["seg"] if isinstance(out, dict) else out
            prob = torch.softmax(seg, 1)
        gt = batch["label"].numpy()
        for b in range(prob.shape[0]):
            native_hw = tuple(int(x) for x in gt[b].shape[-2:])
            threshold = float(cfg.predict.get("threshold", 0.5))
            p = probability_to_mask(
                prob[b, 1].cpu().numpy(), native_hw, threshold,
                bool(cfg.predict.get("resize_probability", False)))
            if cfg.predict.get("lcc", False):
                p = keep_largest_per_class(p, cfg.num_classes)
            pred_present = bool(p.any())
            gt_present = bool(gt[b].any())
            presence["tp" if pred_present and gt_present else
                     "tn" if not pred_present and not gt_present else
                     "fp" if pred_present else "fn"] += 1
            case = evaluate_case(p, gt[b], cfg.num_classes,
                                 hd_ref=cfg.metric.hd_ref, asd_ref=cfg.metric.asd_ref)
            cases.append(case)
            if gt_present:
                positive_cases.append(case)
    metrics = aggregate(cases)
    positive = aggregate(positive_cases)
    metrics.update({f"positive_{key}": positive[key] for key in ("dice", "hd95", "asd")})
    metrics["positive_n"] = positive["n"]
    metrics.update(presence)
    return metrics


@torch.inference_mode()
def validate_legacy_parity(model, loader, cfg, device):
    model.eval()
    cases = []
    positive_cases = []
    presence = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
    for batch in loader:
        out = model(batch["image"].to(device))
        seg = out["seg"] if isinstance(out, dict) else out
        pred = torch.argmax(seg, dim=1).cpu().numpy()
        gt = batch["label"].numpy()
        for b in range(pred.shape[0]):
            p = keep_largest_per_class(pred[b], cfg.num_classes)
            pred_present = bool(p.any())
            gt_present = bool(gt[b].any())
            presence["tp" if pred_present and gt_present else
                     "tn" if not pred_present and not gt_present else
                     "fp" if pred_present else "fn"] += 1
            case = evaluate_case(
                p, gt[b], cfg.num_classes, hd_ref=cfg.metric.hd_ref, asd_ref=cfg.metric.asd_ref
            )
            cases.append(case)
            if gt_present:
                positive_cases.append(case)
    metrics = aggregate(cases)
    positive = aggregate(positive_cases)
    metrics.update({f"positive_{key}": positive[key] for key in ("dice", "hd95", "asd")})
    metrics["positive_n"] = positive["n"]
    metrics.update(presence)
    return metrics


def _recalibrate_bn(model, loader, device, max_batches=200):
    import torch.nn as nn
    model.eval()
    bns = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    if not bns:
        return
    for m in bns:
        m.reset_running_stats(); m.momentum = None; m.train()
    with torch.inference_mode():
        for i, batch in enumerate(loader):
            model(batch["image"].to(device))
            if i + 1 >= max_batches:
                break
    for m in bns:
        m.eval()


def load_initial_weights(model, checkpoint_path, device):
    resolved = str(Path(checkpoint_path).expanduser().resolve())
    state = load_checkpoint(resolved, map_location=device)
    if not isinstance(state, dict) or "model" not in state:
        raise ValueError(f"{resolved}: expected a checkpoint containing model weights")
    report_load(model, state["model"], name="task3.init", require=0.95)
    return resolved


def _build_validation_loader(val_items, cfg, legacy_parity_validation, validation_disabled):
    if validation_disabled:
        return None
    if not val_items:
        return None
    val_ds = T3FrameDataset(
        val_items, cfg, train=False, native_label=not legacy_parity_validation
    )
    batch_size = cfg.train.batch_size if legacy_parity_validation else 1
    return DataLoader(val_ds, batch_size=batch_size, num_workers=2)


def _guarded_optimizer_step(loss, optimizer, scaler, use_scaler, parameters, max_norm):
    parameters = tuple(parameters)
    if use_scaler:
        scaler.unscale_(optimizer)
    finite = bool(torch.isfinite(loss.detach()).all().item())
    if finite and max_norm > 0:
        total_norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm=max_norm)
        finite = bool(torch.isfinite(total_norm).item())
    elif finite:
        finite = all(
            parameter.grad is None
            or bool(torch.isfinite(parameter.grad.detach()).all().item())
            for parameter in parameters
        )
    if finite:
        scaler.step(optimizer)
        scaler.update()
        return True
    if use_scaler:
        scaler.update(
            new_scale=(
                float(scaler.get_scale())
                * float(scaler.get_backoff_factor())
            )
        )
    else:
        scaler.update()
    optimizer.zero_grad(set_to_none=True)
    return False


def _save_endpoint_checkpoints(
    model,
    checkpoint_dir,
    epoch,
    checkpoint_meta,
    validation_disabled,
    write_unvalidated_best,
):
    payload = {
        "model": model.state_dict(),
        "epoch": epoch,
        **checkpoint_meta,
    }
    if validation_disabled:
        payload["metrics"] = None
    torch.save(payload, checkpoint_dir / "last.pt")
    if write_unvalidated_best:
        torch.save(payload, checkpoint_dir / "best.pt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="student")
    ap.add_argument("--init", default=None)
    ap.add_argument("--stop-after-epochs", type=int, default=None)
    ap.add_argument("--legacy-parity-validation", action="store_true")
    ap.add_argument("--disable-validation", action="store_true")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    if args.disable_validation and args.legacy_parity_validation:
        raise ValueError(
            "--disable-validation and --legacy-parity-validation are mutually exclusive"
        )

    cfg = load_config("task3", args.overrides)
    run_epochs = int(cfg.train.epochs)
    if args.stop_after_epochs is not None:
        run_epochs = int(args.stop_after_epochs)
        if not 1 <= run_epochs <= int(cfg.train.epochs):
            raise ValueError("--stop-after-epochs must be between 1 and train.epochs")
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run = setup_run_dir(cfg.paths.runs_dir, "task3", args.exp, cfg)

    tl = cfg.get("target_label", 10)
    labeled = list_labeled_frames(cfg.paths.t3_root, "train", tl)
    before = len(labeled)
    labeled = exclude_frames(labeled, cfg.data.get("exclude_frames", []))
    print(f"[data] labeled={len(labeled)} excluded={before - len(labeled)}")
    if cfg.data.get("train_all_labeled", False):
        train_items, val_items = labeled, []
    else:
        train_items, val_items = split_by_video(
            labeled, val_frac=cfg.data.get("val_frac", 0.2),
            seed=cfg.seed, holdout=cfg.data.get("holdout_video"))

    train_ds = T3FrameDataset(train_items, cfg, train=True)
    balance_target = cfg.train.get("target_positive_fraction")
    sampler = None
    if balance_target is not None:
        sample_weights = presence_sample_weights(train_items, float(balance_target), tl)
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        print(f"[balance] target_positive_fraction={float(balance_target):.3f}")
    train_loader = DataLoader(train_ds, batch_size=cfg.train.batch_size, shuffle=sampler is None,
                              sampler=sampler,
                              num_workers=cfg.train.num_workers, drop_last=True)
    val_loader = _build_validation_loader(
        val_items,
        cfg,
        args.legacy_parity_validation,
        args.disable_validation,
    )
    if args.disable_validation:
        print(
            f"[validation] disabled held_frames={len(val_items)} "
            "held_mask_reads=0"
        )

    unl_frames = [f for fr in list_unlabeled_videos(cfg.paths.images_dir).values() for f in fr]
    unl_loader = None
    if cfg.train.get("unimatch_v2", True) and unl_frames:
        unl_loader = DataLoader(UnlabeledFrames(unl_frames, cfg.data.img_size),
                                batch_size=cfg.train.batch_size, shuffle=True,
                                num_workers=cfg.train.num_workers, drop_last=True)

    model = build_model(cfg).to(device)
    init_checkpoint = None
    if args.init:
        init_checkpoint = load_initial_weights(model, args.init, device)
        print(f"[init] {init_checkpoint}")
    mt_cfg = cfg.train.get("ema_teacher", {}) or {}
    mt_on = bool(mt_cfg.get("enabled", False))
    mean_teacher = ModelEMA(model, decay=float(mt_cfg.get("decay", 0.999))) if mt_on else None
    validate_teacher = mt_on and bool(mt_cfg.get("use_for_validation", True))
    if mt_on:
        print(f"[ema-teacher] decay={float(mt_cfg.get('decay', 0.999))} "
              f"validate_teacher={validate_teacher}")
    dice, focal = DiceCELoss(cfg.num_classes), FocalLoss()
    boundary, edge = BoundaryLoss(cfg.num_classes), EdgeConsistencyLoss(cfg.num_classes)
    cbdice = CBDiceLoss()
    farfp = FarFPLoss(cfg.num_classes, power=float(cfg.loss.get("farfp_power", 2.0)),
                      norm=float(cfg.loss.get("farfp_norm", 64.0)))
    mbls = MbLSLoss(margin=float(cfg.loss.get("mbls_margin", 10.0)))
    from monai.losses import LogHausdorffDTLoss, TverskyLoss
    hddt = LogHausdorffDTLoss(include_background=False, to_onehot_y=True, softmax=True)
    tversky = TverskyLoss(include_background=False, to_onehot_y=True, softmax=True,
                          alpha=float(cfg.loss.get("tversky_alpha", 0.3)),
                          beta=float(cfg.loss.get("tversky_beta", 0.7)))
    shape_consistency = None
    if float(cfg.loss.get("shape_prior", 0)) > 0:
        from common.shape_prior import ShapeConsistencyLoss, load_shape_prior
        _sp = load_shape_prior(cfg.predict.get("shape_prior", {}).get("ckpt"), dims=2, device=device)
        if _sp is not None:
            shape_consistency = ShapeConsistencyLoss(_sp)

    params = list(model.parameters())

    opt = torch.optim.AdamW(params, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    scheduler_epochs = int(cfg.train.get("scheduler_epochs") or cfg.train.epochs)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=scheduler_epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp)

    hist = open(run / "history.csv", "w", newline="")
    writer = csv.writer(hist)
    writer.writerow([
        "epoch", "loss", "dice", "hd95", "asd",
        "positive_dice", "positive_hd95", "positive_asd", "score"])
    selection_metric = str(cfg.train.get("selection_metric", "dice"))
    raw_metrics = {
        "dice": -float("inf"),
        "hd95": float("inf"),
        "asd": float("inf"),
        "positive_dice": -float("inf"),
        "positive_hd95": float("inf"),
        "positive_asd": float("inf"),
    }
    if selection_metric not in raw_metrics:
        raise ValueError(f"selection_metric must be one of {sorted(raw_metrics)}")
    best = raw_metrics[selection_metric]
    raw_best = dict(raw_metrics)
    safe_best = dict(raw_metrics)
    presence_best = (float("inf"), -float("inf"))
    checkpoint_meta = {
        "config": cfg_to_plain(cfg),
        "selection_metric": selection_metric,
        "init_checkpoint": init_checkpoint,
        "executed_epochs": run_epochs,
        "stop_after_epochs": args.stop_after_epochs,
        "legacy_parity_validation": args.legacy_parity_validation,
        "validation_disabled": args.disable_validation,
        "validation_loader_constructed": val_loader is not None,
        "validation_loader_absent": val_loader is None,
        "held_frames": len(val_items),
        "held_mask_reads": 0 if args.disable_validation else None,
        "validation_metrics_used_for_selection": (
            val_loader is not None and not args.disable_validation
        ),
        "checkpoint_selection": (
            "none" if args.disable_validation else selection_metric
        ),
        "best_checkpoint_written": not args.disable_validation,
    }
    step = 0
    optimizer_steps = 0
    skipped_nonfinite_steps = 0

    swad_cfg = cfg.train.get("swad", {}) or {}
    swad_on = bool(swad_cfg.get("enabled", False))
    swad_start = int(cfg.train.epochs * float(swad_cfg.get("start_frac", 0.6)))
    swad_sum, swad_n, swad_end = None, 0, None
    rc_cfg = cfg.train.get("randconv", {}) or {}
    rc_on = bool(rc_cfg.get("enabled", False))
    rc_p, rc_ks = float(rc_cfg.get("p", 0.5)), list(rc_cfg.get("kernel_sizes", [1, 3, 5, 7]))
    rc_sup, rc_cons = float(rc_cfg.get("sup_weight", 1.0)), float(rc_cfg.get("cons_weight", 1.0))
    if swad_on:
        print(f"[swad] tail weight-averaging from epoch {swad_start}/{cfg.train.epochs}")
    if rc_on:
        print(f"[randconv] p={rc_p} kernels={rc_ks} sup_w={rc_sup} cons_w={rc_cons}")
    gin_cfg = cfg.train.get("gin", {}) or {}
    gin_on = bool(gin_cfg.get("enabled", False))
    gin_sup, gin_cons = float(gin_cfg.get("sup_weight", 1.0)), float(gin_cfg.get("cons_weight", 1.0))
    gin = None
    if gin_on:
        from common.gin import GINGroupConv
        gin = GINGroupConv().to(device)
        print(f"[gin] sup_w={gin_sup} cons_w={gin_cons}")
    fm_cfg = cfg.train.get("freematch", {}) or {}
    fm_on = bool(fm_cfg.get("enabled", False))
    fm_m, fm_floor = float(fm_cfg.get("momentum", 0.999)), float(fm_cfg.get("floor", 0.5))
    fm_tau = torch.tensor(float(cfg.train.get("conf_threshold", 0.95)), device=device)
    if fm_on:
        print(f"[freematch] adaptive threshold: momentum={fm_m} floor={fm_floor}")

    for epoch in range(run_epochs):
        model.train()
        running = 0.0
        unl_iter = iter(unl_loader) if unl_loader else None
        cons_w = sigmoid_rampup(epoch, max(cfg.train.epochs // 5, 1))
        b_ramp = linear_rampup(epoch, cfg.loss.boundary.get("rampup_epochs", 20))
        for batch in train_loader:
            img = batch["image"].to(device)
            lbl = batch["label"].to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=cfg.amp and device.type == "cuda"):
                out = model(img)
                seg = out["seg"] if isinstance(out, dict) else out
                loss = (cfg.loss.dice * dice(seg, lbl) + cfg.loss.focal * focal(seg, lbl)
                        + cfg.loss.boundary.weight * b_ramp * boundary(seg, lbl))
                if cfg.loss.get("cbdice", 0) > 0:
                    loss = loss + cfg.loss.cbdice * cbdice(seg, lbl)
                if cfg.loss.get("edge_consistency", 0) > 0:
                    loss = loss + cfg.loss.edge_consistency * edge(seg, lbl)
                if cfg.loss.get("hd_dt", 0) > 0:
                    loss = loss + cfg.loss.hd_dt * b_ramp * hddt(seg, lbl.unsqueeze(1))
                if cfg.loss.get("tversky", 0) > 0:
                    loss = loss + cfg.loss.tversky * tversky(seg, lbl.unsqueeze(1))
                if cfg.loss.get("mbls", 0) > 0:
                    loss = loss + cfg.loss.mbls * mbls(seg)
                if cfg.loss.get("farfp", 0) > 0:
                    loss = loss + cfg.loss.farfp * b_ramp * farfp(seg, lbl)
                if shape_consistency is not None and cfg.loss.get("shape_prior", 0) > 0:
                    loss = loss + cfg.loss.shape_prior * shape_consistency(seg)

                if rc_on:
                    seg_rc = model(rand_conv(img, p=rc_p, kernel_sizes=rc_ks))
                    seg_rc = seg_rc["seg"] if isinstance(seg_rc, dict) else seg_rc
                    loss = loss + rc_sup * (cfg.loss.dice * dice(seg_rc, lbl)
                                            + cfg.loss.focal * focal(seg_rc, lbl))
                    _cons = F.kl_div(F.log_softmax(seg_rc, 1), F.softmax(seg.detach(), 1),
                                     reduction="none").sum(1).mean()
                    loss = loss + rc_cons * _cons

                if gin_on:
                    seg_gin = model(gin(img))
                    seg_gin = seg_gin["seg"] if isinstance(seg_gin, dict) else seg_gin
                    loss = loss + gin_sup * (cfg.loss.dice * dice(seg_gin, lbl)
                                             + cfg.loss.focal * focal(seg_gin, lbl))
                    _cons_g = F.kl_div(F.log_softmax(seg_gin, 1), F.softmax(seg.detach(), 1),
                                       reduction="none").sum(1).mean()
                    loss = loss + gin_cons * _cons_g

                if unl_iter is not None:
                    try:
                        u = next(unl_iter)
                    except StopIteration:
                        unl_iter = iter(unl_loader)
                        u = next(unl_iter)
                    weak, strong = u["weak"].to(device), u["strong"].to(device)
                    with torch.no_grad():
                        w_out = mean_teacher(weak) if mean_teacher is not None else model(weak)
                        w_seg = w_out["seg"] if isinstance(w_out, dict) else w_out
                        conf, pseudo = torch.softmax(w_seg, 1).max(1)
                        if fm_on:
                            fm_tau.mul_(fm_m).add_((1.0 - fm_m) * conf.mean())
                            _thr = max(float(fm_tau), fm_floor)
                        else:
                            _thr = cfg.train.get("conf_threshold", 0.95)
                        mask = reliable_keep_torch(conf, pseudo, _thr,
                                                   cfg.train.get("boundary_conf_threshold", 0.95),
                                                   int(cfg.train.get("boundary_band", 0))).float()
                    s_out = model(strong)
                    s_seg = s_out["seg"] if isinstance(s_out, dict) else s_out
                    ce = F.cross_entropy(s_seg, pseudo, reduction="none")
                    loss = loss + cons_w * (ce * mask).sum() / mask.sum().clamp(min=1.0)
            gclip = float(cfg.train.get("grad_clip", 12.0))
            scaler.scale(loss).backward()
            did_step = _guarded_optimizer_step(
                loss,
                opt,
                scaler,
                bool(cfg.amp and device.type == "cuda"),
                params,
                gclip,
            )
            if did_step:
                optimizer_steps += 1
                if mean_teacher is not None:
                    mean_teacher.update(model, step=step)
                running += float(loss.detach())
            else:
                skipped_nonfinite_steps += 1
            step += 1
        sched.step()

        if val_loader is not None:
            eval_model = mean_teacher.ema if validate_teacher else model
            validator = validate_legacy_parity if args.legacy_parity_validation else validate
            m = validator(eval_model, val_loader, cfg, device)
            writer.writerow([
                epoch, running / max(len(train_loader), 1),
                m["dice"], m["hd95"], m["asd"],
                m["positive_dice"], m["positive_hd95"], m["positive_asd"], m["score"]])
            hist.flush()
            print(f"[e{epoch}] loss={running/max(len(train_loader),1):.4f} dice={m['dice']:.4f} "
                  f"hd={m['hd95']:.3f} asd={m['asd']:.3f} score={m['score']:.4f} "
                  f"positive={m['positive_dice']:.4f}/{m['positive_hd95']:.3f}/"
                  f"{m['positive_asd']:.3f} presence={m['tp']}/{m['tn']}/{m['fp']}/{m['fn']}")
            for metric_name in raw_best:
                improved = (m[metric_name] > raw_best[metric_name] if metric_name.endswith("dice")
                            else m[metric_name] < raw_best[metric_name])
                if improved:
                    raw_best[metric_name] = m[metric_name]
                    filename = "best_hd.pt" if metric_name == "hd95" else f"best_{metric_name}.pt"
                    torch.save({"model": eval_model.state_dict(), "metrics": m, "epoch": epoch,
                                **checkpoint_meta},
                               run / "checkpoints" / filename)
            presence_rank = (m["fp"] + m["fn"], -m["dice"])
            if presence_rank < presence_best:
                presence_best = presence_rank
                torch.save({"model": eval_model.state_dict(), "metrics": m, "epoch": epoch,
                            **checkpoint_meta},
                           run / "checkpoints" / "best_presence.pt")
            if m["fp"] == 0 and m["fn"] == 0:
                for metric_name in safe_best:
                    improved = (m[metric_name] > safe_best[metric_name]
                                if metric_name.endswith("dice")
                                else m[metric_name] < safe_best[metric_name])
                    if improved:
                        safe_best[metric_name] = m[metric_name]
                        filename = ("best_safe_hd.pt" if metric_name == "hd95"
                                    else f"best_safe_{metric_name}.pt")
                        torch.save({"model": eval_model.state_dict(), "metrics": m, "epoch": epoch,
                                    **checkpoint_meta},
                                   run / "checkpoints" / filename)
            selected = (m[selection_metric] > best if selection_metric.endswith("dice")
                        else m[selection_metric] < best)
            if selected:
                best = m[selection_metric]
                torch.save({"model": eval_model.state_dict(), "metrics": m, "epoch": epoch,
                            **checkpoint_meta},
                           run / "checkpoints" / "best.pt")
        else:
            writer.writerow([epoch, running / max(len(train_loader), 1), "", "", "", "", "", "", ""])
            hist.flush()
            print(f"[e{epoch}] loss={running/max(len(train_loader),1):.4f}")
        if swad_on and epoch >= swad_start:
            sd = model.state_dict()
            if swad_sum is None:
                swad_sum = {k: (v.detach().float().clone() if v.is_floating_point()
                                else v.detach().clone()) for k, v in sd.items()}
                swad_n = 1
            else:
                for k, v in sd.items():
                    if v.is_floating_point():
                        swad_sum[k].add_(v.detach().float())
                    else:
                        swad_sum[k] = v.detach().clone()
                swad_n += 1
            swad_end = epoch
    checkpoint_meta.update({
        "optimizer_steps": optimizer_steps,
        "skipped_nonfinite_steps": skipped_nonfinite_steps,
    })
    final_model = mean_teacher.ema if validate_teacher else model
    _save_endpoint_checkpoints(
        final_model,
        run / "checkpoints",
        run_epochs - 1,
        checkpoint_meta,
        args.disable_validation,
        val_loader is None and not args.disable_validation,
    )
    if swad_on and swad_sum is not None:
        ref = model.state_dict()
        avg = {k: ((v / swad_n).to(ref[k].dtype) if v.is_floating_point() else v)
               for k, v in swad_sum.items()}
        model.load_state_dict(avg)
        _recalibrate_bn(model, train_loader, device)
        ms = None
        if val_loader is not None:
            validator = validate_legacy_parity if args.legacy_parity_validation else validate
            ms = validator(model, val_loader, cfg, device)
            print(f"[swad] averaged {swad_n} epochs -> swad.pt  val dice={ms['dice']:.4f} "
                  f"hd={ms['hd95']:.3f} asd={ms['asd']:.3f} score={ms['score']:.4f}")
        torch.save({
            "model": model.state_dict(),
            "metrics": ms,
            "swad_n": swad_n,
            "swad_start_epoch": swad_start,
            "swad_end_epoch": swad_end,
            "bn_recalibration_partition": "held_fold_excluded_training_partition",
            "bn_recalibration_held_frames": False,
            **checkpoint_meta,
        }, run / "checkpoints" / "swad.pt")
        if val_loader is None:
            print(f"[swad] averaged {swad_n} epochs -> swad.pt")
    hist.close()
    print(f"best {selection_metric}: {best:.4f}")


if __name__ == "__main__":
    main()
