from __future__ import annotations

import argparse
import csv
import random
import subprocess
from pathlib import Path

import numpy as np
import torch
import torch.distributed as tdist
from torch.utils.data import DataLoader

from common.dist import (barrier, cleanup, is_dist, is_main, make_loader, reduce_mean,
                         setup_distributed, unwrap, wrap_ddp)
from common.io import cfg_to_plain, load_checkpoint, load_config, nifti_spacing, setup_run_dir
from common.losses import CompositeSegLoss
from common.metrics import aggregate, evaluate_case, q_asd, q_dsc, q_hd
from common.postprocess import postprocess_volume
from common.registry import build_student3d
from common.sliding_window import sliding_window_predict
from task2.dataset import build_datasets


def weighted_select_score(m, w, hd_ref, asd_ref):
    return w["dsc"] * q_dsc(m["dice"]) + w["hd"] * q_hd(m["hd95"], hd_ref) + w["asd"] * q_asd(m["asd"], asd_ref)


@torch.inference_mode()
def validate(model, val_loader, cfg, device):
    model.eval()
    cases = []
    items = val_loader.dataset.data
    for batch, item in zip(val_loader, items):
        img = batch["image"].to(device)
        prob = sliding_window_predict(model, img, cfg.data.patch_size, overlap=cfg.predict.sw_overlap,
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
    ap.add_argument("--exp", default="stunet_b")
    ap.add_argument("--resume", default=None, help="path to last.pt to resume from; 'auto' = this exp's last.pt")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()

    cfg = load_config("task2", args.overrides)
    rank, world, device = setup_distributed()
    random.seed(cfg.seed + rank)
    torch.manual_seed(cfg.seed + rank)
    np.random.seed(cfg.seed + rank)
    run = setup_run_dir(cfg.paths.runs_dir, "task2", args.exp, cfg) if is_main() else None
    try:
        _sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        _sha = "nogit"
    run_dir = Path(cfg.paths.runs_dir) / "task2" / f"{args.exp}_{_sha}"
    resume_path = (run_dir / "checkpoints" / "last.pt") if args.resume == "auto" \
        else (Path(args.resume) if args.resume else None)

    from monai.data import list_data_collate

    dsets = build_datasets(cfg)
    train_ds, val_ds = dsets[0], dsets[1]
    train_eval_ds = dsets[2] if len(dsets) > 2 else None
    train_loader, train_sampler = make_loader(
        train_ds, batch_size=cfg.train.batch_size, shuffle=True, num_workers=cfg.train.num_workers,
        collate_fn=list_data_collate, drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=2)
    train_eval_loader = (DataLoader(train_eval_ds, batch_size=1, num_workers=2)
                         if is_main() and train_eval_ds is not None else None)

    model = build_student3d(cfg).to(device)
    if cfg.train.get("grad_checkpointing", False) and model.enable_grad_checkpointing():
        print("[student] activation checkpointing ON")
    model = wrap_ddp(model, device)
    shape_prior = None
    if float(cfg.loss.get("shape_prior", 0)) > 0:
        from common.shape_prior import load_shape_prior
        shape_prior = load_shape_prior(cfg.predict.get("shape_prior", {}).get("ckpt"), dims=3, device=device)
    loss_fn = CompositeSegLoss(cfg.num_classes, cfg.loss, sampling=cfg.data.get("spacing"),
                               shape_prior=shape_prior)
    opt = torch.optim.SGD(model.parameters(), lr=cfg.train.lr, momentum=cfg.train.momentum,
                          weight_decay=cfg.train.weight_decay, nesterov=True)
    sched = torch.optim.lr_scheduler.PolynomialLR(opt, total_iters=cfg.train.epochs, power=0.9)
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
    start_epoch = 0
    sel_w = dict(cfg.get("ckpt_select", {"dsc": 0.6, "hd": 0.2, "asd": 0.2}))

    if resume_path is not None and Path(resume_path).exists():
        ck = load_checkpoint(resume_path, map_location=device)
        finite = all(not (torch.is_tensor(t) and t.is_floating_point() and not torch.isfinite(t).all())
                     for t in ck["model"].values())
        if not finite:
            if is_main():
                print(f"[student] {resume_path} holds non-finite weights - IGNORING, starting fresh")
        else:
            unwrap(model).load_state_dict(ck["model"])
            opt.load_state_dict(ck["opt"])
            sched.load_state_dict(ck["sched"])
            if ck.get("scaler"):
                scaler.load_state_dict(ck["scaler"])
            start_epoch = int(ck.get("epoch", -1)) + 1
            best = float(ck.get("best", -1.0))
            if is_main():
                print(f"[student] RESUMED from {resume_path} -> start epoch {start_epoch}, best={best:.4f}")
    elif args.resume and is_main():
        print(f"[student] --resume given but no checkpoint at {resume_path} - starting fresh")
    barrier()

    if is_main():
        hist = open(run / "history.csv", "a" if start_epoch else "w", newline="")
        writer = csv.writer(hist)
        if start_epoch == 0:
            writer.writerow(["epoch", "train_loss", "dice", "hd95", "asd", "score", "select",
                             "train_dice", "train_score", "gap"])

    for epoch in range(start_epoch, cfg.train.epochs):
        model.train()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        running = 0.0
        n_ok = n_bad = 0
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
            scaler.scale(loss).backward()
            finite = torch.tensor([1.0 if torch.isfinite(loss) else 0.0], device=device)
            if is_dist():
                tdist.all_reduce(finite, op=tdist.ReduceOp.MIN)
            if finite.item() > 0:
                gclip = float(cfg.train.get("grad_clip", 12.0))
                if gclip > 0:
                    if use_scaler:
                        scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gclip)
                scaler.step(opt)
                scaler.update()
                running += logs["total"]
                n_ok += 1
            else:
                scaler.update()
                n_bad += 1
        sched.step()
        running = reduce_mean(running / max(n_ok, 1), device)

        stop = torch.zeros(1, device=device)
        if epoch % cfg.logging.val_every_n_epochs == 0 and is_main():
            try:
                m = validate(unwrap(model), val_loader, cfg, device)
                sel = weighted_select_score(m, sel_w, cfg.metric.hd_ref, cfg.metric.asd_ref)
                tm = validate(unwrap(model), train_eval_loader, cfg, device) if train_eval_loader else None
                tr_dice = tm["dice"] if tm else float("nan")
                tr_score = tm["score"] if tm else float("nan")
                gap = (tr_score - m["score"]) if tm else float("nan")
                writer.writerow([epoch, running, m["dice"], m["hd95"], m["asd"], m["score"], sel,
                                 tr_dice, tr_score, gap])
                hist.flush()
                skip = f" skipped={n_bad}" if n_bad else ""
                gapstr = f" train_dice={tr_dice:.4f} gap={gap:+.4f}" if tm else ""
                print(f"[e{epoch}] loss={running:.4f} dice={m['dice']:.4f} "
                      f"hd95={m['hd95']:.3f} asd={m['asd']:.3f} score={m['score']:.4f} select={sel:.4f}{gapstr}{skip}")
                if sel > best:
                    best = sel
                    no_improve = 0
                    torch.save({"model": unwrap(model).state_dict(), "select": best, "epoch": epoch},
                               run / "checkpoints" / "best.pt")
                else:
                    no_improve += 1
                torch.save({"model": unwrap(model).state_dict(), "opt": opt.state_dict(),
                            "sched": sched.state_dict(), "scaler": scaler.state_dict(),
                            "epoch": epoch, "best": best, "cfg": cfg_to_plain(cfg)},
                           run / "checkpoints" / "last.pt")
                if patience > 0 and epoch >= min_stop_epoch and no_improve >= patience:
                    print(f"[student] early stop: no select gain in {no_improve} validations "
                          f"(best={best:.4f}) - stopping at epoch {epoch}")
                    stop[0] = 1.0
            except Exception:
                import traceback
                traceback.print_exc()
                print(f"[student] validation failed at epoch {epoch} - stopping cleanly")
                stop[0] = 1.0
        if is_dist():
            tdist.broadcast(stop, src=0)
        if stop.item() > 0:
            break
    if is_main():
        hist.close()
        print(f"best select score: {best:.4f}")
    cleanup()


if __name__ == "__main__":
    main()
