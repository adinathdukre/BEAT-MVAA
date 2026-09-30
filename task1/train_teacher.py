from __future__ import annotations

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F

from common.dist import (barrier, cleanup, is_main, make_loader, setup_distributed, unwrap,
                          wrap_ddp, wrap_fsdp)
from common.distill import FeatureDistill, build_ct_distill_teachers
from common.ema import ModelEMA
from common.io import load_config, nifti_spacing, setup_run_dir
from common.losses import CompositeSegLoss, DiceCELoss, sigmoid_rampup
from common.registry import build_teacher3d
from task1.dataset import build_datasets


def bcp_copy_paste(lab_img, lab_lbl, unl_img, unl_pseudo, mask_ratio=0.6):
    B, _, D, H, W = lab_img.shape
    mask = torch.ones_like(lab_img)
    d, h, w = int(D * mask_ratio), int(H * mask_ratio), int(W * mask_ratio)
    z0, y0, x0 = np.random.randint(0, D - d + 1), np.random.randint(0, H - h + 1), np.random.randint(0, W - w + 1)
    mask[:, :, z0:z0 + d, y0:y0 + h, x0:x0 + w] = 0
    mix_img = lab_img * mask + unl_img * (1 - mask)
    m1 = mask[:, 0].long()
    mix_lbl = lab_lbl * m1 + unl_pseudo * (1 - m1)
    return mix_img, mix_lbl


def abd_displace(lab_img, lab_lbl, unl_img, unl_pseudo, lab_conf, unl_conf, grid=4, frac=0.5):
    B, _, D, H, W = lab_img.shape
    bd, bh, bw = max(D // grid, 1), max(H // grid, 1), max(W // grid, 1)
    pool = lambda c: F.avg_pool3d(c.unsqueeze(1), kernel_size=(bd, bh, bw), stride=(bd, bh, bw))
    diff = pool(unl_conf) - pool(lab_conf)
    flat = diff.flatten(1)
    k = max(int(flat.shape[1] * frac), 1)
    thresh = flat.topk(k, dim=1).values[:, -1].view(B, 1, 1, 1, 1)
    block = (diff >= thresh).float()
    paste = F.interpolate(block, size=(D, H, W), mode="nearest")
    mix_img = lab_img * (1 - paste) + unl_img * paste
    m1 = paste[:, 0].long()
    mix_lbl = lab_lbl * (1 - m1) + unl_pseudo * m1
    return mix_img, mix_lbl


@torch.inference_mode()
def validate_teacher(teacher, val_loader, cfg, device):
    from common.metrics import aggregate, evaluate_case
    from common.sliding_window import sliding_window_predict

    cases = []
    items = val_loader.dataset.data
    for batch, item in zip(val_loader, items):
        x = batch["image"].to(device)
        prob = sliding_window_predict(teacher, x, cfg.data.patch_size,
                                      overlap=cfg.predict.sw_overlap, mode=cfg.predict.sw_mode,
                                      amp=False, flip_axes=[])
        pred = prob.argmax(1)[0].cpu().numpy().astype("uint8")
        gt = batch["label"][0, 0].cpu().numpy().astype("uint8")
        cases.append(evaluate_case(pred, gt, cfg.num_classes, spacing=nifti_spacing(item["image"])))
    return aggregate(cases)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="teacher_admt")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()

    cfg = load_config("task1", args.overrides)
    if cfg.data.get("use_pseudolabels", False):
        cfg.data.use_pseudolabels = False
        if int(os.environ.get("RANK", "0")) == 0:
            print("[teacher] use_pseudolabels forced -> False (teacher is labeled-only)")
    rank, world, device = setup_distributed()
    run = setup_run_dir(cfg.paths.runs_dir, "task1", args.exp, cfg) if is_main() else None

    t_patch = cfg.teacher.get("patch_size", None)
    if t_patch is not None:
        cfg.data.patch_size = list(t_patch)
        if is_main():
            print(f"[teacher] training patch_size -> {cfg.data.patch_size}")

    from monai.data import list_data_collate

    train_ds, val_ds = build_datasets(cfg)
    loader, sampler = make_loader(train_ds, batch_size=cfg.train.batch_size, shuffle=True,
                                  num_workers=cfg.train.num_workers, collate_fn=list_data_collate,
                                  drop_last=True)
    val_loader = None
    if len(val_ds) > 0:
        from torch.utils.data import DataLoader
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                                num_workers=min(2, cfg.train.num_workers), collate_fn=list_data_collate)
    if is_main() and run is not None:
        hist = run / "history.csv"
        if not hist.exists():
            hist.write_text("epoch,sup,cons,dice,hd95,asd,score\n")

    use_fsdp = world > 1

    student = build_teacher3d(cfg)
    use_admt = cfg.teacher.get("admt", True)
    teacher_a = ModelEMA(student, decay=0.99)
    teacher_b = ModelEMA(student, decay=0.999) if use_admt else None
    if not use_fsdp:
        student = student.to(device)

    ct_teachers = build_ct_distill_teachers(cfg) if cfg.teacher.get("distill_ct", True) else []
    ct_adapters = {}
    if ct_teachers:
        ok, s_ch = student.enable_distill_capture()
        if ok:
            for name, t, dim in ct_teachers:
                t.to(device)
                ct_adapters[name] = FeatureDistill(s_ch, dim, dims=3).to(device)
            if is_main():
                print(f"[distill] teacher CT teachers {[n for n,_,_ in ct_teachers]} -> feat ch={s_ch}")
        else:
            ct_teachers = []

    if use_fsdp:
        if cfg.teacher.get("grad_checkpointing", True) and student.enable_grad_checkpointing():
            if is_main():
                print("[teacher] activation checkpointing ON (FSDP)")
        student = wrap_fsdp(student, device)
        teacher_a.fsdp_wrap(device)
        if teacher_b is not None:
            teacher_b.fsdp_wrap(device)
        ct_adapters = {name: wrap_fsdp(a, device) for name, a in ct_adapters.items()}
        scaler = None
        if is_main():
            print(f"[teacher] FSDP FULL_SHARD across {world} GPUs (bf16)")
    else:
        if cfg.teacher.get("grad_checkpointing", True) and student.enable_grad_checkpointing():
            print("[teacher] activation checkpointing ON")
        student = wrap_ddp(student, device)
        scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp)

    from omegaconf import OmegaConf
    _tloss = OmegaConf.to_container(cfg.loss, resolve=True)
    _tloss["cbdice"] = 0.0
    _tloss["edge_consistency"] = 0.0
    sup_loss = CompositeSegLoss(cfg.num_classes, _tloss, sampling=cfg.data.get("spacing"))
    params = list(student.parameters())
    for a in ct_adapters.values():
        params += list(a.parameters())
    opt = torch.optim.SGD(params, lr=cfg.train.lr, momentum=cfg.train.momentum,
                          weight_decay=cfg.train.weight_decay, nesterov=True)

    step = 0
    distill_every = cfg.train.get("distill_every", 1)
    val_every = int(cfg.teacher.get("val_every", 25))
    best_score = -1.0
    sel_min_epoch = int(cfg.train.get("ckpt_select_min_epoch", 0))
    smooth_k = max(1, int(cfg.train.get("ckpt_smooth_k", 3)))
    val_scores: list[float] = []
    for epoch in range(cfg.train.epochs):
        student.train()
        if sampler is not None:
            sampler.set_epoch(epoch)
        cons_w = sigmoid_rampup(epoch, max(cfg.train.epochs // 5, 1))
        for batch in loader:
            img = batch["image"].to(device)
            lbl = batch["label"].squeeze(1).long().to(device)
            dm = batch.get("dist_maps")
            dm = dm.to(device) if dm is not None else None

            teacher = (teacher_a if (step % 2 == 0) else teacher_b) if use_admt else teacher_a
            with torch.no_grad():
                t_out = teacher(img)
                t_logits = t_out["seg"] if isinstance(t_out, dict) else t_out
                conf, pseudo = torch.softmax(t_logits, 1).max(1)

            perm = torch.randperm(img.shape[0])
            if cfg.teacher.get("substrate", "bcp") == "abd":
                mix_img, mix_lbl = abd_displace(img, lbl, img[perm], pseudo[perm], conf, conf[perm],
                                                grid=cfg.teacher.get("abd_grid", 4),
                                                frac=cfg.teacher.get("abd_frac", 0.5))
            else:
                mix_img, mix_lbl = bcp_copy_paste(img, lbl, img[perm], pseudo[perm])

            opt.zero_grad(set_to_none=True)
            do_distill = bool(ct_teachers) and step % distill_every == 0
            w = cfg.train.get("distill_weight", 0.5)

            if use_fsdp:
                out = student(img)
                loss_sup, _ = sup_loss(out, lbl, epoch=epoch, dist_maps=dm)
                loss_lab = loss_sup
                if do_distill:
                    s_feat = unwrap(student)._distill_feat
                    if s_feat is not None:
                        for name, t, _ in ct_teachers:
                            loss_lab = loss_lab + w * ct_adapters[name](s_feat, t(img))
                loss_lab.backward()
                out_m = student(mix_img)
                logits_m = out_m["seg"] if isinstance(out_m, dict) else out_m
                loss_cons = F.cross_entropy(logits_m, mix_lbl)
                (cons_w * loss_cons).backward()
                gclip = float(cfg.train.get("grad_clip", 12.0))
                if gclip > 0:
                    student.clip_grad_norm_(gclip)
                    for a in ct_adapters.values():
                        a.clip_grad_norm_(gclip)
                opt.step()
            else:
                ac = dict(device_type=device.type, enabled=cfg.amp and device.type == "cuda")
                with torch.autocast(**ac):
                    out = student(img)
                    loss_sup, _ = sup_loss(out, lbl, epoch=epoch, dist_maps=dm)
                    loss_lab = loss_sup
                    if do_distill:
                        s_feat = unwrap(student)._distill_feat
                        if s_feat is not None:
                            for name, t, _ in ct_teachers:
                                loss_lab = loss_lab + w * ct_adapters[name](s_feat, t(img))
                scaler.scale(loss_lab).backward()
                with torch.autocast(**ac):
                    out_m = student(mix_img)
                    logits_m = out_m["seg"] if isinstance(out_m, dict) else out_m
                    loss_cons = F.cross_entropy(logits_m, mix_lbl)
                scaler.scale(cons_w * loss_cons).backward()
                gclip = float(cfg.train.get("grad_clip", 12.0))
                if gclip > 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(params, max_norm=gclip)
                scaler.step(opt)
                scaler.update()

            ema_src = student if use_fsdp else unwrap(student)
            teacher_a.update(ema_src, step)
            if teacher_b is not None:
                teacher_b.update(ema_src, step)
            step += 1

        sd = teacher_a.full_state_dict() if use_fsdp else teacher_a.state_dict()
        if is_main():
            print(f"[teacher e{epoch}] sup={float(loss_sup):.4f} cons={float(loss_cons):.4f} w={cons_w:.3f}")
            torch.save({"model": sd, "epoch": epoch}, run / "checkpoints" / "teacher.pt")

        do_val = val_loader is not None and ((epoch + 1) % val_every == 0 or epoch == cfg.train.epochs - 1)
        if do_val:
            torch.cuda.empty_cache()
            agg = validate_teacher(teacher_a, val_loader, cfg, device)
            if is_main():
                print(f"[teacher e{epoch}] VAL dice={agg['dice']:.4f} hd95={agg['hd95']:.3f} "
                      f"asd={agg['asd']:.3f} score={agg['score']:.4f} (n={agg['n']})")
                with open(run / "history.csv", "a") as f:
                    f.write(f"{epoch},{float(loss_sup):.5f},{float(loss_cons):.5f},"
                            f"{agg['dice']:.5f},{agg['hd95']:.5f},{agg['asd']:.5f},{agg['score']:.5f}\n")
                val_scores.append(float(agg["score"]))
                smoothed = float(np.mean(val_scores[-smooth_k:]))
                if epoch >= sel_min_epoch and smoothed > best_score:
                    best_score = smoothed
                    torch.save({"model": sd, "epoch": epoch, "score": best_score,
                                "raw": float(agg["score"]), "metrics": agg},
                               run / "checkpoints" / "teacher_best.pt")
                    print(f"[teacher e{epoch}] new BEST smoothed={best_score:.4f} "
                          f"(raw={agg['score']:.4f}) -> teacher_best.pt")
        barrier()

    if is_main():
        tag = f"best composite={best_score:.4f} -> teacher_best.pt" if best_score >= 0 else "(no val)"
        print(f"teacher saved -> {run/'checkpoints'/'teacher.pt'} ; {tag}")
    cleanup()


if __name__ == "__main__":
    main()
