from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _one_hot(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    spatial = labels.shape[1:]
    oh = torch.zeros((labels.shape[0], num_classes, *spatial), device=labels.device, dtype=torch.float32)
    oh.scatter_(1, labels.unsqueeze(1).long(), 1.0)
    return oh


class DiceCELoss(nn.Module):
    def __init__(self, num_classes: int, smooth: float = 1e-5, ce_weight: float = 1.0,
                 dice_weight: float = 1.0, ignore_index: int | None = None):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.long()
        if self.ignore_index is None:
            target = target.clamp_min(0)
        probs = torch.softmax(logits, dim=1)
        dims = tuple(range(2, probs.ndim))
        if self.ignore_index is not None:
            valid = (target != self.ignore_index)
            ce = F.cross_entropy(logits, target, ignore_index=self.ignore_index) if valid.any() \
                else logits.sum() * 0.0
            safe = torch.where(valid, target, torch.zeros_like(target))
            oh = _one_hot(safe, self.num_classes)
            m = valid.unsqueeze(1).to(probs.dtype)
            probs = probs * m
            oh = oh * m
        else:
            ce = F.cross_entropy(logits, target)
            oh = _one_hot(target, self.num_classes)
        inter = (probs * oh).sum(dims)
        denom = probs.sum(dims) + oh.sum(dims)
        dice = 1.0 - ((2 * inter + self.smooth) / (denom + self.smooth)).mean()
        return self.ce_weight * ce + self.dice_weight * dice


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: Optional[float] = None, ignore_index: int = -100):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.long()
        logp = F.log_softmax(logits, dim=1)
        p = torch.exp(logp)
        ce = F.nll_loss(logp, target, reduction="none", ignore_index=self.ignore_index)
        pt = p.gather(1, target.clamp(min=0).unsqueeze(1)).squeeze(1)
        loss = (1 - pt) ** self.gamma * ce
        if self.alpha is not None:
            loss = self.alpha * loss
        valid = (target != self.ignore_index)
        return loss.sum() / valid.sum().clamp(min=1)


class MbLSLoss(nn.Module):
    def __init__(self, margin: float = 10.0):
        super().__init__()
        self.margin = margin

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        zmax = logits.max(dim=1, keepdim=True).values
        return torch.relu(zmax - logits - self.margin).mean()


def compute_boundary_dist_maps(labels: torch.Tensor, num_classes: int,
                               sampling: Optional[Sequence[float]] = None) -> torch.Tensor:
    from common.sdf import mask_to_sdf

    arr = labels.detach().cpu().numpy()
    B = arr.shape[0]
    chans = []
    for b in range(B):
        per_c = [mask_to_sdf(arr[b] == c, sampling, normalize=False) for c in range(1, max(num_classes, 2))]
        chans.append(np.stack(per_c, axis=0))
    out = np.stack(chans, axis=0).astype(np.float32)
    return torch.from_numpy(out).to(labels.device)


class BoundaryLoss(nn.Module):
    def __init__(self, num_classes: int, sampling: Optional[Sequence[float]] = None):
        super().__init__()
        self.num_classes = num_classes
        self.sampling = sampling

    def forward(self, logits: torch.Tensor, target: torch.Tensor,
                dist_maps: Optional[torch.Tensor] = None) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)[:, 1:]
        if dist_maps is None:
            dist_maps = compute_boundary_dist_maps(target, self.num_classes, self.sampling)
        return (probs * dist_maps).mean()


class FarFPLoss(nn.Module):
    def __init__(self, num_classes: int, power: float = 2.0, norm: float = 64.0,
                 sampling: Optional[Sequence[float]] = None):
        super().__init__()
        self.num_classes = num_classes
        self.power = float(power)
        self.norm = float(norm)
        self.sampling = sampling

    def forward(self, logits: torch.Tensor, target: torch.Tensor,
                dist_maps: Optional[torch.Tensor] = None) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)[:, 1:]
        if dist_maps is None:
            dist_maps = compute_boundary_dist_maps(target, self.num_classes, self.sampling)
        outside = torch.relu(dist_maps) / self.norm
        return (probs * outside.pow(self.power)).mean()


class SDFRegressionLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, sdf_pred: torch.Tensor, sdf_target: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(torch.tanh(sdf_pred), sdf_target)


class SkeletonRecallLoss(nn.Module):
    def __init__(self, num_classes: int, smooth: float = 1e-5):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    @staticmethod
    def gt_skeleton(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
        from skimage.morphology import binary_dilation, skeletonize

        arr = labels.detach().cpu().numpy()
        chans = []
        for b in range(arr.shape[0]):
            per_c = []
            for c in range(1, max(num_classes, 2)):
                m = (arr[b] == c)
                sk = skeletonize(m) if m.any() else np.zeros_like(m)
                if sk.any():
                    sk = binary_dilation(binary_dilation(sk)) & m
                per_c.append(sk.astype(np.float32))
            chans.append(np.stack(per_c, axis=0))
        return torch.from_numpy(np.stack(chans, axis=0).astype(np.float32)).to(labels.device)

    def forward(self, logits: torch.Tensor, target: torch.Tensor,
                skeleton: Optional[torch.Tensor] = None) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)[:, 1:]
        if skeleton is None:
            skeleton = self.gt_skeleton(target, self.num_classes)
        dims = tuple(range(2, probs.ndim))
        tp = (probs * skeleton).sum(dims)
        denom = skeleton.sum(dims)
        recall = (tp + self.smooth) / (denom + self.smooth)
        return (1.0 - recall).mean()


def soft_erode(img: torch.Tensor) -> torch.Tensor:
    if img.ndim == 4:
        p1 = -F.max_pool2d(-img, (3, 1), 1, (1, 0))
        p2 = -F.max_pool2d(-img, (1, 3), 1, (0, 1))
        return torch.min(p1, p2)
    p1 = -F.max_pool3d(-img, (3, 1, 1), 1, (1, 0, 0))
    p2 = -F.max_pool3d(-img, (1, 3, 1), 1, (0, 1, 0))
    p3 = -F.max_pool3d(-img, (1, 1, 3), 1, (0, 0, 1))
    return torch.min(torch.min(p1, p2), p3)


def soft_dilate(img: torch.Tensor) -> torch.Tensor:
    if img.ndim == 4:
        return F.max_pool2d(img, 3, 1, 1)
    return F.max_pool3d(img, 3, 1, 1)


def soft_skel(img: torch.Tensor, iters: int = 3) -> torch.Tensor:
    img1 = soft_erode(img)
    skel = F.relu(img - soft_dilate(img1))
    for _ in range(iters):
        img = soft_erode(img)
        delta = F.relu(img - soft_dilate(soft_erode(img)))
        skel = skel + F.relu(delta - skel * delta)
    return skel


class CLDiceLoss(nn.Module):
    def __init__(self, iters: int = 10, smooth: float = 1e-5):
        super().__init__()
        self.iters = iters
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)[:, 1:].sum(1, keepdim=True).clamp(0, 1)
        gt = (target > 0).float().unsqueeze(1)
        skel_p = soft_skel(probs, self.iters)
        skel_g = soft_skel(gt, self.iters)
        tprec = (skel_p * gt).sum() / (skel_p.sum() + self.smooth)
        tsens = (skel_g * probs).sum() / (skel_g.sum() + self.smooth)
        cl = 2.0 * tprec * tsens / (tprec + tsens + self.smooth)
        return 1.0 - cl


def _edt_per_sample(mask: torch.Tensor) -> torch.Tensor:
    mask = (mask > 0.5).float()
    try:
        from monai.transforms import distance_transform_edt as _medt

        return _medt(mask.unsqueeze(1)).squeeze(1).float()
    except Exception:
        from scipy.ndimage import distance_transform_edt as _sedt

        arr = mask.detach().cpu().numpy()
        out = np.stack([_sedt(arr[i]) for i in range(arr.shape[0])], axis=0)
        return torch.from_numpy(out).to(mask.device, dtype=torch.float32)


def _cb_combine(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
    AC, BC = A * C, B * C
    D = BC.clone()
    m = (A != 0) & (B == 0)
    D[m] = AC[m]
    return D


def _cbdice_weights(mask_input: torch.Tensor, skel_input: torch.Tensor, dim: int, prob_flag: bool):
    if prob_flag:
        mask_prob, skel_prob = mask_input, skel_input
        mask = (mask_prob > 0.5).int()
        skel = (skel_prob > 0.5).int()
    else:
        mask = mask_input
        skel = skel_input

    distances = _edt_per_sample(mask).clone()
    distances[mask == 0] = 0
    skel_radius = torch.zeros_like(distances)
    skel_radius[skel == 1] = distances[skel == 1]

    dist_map_norm = torch.zeros_like(distances)
    skel_R_norm = torch.zeros_like(distances)
    I_norm = torch.zeros_like(distances)
    for i in range(distances.shape[0]):
        smax = float(max(skel_radius[i].max().item(), 1.0))
        _sr = skel_radius[i]
        _pos = _sr[_sr > 0]
        smin = float(max(_pos.min().item(), 1.0)) if _pos.numel() > 0 else 1.0
        dist_map_norm[i] = distances[i].clamp(max=smax) / smax
        skel_R_norm[i] = skel_radius[i] / smax
        inv = (smax - skel_radius[i] + smin) / smax
        I_norm[i] = inv if dim == 2 else inv ** 2
    I_norm[skel == 0] = 0

    if prob_flag:
        return dist_map_norm * mask_prob, skel_R_norm * mask_prob, I_norm * skel_prob
    return dist_map_norm * mask, skel_R_norm * mask, I_norm * skel


class CBDiceLoss(nn.Module):
    def __init__(self, iters: int = 10, smooth: float = 1.0):
        super().__init__()
        self.iters = iters
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        dim = logits.ndim - 2
        fore = logits[:, 1:].amax(dim=1, keepdim=True)
        prob = torch.softmax(torch.cat([logits[:, :1], fore], dim=1), dim=1)[:, 1]
        with torch.no_grad():
            true = (target > 0).float()
            pred_hard = (prob > 0.5).float()
            skel_pred = soft_skel(pred_hard.unsqueeze(1), self.iters).squeeze(1)
            skel_true = soft_skel(true.unsqueeze(1), self.iters).squeeze(1)
        skel_pred_prob = skel_pred * prob

        q_vl, q_slvl, q_sl = _cbdice_weights(true, skel_true, dim, prob_flag=False)
        q_vp, q_spvp, q_sp = _cbdice_weights(prob, skel_pred_prob, dim, prob_flag=True)

        w_tprec = (torch.sum(q_sp * q_vl) + self.smooth) / (torch.sum(_cb_combine(q_spvp, q_slvl, q_sp)) + self.smooth)
        w_tsens = (torch.sum(q_sl * q_vp) + self.smooth) / (torch.sum(_cb_combine(q_slvl, q_spvp, q_sl)) + self.smooth)
        hm = 2.0 * w_tprec * w_tsens / (w_tprec + w_tsens + 1e-8)
        return 1.0 - hm


class GeneralizedSurfaceLoss(nn.Module):
    def __init__(self, num_classes: int, sampling: Optional[Sequence[float]] = None, smooth: float = 1e-5):
        super().__init__()
        self.num_classes = num_classes
        self.sampling = sampling
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor,
                dist_maps: Optional[torch.Tensor] = None) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)[:, 1:]
        oh = _one_hot(target, self.num_classes)[:, 1:]
        if dist_maps is None:
            dist_maps = compute_boundary_dist_maps(target, self.num_classes, self.sampling)
        phi = dist_maps.abs()
        dims = tuple(range(2, probs.ndim))
        diff = 1.0 - (oh + probs)
        num = ((phi * diff) ** 2).sum(dims)
        den = (phi ** 2).sum(dims) + self.smooth
        return (1.0 - num / den).mean()


def _sobel_edges(mask: torch.Tensor) -> torch.Tensor:
    if mask.ndim == 4:
        kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=mask.dtype, device=mask.device)
        ky = kx.t()
        kx = kx.view(1, 1, 3, 3)
        ky = ky.view(1, 1, 3, 3)
        gx = F.conv2d(mask, kx, padding=1)
        gy = F.conv2d(mask, ky, padding=1)
        return torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)
    g = 0.0
    for ax in range(2, 5):
        g = g + torch.gradient(mask, dim=ax)[0] ** 2
    return torch.sqrt(g + 1e-6)


class EdgeConstraintLoss(nn.Module):
    def __init__(self, num_classes: int, smooth: float = 1e-5):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, edge_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        gt = (target > 0).float().unsqueeze(1)
        gt_edge = (_sobel_edges(gt) > 0.5).float()
        pred = torch.sigmoid(edge_logits)
        bce = F.binary_cross_entropy(pred.clamp(1e-6, 1 - 1e-6), gt_edge)
        inter = (pred * gt_edge).sum()
        dice = 1.0 - (2 * inter + self.smooth) / (pred.sum() + gt_edge.sum() + self.smooth)
        return bce + dice


class WaveletHFLoss(nn.Module):
    def __init__(self, num_classes: int, levels: int = 1):
        super().__init__()
        self.num_classes = num_classes
        self.levels = levels

    def _highfreq(self, x: torch.Tensor) -> torch.Tensor:
        from common.freq import _haar_kernels

        dims = x.ndim - 2
        out = []
        cur = x
        for _ in range(self.levels):
            k = _haar_kernels(1, dims, cur.device, cur.dtype)
            if dims == 2:
                sub = F.conv2d(cur, k, stride=2)
            else:
                sub = F.conv3d(cur, k, stride=2)
            out.append(sub[:, 1:])
            cur = sub[:, :1]
        return torch.cat([o.flatten(1) for o in out], dim=1)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prob = torch.softmax(logits, dim=1)[:, 1:].sum(1, keepdim=True).clamp(0, 1)
        gt = (target > 0).float().unsqueeze(1)
        return F.l1_loss(self._highfreq(prob), self._highfreq(gt))


class EdgeConsistencyLoss(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prob = torch.softmax(logits, dim=1)[:, 1:].sum(1, keepdim=True).clamp(0, 1)
        gt = (target > 0).float().unsqueeze(1)
        return F.l1_loss(_sobel_edges(prob), _sobel_edges(gt))


def linear_rampup(epoch: int, rampup_epochs: int, start: float = 0.0, end: float = 1.0) -> float:
    if rampup_epochs <= 0:
        return end
    t = min(max(epoch / float(rampup_epochs), 0.0), 1.0)
    return start + (end - start) * t


def sigmoid_rampup(epoch: int, rampup_epochs: int) -> float:
    if rampup_epochs <= 0:
        return 1.0
    t = float(np.clip(epoch, 0.0, rampup_epochs)) / rampup_epochs
    return float(np.exp(-5.0 * (1.0 - t) ** 2))


class CompositeSegLoss(nn.Module):
    def __init__(self, num_classes: int, weights: dict, sampling=None, shape_prior=None):
        super().__init__()
        self.num_classes = num_classes
        self.w = weights
        self.sampling = sampling
        self.ignore_index = weights.get("ignore_index", None)
        self.dice_ce = DiceCELoss(num_classes, ignore_index=self.ignore_index)
        self.shape_consistency = None
        if shape_prior is not None:
            from common.shape_prior import ShapeConsistencyLoss

            self.shape_consistency = ShapeConsistencyLoss(shape_prior)
        self.boundary = BoundaryLoss(num_classes, sampling)
        self.gsl = GeneralizedSurfaceLoss(num_classes, sampling)
        self.skel = SkeletonRecallLoss(num_classes)
        self.cldice = CLDiceLoss()
        self.cbdice = CBDiceLoss()
        self.sdf = SDFRegressionLoss()
        self.edge = EdgeConstraintLoss(num_classes)
        self.focal = FocalLoss()
        self.wavelet_hf = WaveletHFLoss(num_classes)
        self.edge_consistency = EdgeConsistencyLoss(num_classes)

    def forward(self, out: dict, target: torch.Tensor, epoch: int = 0, sdf_target=None,
                dist_maps=None):
        seg = out["seg"] if isinstance(out, dict) else out
        logs = {}
        tgt = target.clamp(min=0)
        total = self.w.get("dice_ce", 1.0) * self.dice_ce(seg, target)
        logs["dice_ce"] = float(total.detach())

        if self.ignore_index is None:
            clean = torch.ones(seg.shape[0], dtype=torch.bool, device=seg.device)
        else:
            has_band = (target == self.ignore_index).reshape(seg.shape[0], -1).any(1)
            clean = ~has_band

        def _shape_loss(fn):
            if bool(clean.any()):
                return fn(seg[clean], tgt[clean])
            return seg.sum() * 0.0

        if isinstance(out, dict) and out.get("ds"):
            ds_loss = 0.0
            for i, ds in enumerate(out["ds"]):
                t = _downsample_label(target, ds.shape[2:])
                ds_loss = ds_loss + 0.5 ** (i + 1) * self.dice_ce(ds, t)
            total = total + ds_loss

        bw = self.w.get("boundary", None)
        w_b = float(bw.get("weight", 0.0)) if bw is not None and hasattr(bw, "get") else 0.0
        if w_b > 0:
            ramp = linear_rampup(epoch, int(bw.get("rampup_epochs", 50)))
            bl = self.boundary(seg, tgt, dist_maps=dist_maps)
            total = total + w_b * ramp * bl
            logs["boundary"] = float(bl.detach())
        if self.w.get("generalized_surface", 0) > 0:
            gl = self.gsl(seg, tgt, dist_maps=dist_maps)
            total = total + self.w["generalized_surface"] * gl
            logs["gsl"] = float(gl.detach())
        if self.w.get("skeleton_recall", 0) > 0:
            sk = _shape_loss(self.skel)
            total = total + self.w["skeleton_recall"] * sk
            logs["skel"] = float(sk.detach())
        if self.w.get("cldice", 0) > 0:
            cl = _shape_loss(self.cldice)
            total = total + self.w["cldice"] * cl
            logs["cldice"] = float(cl.detach())
        if self.w.get("cbdice", 0) > 0:
            cb = _shape_loss(self.cbdice)
            total = total + self.w["cbdice"] * cb
            logs["cbdice"] = float(cb.detach())
        if self.w.get("focal", 0) > 0:
            fl = _shape_loss(self.focal)
            total = total + self.w["focal"] * fl
            logs["focal"] = float(fl.detach())
        if self.w.get("wavelet_hf", 0) > 0:
            wl = _shape_loss(self.wavelet_hf)
            total = total + self.w["wavelet_hf"] * wl
            logs["wavelet_hf"] = float(wl.detach())
        if self.w.get("edge_consistency", 0) > 0:
            ec = _shape_loss(self.edge_consistency)
            total = total + self.w["edge_consistency"] * ec
            logs["edge_consistency"] = float(ec.detach())
        if self.w.get("shape_prior", 0) > 0 and self.shape_consistency is not None:
            sp = self.shape_consistency(seg)
            total = total + self.w["shape_prior"] * sp
            logs["shape_prior"] = float(sp.detach())

        if self.w.get("sdf_l1", 0) > 0 and isinstance(out, dict) and "sdf" in out:
            if sdf_target is None:
                from common.sdf import batch_labels_to_sdf

                sdf_target = batch_labels_to_sdf(tgt, self.num_classes, self.sampling)
            sl = self.sdf(out["sdf"], sdf_target)
            total = total + self.w["sdf_l1"] * sl
            logs["sdf"] = float(sl.detach())
        if self.w.get("edge_constraint", 0) > 0 and isinstance(out, dict) and "edge" in out:
            el = self.edge(out["edge"], tgt)
            total = total + self.w["edge_constraint"] * el
            logs["edge"] = float(el.detach())

        logs["total"] = float(total.detach())
        return total, logs


def _downsample_label(target: torch.Tensor, size) -> torch.Tensor:
    t = target.unsqueeze(1).float()
    mode = "nearest"
    t = F.interpolate(t, size=tuple(size), mode=mode)
    return t.squeeze(1).long()
