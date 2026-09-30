from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureDistill(nn.Module):
    def __init__(self, student_dim: int, teacher_dim: int, dims: int = 2):
        super().__init__()
        conv = nn.Conv3d if dims == 3 else nn.Conv2d
        self.adapt = conv(student_dim, teacher_dim, kernel_size=1)
        self.dims = dims

    def forward(self, student_feat: torch.Tensor, teacher_feat: torch.Tensor) -> torch.Tensor:
        s = self.adapt(student_feat)
        t = teacher_feat.detach().to(s.dtype)
        if s.shape[2:] != t.shape[2:]:
            mode = "trilinear" if self.dims == 3 else "bilinear"
            t = F.interpolate(t, size=s.shape[2:], mode=mode, align_corners=False)
        s = F.normalize(s, dim=1, eps=1e-6)
        t = F.normalize(t, dim=1, eps=1e-6)
        return F.mse_loss(s, t)


ENDOVIT_MEAN = (0.3464, 0.2280, 0.2228)
ENDOVIT_STD = (0.2520, 0.2128, 0.2093)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class LighterCT3DTeacher(nn.Module):
    def __init__(self, model_id: str, kind: str = "encoder", src_hu=(-1000.0, 1000.0)):
        super().__init__()
        try:
            from lighter_zoo import SegResEncoder, SegResNet
        except Exception as e:
            raise ImportError("lighter-zoo not installed - `pip install lighter-zoo`.") from e
        if kind == "segresnet":
            self.enc = SegResNet.from_pretrained(model_id).eval().encoder
        else:
            self.enc = SegResEncoder.from_pretrained(model_id).eval()
        self.src_lo, self.src_hi = float(src_hu[0]), float(src_hu[1])
        self.ct_lo, self.ct_hi = -1024.0, 2048.0
        for p in self.parameters():
            p.requires_grad_(False)
        with torch.no_grad():
            feat = self.enc(torch.zeros(1, 1, 64, 64, 64))[-1]
            self.embed_dim = int(feat.shape[1])

    @torch.no_grad()
    def forward(self, vol: torch.Tensor) -> torch.Tensor:
        hu = vol * (self.src_hi - self.src_lo) + self.src_lo
        x = ((hu - self.ct_lo) / (self.ct_hi - self.ct_lo)).clamp(0, 1)
        return self.enc(x)[-1]


def build_ct_distill_teachers(cfg):
    src_hu = tuple(cfg.data.get("hu_clip", (-1000.0, 1000.0)))
    teachers = []
    if cfg.train.get("distill_ctfm", False):
        mid = cfg.distill.get("ctfm_model_id", "project-lighter/ct_fm_feature_extractor")
        t = LighterCT3DTeacher(mid, kind="encoder", src_hu=src_hu)
        teachers.append(("ctfm", t, t.embed_dim))
    if cfg.train.get("distill_wholebody", False):
        mid = cfg.distill.get("whole_body_seg_id", "project-lighter/whole_body_segmentation")
        t = LighterCT3DTeacher(mid, kind="segresnet", src_hu=src_hu)
        teachers.append(("wholebody", t, t.embed_dim))
    return teachers
