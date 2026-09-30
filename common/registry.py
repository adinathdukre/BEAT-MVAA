from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch.nn as nn


class STUNet3D(nn.Module):
    def __init__(self, net: nn.Module, num_classes: int, deep_supervision: bool = True):
        super().__init__()
        self.net = net
        self.num_classes = num_classes
        self.deep_supervision = deep_supervision
        self.deploy = False
        self._distill_feat = None
        self._fcon_feat = None
        self._replace_seg_head(num_classes)
        self._set_ds(deep_supervision)

    def _set_ds(self, flag: bool) -> None:
        if hasattr(self.net, "do_ds"):
            self.net.do_ds = flag
        dec = getattr(self.net, "decoder", None)
        if dec is not None and hasattr(dec, "deep_supervision"):
            dec.deep_supervision = flag

    def enable_grad_checkpointing(self) -> bool:
        import torch
        import torch.utils.checkpoint as cp

        stages = []
        for attr in ("conv_blocks_context", "conv_blocks_localization"):
            mods = getattr(self.net, attr, None)
            if mods is not None:
                stages.extend(list(mods))
        if not stages:
            return False

        def _wrap(module):
            if getattr(module, "_ckpt_wrapped", False):
                return
            orig_forward = module.forward

            def fwd(*a, **k):
                if module.training and torch.is_grad_enabled():
                    return cp.checkpoint(orig_forward, *a, use_reentrant=False, **k)
                return orig_forward(*a, **k)

            module.forward = fwd
            module._ckpt_wrapped = True

        for m in stages:
            _wrap(m)
        return True

    def enable_distill_capture(self):
        so = getattr(self.net, "seg_outputs", None)
        if not so or len(so) == 0:
            return False, 0

        def _hook(_module, _inp):
            self._distill_feat = _inp[0]

        so[0].register_forward_pre_hook(_hook)
        return True, so[0].in_channels

    def enable_fconsist_capture(self, stage_idx: int = 1):
        ctx = getattr(self.net, "conv_blocks_context", None)
        if not ctx or len(ctx) <= stage_idx:
            return False, -1

        def _hook(_module, _inp, _out):
            self._fcon_feat = _out

        ctx[stage_idx].register_forward_hook(_hook)
        return True, stage_idx

    def _replace_seg_head(self, num_classes: int) -> None:
        from common.log import info, warn

        if hasattr(self.net, "seg_outputs") and isinstance(self.net.seg_outputs, (nn.ModuleList, list)):
            new = nn.ModuleList()
            for conv in self.net.seg_outputs:
                new.append(nn.Conv3d(conv.in_channels, num_classes, kernel_size=1, bias=conv.bias is not None))
            self.net.seg_outputs = new
            info(f"[STU-Net] re-initialized {len(new)} seg head(s) -> {num_classes} classes")
        else:
            warn("[STU-Net] no `seg_outputs` found - head NOT re-init; check the medim STU-Net version.")
        if hasattr(self.net, "num_classes"):
            self.net.num_classes = num_classes

    def forward(self, x):
        self._set_ds(self.deep_supervision and not self.deploy)
        out = self.net(x)
        if self.deploy:
            return out[0] if isinstance(out, (list, tuple)) else out
        if isinstance(out, (list, tuple)):
            return {"seg": out[0], "ds": list(out[1:])}
        return {"seg": out, "ds": []}


def build_stunet(size: str, ckpt: Optional[str], num_classes: int,
                 in_channels: int = 1, deep_supervision: bool = True,
                 load_pretrained: bool = True) -> STUNet3D:
    import medim

    from common.log import info, warn

    has_ckpt = load_pretrained and ckpt is not None and Path(str(ckpt)).exists()
    net = medim.create_model(f"STU-Net-{size}", pretrained=has_ckpt,
                             checkpoint_path=str(ckpt) if has_ckpt else None)
    if has_ckpt:
        info(f"[STU-Net-{size}] loaded pretrained weights from {ckpt}")
    elif not load_pretrained:
        info(f"[STU-Net-{size}] skipped base-weight load (a task checkpoint is loaded afterwards)")
    elif ckpt:
        raise FileNotFoundError(
            f"[STU-Net-{size}] configured pretrained checkpoint NOT found: {ckpt}. Training from here would "
            f"silently discard the TotalSegmentator pretraining (the whole point of STU-Net) and waste GPU-days. "
            f"Download the STU-Net weights (see README), or set model.stunet_ckpt=null to force random init.")
    else:
        info(f"[STU-Net-{size}] random init (no pretrained checkpoint configured)")
    return STUNet3D(net, num_classes, deep_supervision)


def build_student3d(cfg, load_pretrained: bool = True):
    m = cfg.model
    return build_stunet(m.get("stunet_size", "B"), m.get("stunet_ckpt"), cfg.num_classes,
                        in_channels=m.get("in_channels", 1), deep_supervision=m.get("deep_supervision", True),
                        load_pretrained=load_pretrained)


def build_teacher3d(cfg, load_pretrained: bool = True) -> STUNet3D:
    t = cfg.get("teacher", {})
    return build_stunet(t.get("stunet_size", "H"), t.get("stunet_ckpt"), cfg.num_classes,
                        in_channels=cfg.model.get("in_channels", 1), deep_supervision=True,
                        load_pretrained=load_pretrained)


def build_smp2d(cfg) -> nn.Module:
    import segmentation_models_pytorch as smp

    from common.log import warn

    m = cfg.model
    arch = m.get("arch", "unetplusplus").lower()
    encoder = m.get("encoder", "efficientnet-b3")
    _ARCHES = {"unetplusplus": "UnetPlusPlus", "deeplabv3plus": "DeepLabV3Plus", "unet": "Unet",
               "fpn": "FPN", "segformer": "Segformer", "manet": "MAnet"}
    cls_name = _ARCHES.get(arch)
    if cls_name is None or not hasattr(smp, cls_name):
        raise ValueError(f"[smp2d] decoder arch '{arch}' unavailable in smp {smp.__version__} "
                         f"(have: {[a for a in _ARCHES if hasattr(smp, _ARCHES[a])]})")
    builder = getattr(smp, cls_name)
    _ew = m.get("encoder_weights", "imagenet")
    enc_weights = None if str(_ew).lower() in ("null", "none", "") else _ew
    net = builder(encoder_name=encoder, encoder_weights=enc_weights,
                  in_channels=m.get("in_channels", 3), classes=cfg.num_classes)
    return net


def export_student(model: nn.Module) -> nn.Module:
    if isinstance(model, STUNet3D):
        model.deep_supervision = False
        model.deploy = True
        model._set_ds(False)
        so = getattr(model.net, "seg_outputs", None)
        if so:
            so[0]._forward_pre_hooks.clear()
        model._distill_feat = None
    model.eval()
    return model


def has_aux_heads(model: nn.Module) -> bool:
    m = model.module if hasattr(model, "module") else model
    if isinstance(m, STUNet3D):
        so = getattr(m.net, "seg_outputs", None)
        if so and len(getattr(so[0], "_forward_pre_hooks", {})) > 0:
            return True
        if getattr(m, "_distill_feat", None) is not None:
            return True
    return False
