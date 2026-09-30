from __future__ import annotations

import torch
import torch.nn as nn

from common.registry import build_smp2d

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _pool_presence_feature(feature, pooling):
    average = torch.mean(feature, dim=(2, 3))
    if pooling == "gap":
        return average
    if pooling == "gap_max":
        maximum = torch.amax(feature, dim=(2, 3))
        return torch.cat([average, maximum], dim=1)
    raise ValueError(f"unsupported presence pooling {pooling!r}")


class GlobalPresenceHead(nn.Module):
    def __init__(self, channels, dropout, pooling):
        super().__init__()
        self.pooling = pooling
        self.dropout = nn.Dropout(dropout)
        multiplier = 2 if pooling == "gap_max" else 1
        self.classifier = nn.Linear(channels * multiplier, 1)

    def forward(self, feature):
        pooled = _pool_presence_feature(feature, self.pooling)
        return self.classifier(self.dropout(pooled))


class SpatialTopKPresenceHead(nn.Module):
    def __init__(self, channels, dropout, hidden_channels, topk_fraction):
        super().__init__()
        self.local = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, 1),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(hidden_channels, 1, 1),
        )
        self.topk_fraction = float(topk_fraction)
        if not 0.0 < self.topk_fraction <= 1.0:
            raise ValueError("presence topk_fraction must be in (0, 1]")

    def pool(self, local_logits):
        flattened = local_logits.flatten(1)
        k = max(1, int(round(flattened.shape[1] * self.topk_fraction)))
        return torch.topk(flattened, k, dim=1).values.mean(1)

    def forward(self, feature):
        return self.pool(self.local(feature))

    def forward_with_local(self, feature):
        local_logits = self.local(feature)
        return self.pool(local_logits), local_logits


class RankPresenceHead(nn.Module):
    def __init__(self, channels, dropout, pooling):
        super().__init__()
        self.pooling = pooling
        self.dropout = nn.Dropout(dropout)
        multiplier = 2 if pooling == "gap_max" else 1
        self.classifier = nn.Linear(channels * multiplier + 1, 1)

    def forward(self, feature, frame_rank):
        if frame_rank is None:
            raise ValueError("frame_rank is required")
        pooled = _pool_presence_feature(feature, self.pooling)
        rank = frame_rank.to(pooled).reshape(-1, 1)
        return self.classifier(self.dropout(torch.cat([pooled, rank], dim=1)))


class T3Student(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        presence_cfg = cfg.model.get("presence_head", {}) or {}
        pooling = str(presence_cfg.get("pooling", "gap"))
        if (
            presence_cfg.get("enabled", False)
            and float(presence_cfg.get("dense_weight", 0.0)) > 0.0
            and pooling != "spatial_topk"
        ):
            raise ValueError(
                "dense presence supervision requires spatial_topk pooling"
            )
        self.net = build_smp2d(cfg)
        self.deploy = False
        self.imagenet_norm = bool(cfg.model.get("imagenet_norm", True))
        self.register_buffer("_mean", torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("_std", torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1))
        self.sdf_head = None
        if cfg.model.get("aux_sdf_head", False):
            self.sdf_head = nn.Conv2d(cfg.num_classes, max(cfg.num_classes - 1, 1), 1)
        self.presence_head = None
        self.presence_uses_rank = False
        self.presence_feature_index = -1
        if presence_cfg.get("enabled", False):
            self.presence_feature_index = int(presence_cfg.get("feature_index", -1))
            channels = int(self.net.encoder.out_channels[self.presence_feature_index])
            if channels <= 0:
                raise ValueError("presence feature stage has no channels")
            dropout = float(presence_cfg.get("dropout", 0.2))
            self.presence_uses_rank = bool(presence_cfg.get("temporal_rank", False))
            if self.presence_uses_rank:
                self.presence_head = RankPresenceHead(channels, dropout, pooling)
            elif pooling == "spatial_topk":
                self.presence_head = SpatialTopKPresenceHead(
                    channels,
                    dropout,
                    int(presence_cfg.get("hidden_channels", 64)),
                    float(presence_cfg.get("topk_fraction", 0.05)),
                )
            elif pooling != "gap":
                self.presence_head = GlobalPresenceHead(channels, dropout, pooling)
            else:
                self.presence_head = nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Flatten(),
                    nn.Dropout(dropout),
                    nn.Linear(channels, 1),
                )

    def _normalize(self, x):
        if self.imagenet_norm:
            x = (x - self._mean) / self._std
        return x

    def _presence(self, feature, frame_rank):
        if self.presence_uses_rank:
            return self.presence_head(feature, frame_rank).flatten()
        return self.presence_head(feature).flatten()

    def presence_logits(self, x, frame_rank=None):
        if self.presence_head is None:
            raise RuntimeError("presence head is disabled")
        features = self.net.encoder(self._normalize(x))
        return self._presence(features[self.presence_feature_index], frame_rank)

    def presence_outputs(self, x, frame_rank=None):
        if self.presence_head is None:
            raise RuntimeError("presence head is disabled")
        features = self.net.encoder(self._normalize(x))
        feature = features[self.presence_feature_index]
        if isinstance(self.presence_head, SpatialTopKPresenceHead):
            return self.presence_head.forward_with_local(feature)
        return self._presence(feature, frame_rank), None

    def forward_with_presence(self, x, frame_rank=None):
        if self.presence_head is None:
            raise RuntimeError("presence head is disabled")
        features = self.net.encoder(self._normalize(x))
        decoder = self.net.decoder(features)
        seg = self.net.segmentation_head(decoder)
        presence = self._presence(features[self.presence_feature_index], frame_rank)
        return seg, presence

    def forward(self, x, frame_rank=None):
        x = self._normalize(x)
        if self.presence_head is not None:
            features = self.net.encoder(x)
            decoder = self.net.decoder(features)
            seg = self.net.segmentation_head(decoder)
            if self.deploy:
                return seg
            presence = self._presence(features[self.presence_feature_index], frame_rank)
            out = {"seg": seg, "presence": presence}
            if self.sdf_head is not None:
                out["sdf"] = self.sdf_head(seg)
            return out
        seg = self.net(x)
        if self.deploy or self.sdf_head is None:
            return seg
        return {"seg": seg, "sdf": self.sdf_head(seg)}


def build_model(cfg):
    return T3Student(cfg)


def export_student(model: T3Student) -> T3Student:
    model.sdf_head = None
    model.deploy = True
    model.eval()
    return model
