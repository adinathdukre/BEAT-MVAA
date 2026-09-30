from __future__ import annotations

from common.registry import build_student3d, export_student  # noqa: F401


def build_model(cfg):
    return build_student3d(cfg)
