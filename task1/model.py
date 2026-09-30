from __future__ import annotations

from common.registry import build_student3d, build_teacher3d, export_student  # noqa: F401


def build_student(cfg):
    return build_student3d(cfg)


def build_teacher(cfg):
    return build_teacher3d(cfg)
