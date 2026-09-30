from __future__ import annotations

import logging
import sys
from typing import Optional

_logger = None


def get_logger(name: str = "mvaa") -> logging.Logger:
    global _logger
    if _logger is not None:
        return _logger
    lg = logging.getLogger(name)
    if not lg.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        lg.addHandler(h)
        lg.setLevel(logging.INFO)
        lg.propagate = False
    logging.captureWarnings(True)
    _logger = lg
    return lg


def info(msg: str) -> None:
    get_logger().info(msg)


def warn(msg: str) -> None:
    get_logger().warning(msg)


def report_load(module, state_dict: dict, name: str = "model", warn_below: float = 0.5,
                require: Optional[float] = None):
    model_keys = set(module.state_dict().keys())
    ckpt_keys = set(state_dict.keys())
    matched = model_keys & ckpt_keys
    missing = sorted(model_keys - ckpt_keys)
    unexpected = sorted(ckpt_keys - model_keys)
    try:
        out = module.load_state_dict(state_dict, strict=False)
    except RuntimeError as e:
        warn(f"[load:{name}] load FAILED (shape/dtype mismatch): {e}")
        raise
    frac = len(matched) / max(len(model_keys), 1)
    lg = get_logger()
    lg.info(f"[load:{name}] matched {len(matched)}/{len(model_keys)} params "
            f"| missing {len(missing)} | unexpected {len(unexpected)}")
    if require is not None and frac < require:
        raise RuntimeError(f"[load:{name}] only {frac:.0%} of params matched (< required {require:.0%}) "
                           f"- refusing to deploy a wrong checkpoint. missing={missing[:4]} "
                           f"unexpected={unexpected[:4]}")
    if frac < warn_below:
        lg.warning(f"[load:{name}] ONLY {frac:.0%} of params matched - likely a WRONG/incompatible "
                   f"checkpoint. e.g. missing={missing[:4]} unexpected={unexpected[:4]}")
    elif missing or unexpected:
        lg.info(f"[load:{name}] e.g. missing={missing[:3]} unexpected={unexpected[:3]}")
    return out
