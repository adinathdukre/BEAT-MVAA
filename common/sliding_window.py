from __future__ import annotations

from typing import Sequence

import torch
from monai.inferers import sliding_window_inference as _monai_sw


@torch.inference_mode()
def sliding_window_predict(
    model: torch.nn.Module,
    image: torch.Tensor,
    roi_size: Sequence[int],
    sw_batch_size: int = 1,
    overlap: float = 0.5,
    mode: str = "gaussian",
    amp: bool = True,
    flip_axes: Sequence[int] = (),
    flip_combos: bool = False,
) -> torch.Tensor:
    from itertools import combinations as _combs
    device = image.device

    def _infer(x: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            out = model(x)
        if isinstance(out, dict):
            out = out["seg"]
        elif isinstance(out, (list, tuple)):
            out = out[0]
        return out.float()

    def _run(x: torch.Tensor) -> torch.Tensor:
        logits = _monai_sw(
            inputs=x,
            roi_size=tuple(roi_size),
            sw_batch_size=sw_batch_size,
            predictor=_infer,
            overlap=overlap,
            mode=mode,
        )
        return torch.softmax(logits, dim=1)

    prob = _run(image)
    if not flip_axes:
        return prob
    if flip_combos:
        dims_all = [ax + 2 for ax in flip_axes]
        n = 1
        for r in range(1, len(dims_all) + 1):
            for combo in _combs(dims_all, r):
                flipped = _run(torch.flip(image, dims=combo))
                prob = prob + torch.flip(flipped, dims=combo)
                n += 1
        return prob / n
    n = 1
    for ax in flip_axes:
        dim = ax + 2
        flipped = _run(torch.flip(image, dims=(dim,)))
        prob = prob + torch.flip(flipped, dims=(dim,))
        n += 1
    return prob / n
