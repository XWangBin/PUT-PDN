"""Direct and overlap-tiled PUT inference."""

from __future__ import annotations

from typing import List

import torch


def _starts(length: int, tile: int, overlap: int) -> List[int]:
    if tile >= length:
        return [0]
    starts = list(range(0, length - tile + 1, tile - overlap))
    if starts[-1] != length - tile:
        starts.append(length - tile)
    return starts


@torch.no_grad()
def predict(
    model: torch.nn.Module,
    lr_hsi: torch.Tensor,
    pan: torch.Tensor,
    *,
    ratio: int,
    tile_size: int = 0,
    overlap: int = 16,
) -> torch.Tensor:
    """Predict one BCHW scene; tiled outputs are blended on CPU."""
    if lr_hsi.ndim != 4 or pan.ndim != 4 or lr_hsi.shape[0] != 1 or pan.shape[0] != 1:
        raise ValueError("predict expects one BCHW LR-HSI/PAN pair")
    expected = (lr_hsi.shape[-2] * ratio, lr_hsi.shape[-1] * ratio)
    if pan.shape[-2:] != expected:
        raise ValueError(f"PAN size must be {expected}, got {pan.shape[-2:]}")
    model.eval()
    if tile_size <= 0 or (tile_size >= pan.shape[-2] and tile_size >= pan.shape[-1]):
        return model(lr_hsi, pan)[0]

    if tile_size % ratio or overlap % ratio:
        raise ValueError("tile_size and overlap must be divisible by ratio")
    if overlap < 0 or overlap >= tile_size:
        raise ValueError("overlap must satisfy 0 <= overlap < tile_size")
    if tile_size % 4:
        raise ValueError("tile_size must be divisible by 4 for the two-level SSSB")

    height, width = pan.shape[-2:]
    tile_h, tile_w = min(tile_size, height), min(tile_size, width)
    if tile_h % ratio or tile_w % ratio:
        raise ValueError("Scene edges and effective tile sizes must be divisible by ratio")
    tops = _starts(height, tile_h, min(overlap, tile_h - ratio))
    lefts = _starts(width, tile_w, min(overlap, tile_w - ratio))
    if any(value % ratio for value in tops + lefts):
        raise ValueError("Scene dimensions must align to the spatial ratio for tiled inference")

    window_y = torch.hann_window(tile_h, periodic=False).clamp_min(1e-3)
    window_x = torch.hann_window(tile_w, periodic=False).clamp_min(1e-3)
    window = (window_y[:, None] * window_x[None, :]).float()[None, None]
    output = torch.zeros((1, lr_hsi.shape[1], height, width), dtype=torch.float32)
    weights = torch.zeros((1, 1, height, width), dtype=torch.float32)

    for top in tops:
        for left in lefts:
            low_top, low_left = top // ratio, left // ratio
            low_h, low_w = tile_h // ratio, tile_w // ratio
            pred = model(
                lr_hsi[:, :, low_top : low_top + low_h, low_left : low_left + low_w],
                pan[:, :, top : top + tile_h, left : left + tile_w],
            )[0]
            pred = pred.detach().float().cpu()
            output[:, :, top : top + tile_h, left : left + tile_w] += pred * window
            weights[:, :, top : top + tile_h, left : left + tile_w] += window
    return output / weights.clamp_min(1e-8)
