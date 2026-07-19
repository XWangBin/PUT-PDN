"""Full-reference fusion metrics used in the PUT experiments."""

from __future__ import annotations

import warnings
from typing import Dict, Union

import numpy as np
import torch
from skimage.metrics import structural_similarity

Array = Union[np.ndarray, torch.Tensor]

try:
    import imgvision as _imgvision
except ImportError:  # Transparent fallback for minimal installations.
    _imgvision = None


def _to_chw(array: Array) -> np.ndarray:
    if isinstance(array, torch.Tensor):
        array = array.detach().float().cpu().numpy()
    result = np.asarray(array, dtype=np.float64)
    if result.ndim == 4:
        if result.shape[0] != 1:
            raise ValueError("Metrics accept one image at a time")
        result = result[0]
    if result.ndim != 3:
        raise ValueError("Expected a CHW or 1CHW image")
    return result


def fusion_metrics(
    reference: Array,
    estimate: Array,
    ratio: int = 4,
    *,
    backend: str = "paper",
) -> Dict[str, float]:
    """Return PSNR, SSIM, SAM (degrees), ERGAS, and RMSE.

    ``backend="paper"`` uses imgvision 0.1.7.3, matching the archived
    experiment scripts.  ``backend="standard"`` uses the transparent NumPy /
    scikit-image definitions below.  The paper backend falls back with a
    warning only when imgvision is unavailable.
    """
    reference = _to_chw(reference)
    estimate = _to_chw(estimate)
    if reference.shape != estimate.shape:
        raise ValueError(f"Metric shapes differ: {reference.shape} and {estimate.shape}")
    if backend not in {"paper", "standard"}:
        raise ValueError("backend must be 'paper' or 'standard'")
    if backend == "paper" and _imgvision is not None:
        metric = _imgvision.spectra_metric(
            np.moveaxis(reference.astype(np.float32, copy=False), 0, -1),
            np.moveaxis(estimate.astype(np.float32, copy=False), 0, -1),
            scale=ratio,
        )
        mse = float(metric.MSE())
        return {
            "psnr": float(metric.PSNR()),
            "ssim": float(metric.SSIM()),
            "sam": float(metric.SAM()),
            "ergas": float(metric.ERGAS()),
            "rmse": float(np.sqrt(mse)),
        }
    if backend == "paper":
        warnings.warn(
            "imgvision is unavailable; using the standard metric backend instead",
            stacklevel=2,
        )

    error = estimate - reference
    mse = float(np.mean(error * error))
    rmse = float(np.sqrt(mse))
    data_range = float(reference.max() - reference.min())
    if data_range <= 0:
        data_range = 1.0
    psnr = float("inf") if mse == 0 else float(10.0 * np.log10(data_range * data_range / mse))

    height, width = reference.shape[-2:]
    window = min(7, height, width)
    if window % 2 == 0:
        window -= 1
    if window < 3:
        raise ValueError("SSIM requires spatial dimensions of at least 3x3")
    ssim = float(
        np.mean(
            [
                structural_similarity(
                    reference[band],
                    estimate[band],
                    data_range=data_range,
                    win_size=window,
                )
                for band in range(reference.shape[0])
            ]
        )
    )

    reference_pixels = np.moveaxis(reference, 0, -1).reshape(-1, reference.shape[0])
    estimate_pixels = np.moveaxis(estimate, 0, -1).reshape(-1, estimate.shape[0])
    dot = np.sum(reference_pixels * estimate_pixels, axis=1)
    norms = np.linalg.norm(reference_pixels, axis=1) * np.linalg.norm(estimate_pixels, axis=1)
    valid = norms > np.finfo(np.float64).eps
    cosine = np.clip(dot[valid] / norms[valid], -1.0, 1.0)
    sam = float(np.degrees(np.mean(np.arccos(cosine)))) if cosine.size else float("nan")

    band_rmse = np.sqrt(np.mean(error * error, axis=(1, 2)))
    band_mean = np.mean(reference, axis=(1, 2))
    valid_bands = np.abs(band_mean) > np.finfo(np.float64).eps
    ergas = float(
        100.0
        / ratio
        * np.sqrt(np.mean((band_rmse[valid_bands] / band_mean[valid_bands]) ** 2))
    )
    return {"psnr": psnr, "ssim": ssim, "sam": sam, "ergas": ergas, "rmse": rmse}
