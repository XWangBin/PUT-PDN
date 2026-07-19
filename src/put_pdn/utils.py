"""Runtime, reproducibility, and checkpoint helpers."""

from __future__ import annotations

import json
import random
import warnings
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("A CUDA device was requested, but CUDA is unavailable")
    return device


def _model_state(payload: Any) -> Mapping[str, torch.Tensor]:
    if isinstance(payload, Mapping):
        for key in ("state_dict", "model_state_dict", "model"):
            candidate = payload.get(key)
            if isinstance(candidate, Mapping):
                return candidate
        if payload and all(isinstance(value, torch.Tensor) for value in payload.values()):
            return payload
    raise ValueError("Checkpoint does not contain a recognised model state dictionary")


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    *,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
    strict: bool = True,
) -> Dict[str, Any]:
    """Load a PUT checkpoint with strict active-parameter validation."""
    payload = torch.load(path, map_location=device)
    state = _model_state(payload)
    if state and all(key.startswith("module.") for key in state):
        state = {key[len("module.") :]: value for key, value in state.items()}
    # One Chikusei experiment variant stored weights for an optional lms-only
    # initialiser that is never used by the published ms+pan inference path.
    # Other dataset variants comment those layers out entirely.  The public
    # implementation removes the inconsistent dead branch and explicitly
    # discards only its known keys while keeping strict checks for everything
    # else.
    inactive_prefixes = (
        "init_model.conv_z.",
        "init_model.channel_attention.",
        "init_model.spatial_attention.",
    )
    inactive_keys = [key for key in state if key.startswith(inactive_prefixes)]
    if inactive_keys:
        state = {key: value for key, value in state.items() if key not in inactive_keys}
        warnings.warn(
            "Discarded inactive legacy initialiser weights: " + ", ".join(inactive_keys),
            stacklevel=2,
        )
    model.load_state_dict(state, strict=strict)

    if optimizer is not None and isinstance(payload, Mapping):
        optimizer_state = payload.get("optimizer") or payload.get("optimizer_state_dict")
        if optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)
    return dict(payload) if isinstance(payload, Mapping) else {}


def save_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    epoch: int,
    iteration: int,
    best_psnr: float,
    model_config: Mapping[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "epoch": epoch,
            "iteration": iteration,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_psnr": best_psnr,
            "model_config": dict(model_config),
        },
        temporary,
    )
    temporary.replace(path)


def append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(dict(record), ensure_ascii=False) + "\n")
