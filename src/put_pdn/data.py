"""Dataset loaders used by the six PUT experiment directories.

All returned tensors use channel-first layout.  Simulated data are represented
as ``pan`` (HR-PAN), ``lr_hsi`` (LR-HSI), and ``target`` (HR-HSI).  The
full-resolution Liaoning splits have no target and therefore omit that key.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import cv2
import h5py
import numpy as np
import scipy.io as sio
import tifffile
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class DatasetDefaults:
    bands: int
    epochs: int
    stages: int = 3
    ratio: int = 4
    patch_size: int = 64
    stride: int = 24


DATASET_DEFAULTS: Dict[str, DatasetDefaults] = {
    "chikusei": DatasetDefaults(bands=128, epochs=300, stages=5),
    "indian": DatasetDefaults(bands=220, epochs=100, stages=3),
    "pavia": DatasetDefaults(bands=102, epochs=100, stages=3, stride=64),
    "xiongan": DatasetDefaults(bands=93, epochs=300, stages=5),
    "ln1": DatasetDefaults(bands=166, epochs=100, stages=3),
    "ln2": DatasetDefaults(bands=144, epochs=100, stages=3),
}

ALIASES = {
    "liaoning-1": "ln1",
    "liaoning1": "ln1",
    "liaoning-2": "ln2",
    "liaoning2": "ln2",
}


def canonical_dataset_name(name: str) -> str:
    key = name.strip().lower()
    key = ALIASES.get(key, key)
    if key not in DATASET_DEFAULTS:
        choices = ", ".join(DATASET_DEFAULTS)
        raise ValueError(f"Unknown dataset '{name}'. Choose one of: {choices}")
    return key


def _normalise_max(cube: np.ndarray) -> np.ndarray:
    cube = np.asarray(cube, dtype=np.float32)
    maximum = float(cube.max())
    if maximum <= 0:
        raise ValueError("Cannot normalise an array with a non-positive maximum")
    return cube / maximum


def _chw(array: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(array, dtype=np.float32).transpose(2, 0, 1))


def _pan_from_bands(cube_hwc: np.ndarray, band_slice: slice) -> np.ndarray:
    pan = cube_hwc[:, :, band_slice].mean(axis=2, dtype=np.float32)
    return np.ascontiguousarray(pan[None, :, :])


def gaussian_downsample(hsi_chw: np.ndarray, ratio: int) -> np.ndarray:
    """Apply the Gaussian degradation used by the original simulated loaders."""
    if ratio < 1:
        raise ValueError("ratio must be positive")
    hsi_hwc = np.asarray(hsi_chw, dtype=np.float32).transpose(1, 2, 0)
    kernel = ratio * 2 + 1
    blurred = cv2.GaussianBlur(
        hsi_hwc,
        ksize=(kernel, kernel),
        sigmaX=ratio * 0.666,
        sigmaY=ratio * 0.666,
    )
    offset = ratio // 2
    return _chw(blurred[offset::ratio, offset::ratio, :])


def _resize_cube(cube_hwc: np.ndarray, height: int, width: int) -> np.ndarray:
    output = np.empty((height, width, cube_hwc.shape[2]), dtype=np.float32)
    for band in range(cube_hwc.shape[2]):
        output[:, :, band] = cv2.resize(
            cube_hwc[:, :, band],
            (width, height),
            interpolation=cv2.INTER_AREA,
        )
    return output


def _augment_triplet(arrays: Sequence[np.ndarray]) -> Tuple[np.ndarray, ...]:
    rotations = random.randint(0, 3)
    vertical_flip = random.randint(0, 1)
    horizontal_flip = random.randint(0, 1)
    augmented = []
    for array in arrays:
        result = np.rot90(array, rotations, axes=(1, 2))
        if vertical_flip:
            result = result[:, :, ::-1]
        if horizontal_flip:
            result = result[:, ::-1, :]
        augmented.append(np.ascontiguousarray(result))
    return tuple(augmented)


class TripletDataset(Dataset):
    """Patch or pre-patched supervised HSI-PAN fusion data."""

    def __init__(
        self,
        pan: np.ndarray,
        target: np.ndarray,
        lr_hsi: Optional[np.ndarray],
        *,
        ratio: int,
        patch_size: Optional[int],
        stride: int,
        augment: bool,
        derive_lr_hsi: bool = False,
    ) -> None:
        self.pan = np.asarray(pan, dtype=np.float32)
        self.target = np.asarray(target, dtype=np.float32)
        self.lr_hsi = None if lr_hsi is None else np.asarray(lr_hsi, dtype=np.float32)
        self.ratio = int(ratio)
        self.patch_size = patch_size
        self.stride = int(stride)
        self.augment = bool(augment)
        self.derive_lr_hsi = bool(derive_lr_hsi)
        self.prepatched = self.target.ndim == 4

        if self.prepatched:
            if self.pan.ndim != 4 or self.lr_hsi is None or self.lr_hsi.ndim != 4:
                raise ValueError("Pre-patched data must be NCHW arrays for PAN, GT, and LR-HSI")
            if not (len(self.pan) == len(self.target) == len(self.lr_hsi)):
                raise ValueError("Pre-patched PAN, GT, and LR-HSI sample counts differ")
            self.positions = []
            return

        if self.pan.ndim != 3 or self.target.ndim != 3:
            raise ValueError("Scene data must be CHW arrays")
        if self.pan.shape[-2:] != self.target.shape[-2:]:
            raise ValueError("PAN and target HR-HSI spatial sizes differ")
        if not self.derive_lr_hsi and self.lr_hsi is None:
            raise ValueError("lr_hsi is required unless derive_lr_hsi=True")

        height, width = self.target.shape[-2:]
        if patch_size is None:
            self.positions = [(0, 0, height, width)]
        else:
            if patch_size % self.ratio:
                raise ValueError("patch_size must be divisible by ratio")
            if self.stride % self.ratio:
                raise ValueError("stride must be divisible by ratio")
            if height < patch_size or width < patch_size:
                raise ValueError("patch_size is larger than the scene")
            self.positions = [
                (top, left, patch_size, patch_size)
                for top in range(0, height - patch_size + 1, self.stride)
                for left in range(0, width - patch_size + 1, self.stride)
            ]

    def __len__(self) -> int:
        return len(self.target) if self.prepatched else len(self.positions)

    def __getitem__(self, index: int):
        if self.prepatched:
            pan = self.pan[index]
            target = self.target[index]
            lr_hsi = self.lr_hsi[index]
        else:
            top, left, height, width = self.positions[index]
            pan = self.pan[:, top : top + height, left : left + width]
            target = self.target[:, top : top + height, left : left + width]
            if self.derive_lr_hsi:
                lr_hsi = gaussian_downsample(target, self.ratio)
            else:
                low_top, low_left = top // self.ratio, left // self.ratio
                low_height, low_width = height // self.ratio, width // self.ratio
                lr_hsi = self.lr_hsi[
                    :, low_top : low_top + low_height, low_left : low_left + low_width
                ]

        arrays = tuple(np.ascontiguousarray(x) for x in (pan, target, lr_hsi))
        if self.augment:
            arrays = _augment_triplet(arrays)
        pan, target, lr_hsi = arrays
        return {
            "pan": torch.from_numpy(pan),
            "target": torch.from_numpy(target),
            "lr_hsi": torch.from_numpy(lr_hsi),
        }


class FullSceneDataset(Dataset):
    """One full-resolution scene, optionally with a target."""

    def __init__(
        self,
        pan: np.ndarray,
        lr_hsi: np.ndarray,
        target: Optional[np.ndarray] = None,
    ) -> None:
        self.pan = np.ascontiguousarray(pan, dtype=np.float32)
        self.lr_hsi = np.ascontiguousarray(lr_hsi, dtype=np.float32)
        self.target = None if target is None else np.ascontiguousarray(target, dtype=np.float32)

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int):
        if index != 0:
            raise IndexError(index)
        sample = {
            "pan": torch.from_numpy(self.pan),
            "lr_hsi": torch.from_numpy(self.lr_hsi),
        }
        if self.target is not None:
            sample["target"] = torch.from_numpy(self.target)
        return sample


def _load_chikusei(root: Path, split: str, ratio: int, patch: int, stride: int, augment: bool):
    if split == "val":
        cube = _normalise_max(np.load(root / "GT.npy"))
        pan = _pan_from_bands(cube, slice(60, 80))
        target = _chw(cube)
        lr_hsi = gaussian_downsample(target, ratio)
        return TripletDataset(pan, target, lr_hsi, ratio=ratio, patch_size=None, stride=stride, augment=False)

    with h5py.File(root / "Chikusei.mat", "r") as handle:
        cube = np.asarray(handle["chikusei"], dtype=np.float32).transpose(1, 2, 0)
    cube = cube.copy()
    cube[300:812, 300:812] = cube[812:1324, 812:1324]
    cube = _normalise_max(cube)
    return TripletDataset(
        _pan_from_bands(cube, slice(60, 80)),
        _chw(cube),
        None,
        ratio=ratio,
        patch_size=patch,
        stride=stride,
        augment=augment,
        derive_lr_hsi=True,
    )


def _load_indian(root: Path, split: str, ratio: int, patch: int, stride: int, augment: bool):
    cube = tifffile.imread(root / "19920612_AVIRIS_IndianPine_NS-line.tif")
    cube = np.asarray(cube, dtype=np.float32).transpose(1, 2, 0)
    if split == "val":
        cube = _normalise_max(cube)
        cube = cube[1700:2212, 80:592]
        target = _chw(cube)
        return TripletDataset(
            _pan_from_bands(cube, slice(15, 55)),
            target,
            gaussian_downsample(target, ratio),
            ratio=ratio,
            patch_size=None,
            stride=stride,
            augment=False,
        )

    cube = cube.copy()
    cube[1700:2212, 80:592] = cube[:512, -512:]
    cube = _normalise_max(cube)
    return TripletDataset(
        _pan_from_bands(cube, slice(15, 55)),
        _chw(cube),
        None,
        ratio=ratio,
        patch_size=patch,
        stride=stride,
        augment=augment,
        derive_lr_hsi=True,
    )


def _load_pavia(root: Path, split: str, ratio: int, augment: bool):
    filename = "Train_Pavia.h5" if split == "train" else "Test_Pavia.h5"
    with h5py.File(root / filename, "r") as handle:
        target = np.asarray(handle["GT"], dtype=np.float32)
        lr_hsi = np.asarray(handle["MS"], dtype=np.float32)
        pan = np.asarray(handle["PAN"], dtype=np.float32)
    return TripletDataset(
        pan,
        target,
        lr_hsi,
        ratio=ratio,
        patch_size=None,
        stride=64,
        augment=augment and split == "train",
    )


def _load_xiongan(root: Path, split: str, ratio: int, patch: int, stride: int, augment: bool):
    cube = np.asarray(np.load(root / "xantrain.npy"), dtype=np.float32)
    if split == "val":
        cube = _normalise_max(cube)
        cube = cube[1750:2262, 650:1162]
        target = _chw(cube)
        return TripletDataset(
            _pan_from_bands(cube, slice(0, 80)),
            target,
            gaussian_downsample(target, ratio),
            ratio=ratio,
            patch_size=None,
            stride=stride,
            augment=False,
        )

    cube = cube.copy()
    cube[1750:2262, 650:1162] = cube[2000:2512, 0:512]
    cube = _normalise_max(cube)
    return TripletDataset(
        _pan_from_bands(cube, slice(0, 80)),
        _chw(cube),
        None,
        ratio=ratio,
        patch_size=patch,
        stride=stride,
        augment=augment,
        derive_lr_hsi=True,
    )


def _load_liaoning(
    root: Path,
    name: str,
    split: str,
    ratio: int,
    patch: int,
    stride: int,
    augment: bool,
):
    filename = "LN01.mat" if name == "ln1" else "LN02.mat"
    data = sio.loadmat(root / filename)
    hsi = np.asarray(data["HSI"], dtype=np.float32)
    pan = np.asarray(data["PAN"], dtype=np.float32)
    if split == "full":
        return FullSceneDataset(pan[None, :, :], _chw(hsi))

    target_height, target_width = hsi.shape[:2]
    pan_wald = cv2.resize(pan, (target_width, target_height), interpolation=cv2.INTER_AREA)
    hsi_wald = _resize_cube(hsi, target_height // ratio, target_width // ratio)
    return TripletDataset(
        pan_wald[None, :, :],
        _chw(hsi),
        _chw(hsi_wald),
        ratio=ratio,
        patch_size=patch if split == "train" else None,
        stride=stride,
        augment=augment and split == "train",
    )


def build_dataset(
    name: str,
    data_root: str,
    split: str,
    *,
    ratio: Optional[int] = None,
    patch_size: Optional[int] = None,
    stride: Optional[int] = None,
    augment: Optional[bool] = None,
) -> Dataset:
    """Build one of the six datasets from the original experiment file layout."""
    name = canonical_dataset_name(name)
    split = split.lower()
    if split not in {"train", "val", "full"}:
        raise ValueError("split must be 'train', 'val', or 'full'")
    if split == "full" and name not in {"ln1", "ln2"}:
        raise ValueError("The full split is only defined for Liaoning-1/2")

    defaults = DATASET_DEFAULTS[name]
    ratio = defaults.ratio if ratio is None else int(ratio)
    patch = defaults.patch_size if patch_size is None else int(patch_size)
    stride = defaults.stride if stride is None else int(stride)
    augment = split == "train" if augment is None else bool(augment)
    root = Path(data_root).expanduser()

    if name == "chikusei":
        return _load_chikusei(root, split, ratio, patch, stride, augment)
    if name == "indian":
        return _load_indian(root, split, ratio, patch, stride, augment)
    if name == "pavia":
        return _load_pavia(root, split, ratio, augment)
    if name == "xiongan":
        return _load_xiongan(root, split, ratio, patch, stride, augment)
    return _load_liaoning(root, name, split, ratio, patch, stride, augment)
