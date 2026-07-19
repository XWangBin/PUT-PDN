"""Checkpoint evaluation and Liaoning full-scene inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import DataLoader

from .data import DATASET_DEFAULTS, build_dataset, canonical_dataset_name
from .inference import predict
from .metrics import fusion_metrics
from .models import PUT
from .utils import load_checkpoint, resolve_device


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a PUT checkpoint")
    parser.add_argument("--dataset", required=True, choices=tuple(DATASET_DEFAULTS))
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="results")
    parser.add_argument("--split", choices=("val", "full"), default="val")
    parser.add_argument("--stage", type=int, default=None)
    parser.add_argument("--ratio", type=int, default=None)
    parser.add_argument("--tile-size", type=int, default=0, help="0 performs direct inference")
    parser.add_argument("--overlap", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--metrics-only",
        action="store_true",
        help="Compute metrics without writing prediction MAT files",
    )
    return parser


def main(argv=None) -> None:
    args = _parser().parse_args(argv)
    name = canonical_dataset_name(args.dataset)
    defaults = DATASET_DEFAULTS[name]
    stage = defaults.stages if args.stage is None else args.stage
    ratio = defaults.ratio if args.ratio is None else args.ratio
    if args.split == "full" and name not in {"ln1", "ln2"}:
        raise ValueError("The full split is available only for ln1 and ln2")

    device = resolve_device(args.device)
    model = PUT(
        ratio=ratio,
        in_channels=1,
        out_channels=defaults.bands,
        stage=stage,
        channels=defaults.bands,
    ).to(device)
    load_checkpoint(args.checkpoint, model, device=device)
    dataset = build_dataset(name, args.data_root, args.split, ratio=ratio)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    output_dir = Path(args.output) / name / args.split
    output_dir.mkdir(parents=True, exist_ok=True)

    all_metrics = []
    for index, batch in enumerate(loader):
        lr_hsi = batch["lr_hsi"].to(device)
        pan = batch["pan"].to(device)
        prediction = predict(
            model,
            lr_hsi,
            pan,
            ratio=ratio,
            tile_size=args.tile_size,
            overlap=args.overlap,
        ).detach().float().cpu()
        if not args.metrics_only:
            prediction_hwc = np.moveaxis(prediction[0].numpy(), 0, -1)
            sio.savemat(output_dir / f"prediction_{index:04d}.mat", {"HSI": prediction_hwc})
        if "target" in batch:
            row = fusion_metrics(batch["target"], prediction, ratio=ratio)
            row["sample"] = index
            all_metrics.append(row)
            print(json.dumps(row, ensure_ascii=False))

    if all_metrics:
        metric_names = ("psnr", "ssim", "sam", "ergas", "rmse")
        summary = {
            key: float(sum(float(row[key]) for row in all_metrics) / len(all_metrics))
            for key in metric_names
        }
        with (output_dir / "metrics.json").open("w", encoding="utf-8") as stream:
            json.dump({"mean": summary, "samples": all_metrics}, stream, indent=2, ensure_ascii=False)
        print("Mean:", json.dumps(summary, ensure_ascii=False))
    if args.metrics_only:
        print(f"Saved metrics to {output_dir}")
    else:
        print(f"Saved predictions to {output_dir}")


if __name__ == "__main__":
    main()
