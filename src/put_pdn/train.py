"""Command-line training entry point for PUT fusion."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict

import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import DATASET_DEFAULTS, build_dataset, canonical_dataset_name
from .metrics import fusion_metrics
from .models import PUT
from .utils import append_jsonl, load_checkpoint, resolve_device, save_checkpoint, seed_everything


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the PUT hyperspectral fusion network")
    parser.add_argument("--dataset", required=True, choices=tuple(DATASET_DEFAULTS))
    parser.add_argument("--data-root", required=True, help="Directory containing this dataset's source files")
    parser.add_argument("--output", default="runs", help="Checkpoint/log root")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--stage", type=int, default=None, help="Unfolding stages; dataset default if omitted")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--steps-per-epoch", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--min-learning-rate", type=float, default=1e-6)
    parser.add_argument("--ratio", type=int, default=None)
    parser.add_argument("--patch-size", type=int, default=None)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--no-augmentation", action="store_true")
    return parser


@torch.no_grad()
def _validate(model: PUT, loader: DataLoader, device: torch.device, ratio: int) -> Dict[str, float]:
    model.eval()
    records = []
    for batch in loader:
        pan = batch["pan"].to(device, non_blocking=True)
        lr_hsi = batch["lr_hsi"].to(device, non_blocking=True)
        prediction = model(lr_hsi, pan)[0]
        records.append(fusion_metrics(batch["target"], prediction, ratio=ratio))
    return {key: float(sum(row[key] for row in records) / len(records)) for key in records[0]}


def main(argv=None) -> None:
    args = _parser().parse_args(argv)
    name = canonical_dataset_name(args.dataset)
    defaults = DATASET_DEFAULTS[name]
    stage = defaults.stages if args.stage is None else args.stage
    epochs = defaults.epochs if args.epochs is None else args.epochs
    ratio = defaults.ratio if args.ratio is None else args.ratio
    patch_size = defaults.patch_size if args.patch_size is None else args.patch_size
    stride = defaults.stride if args.stride is None else args.stride
    if epochs < 1 or args.steps_per_epoch < 1 or args.batch_size < 1:
        raise ValueError("epochs, steps-per-epoch, and batch-size must be positive")

    seed_everything(args.seed)
    device = resolve_device(args.device)
    pin_memory = device.type == "cuda"
    train_data = build_dataset(
        name,
        args.data_root,
        "train",
        ratio=ratio,
        patch_size=patch_size,
        stride=stride,
        augment=not args.no_augmentation,
    )
    val_data = build_dataset(name, args.data_root, "val", ratio=ratio)
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=0, pin_memory=pin_memory)

    model_config = {
        "ratio": ratio,
        "in_channels": 1,
        "out_channels": defaults.bands,
        "stage": stage,
        "channels": defaults.bands,
        "neumann_terms": 3,
    }
    model = PUT(**model_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999))
    total_iterations = epochs * args.steps_per_epoch
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_iterations, eta_min=args.min_learning_rate
    )
    criterion = nn.L1Loss()
    start_epoch, iteration, best_psnr = 0, 0, float("-inf")
    if args.resume:
        payload = load_checkpoint(args.resume, model, device=device, optimizer=optimizer)
        start_epoch = int(payload.get("epoch", 0))
        iteration = int(payload.get("iteration", start_epoch * args.steps_per_epoch))
        best_psnr = float(payload.get("best_psnr", best_psnr))
        if "scheduler" in payload:
            scheduler.load_state_dict(payload["scheduler"])

    run_dir = Path(args.output) / name / f"{stage}stages"
    log_path = run_dir / "metrics.jsonl"
    print(f"Dataset={name} samples={len(train_data)} device={device} output={run_dir}")
    for epoch in range(start_epoch, epochs):
        model.train()
        running_loss = 0.0
        loader_iterator = iter(train_loader)
        started = time.time()
        for step in range(args.steps_per_epoch):
            try:
                batch = next(loader_iterator)
            except StopIteration:
                loader_iterator = iter(train_loader)
                batch = next(loader_iterator)
            pan = batch["pan"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            lr_hsi = batch["lr_hsi"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            prediction, reconstructed_pan, reconstructed_lr_hsi = model(lr_hsi, pan)
            loss = (
                criterion(prediction, target)
                + 0.1 * criterion(reconstructed_pan, pan)
                + 0.1 * criterion(reconstructed_lr_hsi, lr_hsi)
            )
            loss.backward()
            optimizer.step()
            scheduler.step()
            running_loss += float(loss.detach())
            iteration += 1
            interval = max(1, args.steps_per_epoch // 10)
            if (step + 1) % interval == 0:
                print(
                    f"epoch={epoch + 1}/{epochs} step={step + 1}/{args.steps_per_epoch} "
                    f"loss={running_loss / (step + 1):.6f} lr={optimizer.param_groups[0]['lr']:.8f}"
                )

        validation = _validate(model, val_loader, device, ratio)
        train_loss = running_loss / args.steps_per_epoch
        record = {
            "epoch": epoch + 1,
            "iteration": iteration,
            "seconds": time.time() - started,
            "train_loss": train_loss,
            "learning_rate": optimizer.param_groups[0]["lr"],
            **validation,
        }
        append_jsonl(log_path, record)
        improved = validation["psnr"] > best_psnr
        best_psnr = max(best_psnr, validation["psnr"])
        save_checkpoint(
            run_dir / "latest.pth",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch + 1,
            iteration=iteration,
            best_psnr=best_psnr,
            model_config=model_config,
        )
        if improved:
            save_checkpoint(
                run_dir / "best.pth",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch + 1,
                iteration=iteration,
                best_psnr=best_psnr,
                model_config=model_config,
            )
        print(
            f"validation epoch={epoch + 1}: PSNR={validation['psnr']:.4f}, "
            f"SSIM={validation['ssim']:.4f}, SAM={validation['sam']:.4f}, "
            f"ERGAS={validation['ergas']:.4f}, RMSE={validation['rmse']:.6f}"
        )


if __name__ == "__main__":
    main()
