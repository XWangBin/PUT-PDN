"""Verify that a PUT checkpoint strictly matches the public model."""

import argparse

import torch

from put_pdn.models import PUT
from put_pdn.utils import load_checkpoint, resolve_device


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--bands", type=int, required=True)
    parser.add_argument("--stage", type=int, required=True)
    parser.add_argument("--ratio", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--forward", action="store_true", help="Also run a 16x16 synthetic forward pass")
    args = parser.parse_args()

    device = resolve_device(args.device)
    model = PUT(
        ratio=args.ratio,
        in_channels=1,
        out_channels=args.bands,
        stage=args.stage,
        channels=args.bands,
    ).to(device)
    load_checkpoint(args.checkpoint, model, device=device, strict=True)
    print(
        f"CHECKPOINT_OK bands={args.bands} stage={args.stage} "
        f"parameters={sum(parameter.numel() for parameter in model.parameters())}"
    )
    if args.forward:
        model.eval()
        with torch.no_grad():
            lr_hsi = torch.rand(1, args.bands, 4, 4, device=device)
            pan = torch.rand(1, 1, 16, 16, device=device)
            outputs = model(lr_hsi, pan)
        print("FORWARD_OK", *(tuple(output.shape) for output in outputs))


if __name__ == "__main__":
    main()
