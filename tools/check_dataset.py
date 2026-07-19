"""Load one dataset split and report the first sample shapes."""

import argparse

from put_pdn.data import build_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--split", default="val", choices=("train", "val", "full"))
    parser.add_argument("--ratio", type=int, default=4)
    args = parser.parse_args()
    dataset = build_dataset(args.dataset, args.data_root, args.split, ratio=args.ratio, augment=False)
    sample = dataset[0]
    shapes = {key: tuple(value.shape) for key, value in sample.items()}
    print(f"DATASET_OK name={args.dataset} split={args.split} samples={len(dataset)} shapes={shapes}")


if __name__ == "__main__":
    main()
