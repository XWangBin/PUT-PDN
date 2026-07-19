"""Verify that rerun metrics match the paper at its displayed precision."""

import json
from pathlib import Path


PRECISION = {"psnr": 2, "ssim": 4, "sam": 4, "ergas": 4, "rmse": 4}


def main() -> None:
    result_path = Path(__file__).resolve().parents[1] / "results" / "reproduction_20260719.json"
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    checked = 0
    failures = []
    for run in payload["runs"]:
        for metric, digits in PRECISION.items():
            actual = round(float(run["reproduced"][metric]), digits)
            expected = float(run["paper"][metric])
            checked += 1
            if actual != expected:
                failures.append(
                    f"{run['dataset']}-{run['stages']} {metric}: {actual} != {expected}"
                )
    if failures:
        raise SystemExit("\n".join(failures))
    print(f"REPRODUCTION_MATCH: {checked}/{checked} displayed metric cells")


if __name__ == "__main__":
    main()
