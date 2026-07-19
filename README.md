# PUT-PDN

Official-code preparation for **“Physics-Constrained Fusion and Prior-Guided Classification Framework for Remote Sensing Perception”**, published in *Information Fusion* (2026), article 104624. [https://doi.org/10.1016/j.inffus.2026.104624](https://doi.org/10.1016/j.inffus.2026.104624)

[中文说明](README_zh-CN.md)

## Release status

This repository currently provides the cleaned and consolidated **Physics-Constrained Unfolding Transformer (PUT)** code for HSI–PAN fusion on Indian, Pavia, Chikusei, Xiongan, Liaoning-1, and Liaoning-2.

The six archived experiment directories did **not** contain a runnable implementation of the paper's **Physics-Aware Dual-Prior Classification Network (PDN)**. PDN is therefore not included in this preparation, and this repository should not yet be advertised as a complete PUT–PDN reproduction. Add and verify the authors' final PDN source before the public release.

## Method

PUT unfolds the HSI–PAN observation model into alternating learned-prior and data-fidelity updates. Its main components are:

- a Spatial–Spectral Synergy Block (SSSB) that couples PAN-conditioned spatial information with HSI spectral information;
- learnable spatial and spectral degradation operators;
- a truncated Neumann-style recurrence in each physics-constrained data-fidelity block;
- observation-consistency supervision for both reconstructed PAN and LR-HSI.

The fusion objective used by the released trainer is

```text
L = L1(HR-HSI_hat, HR-HSI)
  + 0.1 L1(PAN_hat, PAN)
  + 0.1 L1(LR-HSI_hat, LR-HSI).
```

## Repository layout

```text
PUT-PDN/
├── src/put_pdn/
│   ├── models/put.py       # cleaned PUT (original name: Net0512/ZAB0512)
│   ├── data.py             # unified loaders for all six datasets
│   ├── train.py            # training CLI
│   ├── evaluate.py         # validation and full-scene inference CLI
│   ├── inference.py        # overlap-tiled inference
│   └── metrics.py          # PSNR, SSIM, SAM, ERGAS, RMSE
├── legacy/zab2.py          # traceable earlier architecture, not the final trained PUT
├── docs/DATASETS.md        # exact data layouts and preprocessing
├── docs/SOURCE_PROVENANCE.md
├── train_put.py
└── test_put.py
```

## Installation

Python 3.9 and PyTorch 2.1 were used in the archived server environment. Install the PyTorch build appropriate for your CUDA version first, then install the project:

```bash
conda create -n put-pdn python=3.9 -y
conda activate put-pdn
# Install the matching PyTorch/CUDA build from https://pytorch.org/get-started/locally/
pip install -e .
```

For a record of the original package versions, see [`environment.yml`](environment.yml).

## Data

Datasets are not redistributed. Arrange the original files as described in [`docs/DATASETS.md`](docs/DATASETS.md), then pass that dataset directory with `--data-root`.

| CLI name | Bands | Paper epochs | Default stages | Split type |
|---|---:|---:|---:|---|
| `indian` | 220 | 100 | 3 | simulated |
| `pavia` | 102 | 100 | 3 | simulated/prepared HDF5 |
| `chikusei` | 128 | 300 | 5 | simulated |
| `xiongan` | 93 | 300 | 5 | simulated |
| `ln1` | 166 | 100 | 3 | Wald + real full scene |
| `ln2` | 144 | 100 | 3 | Wald + real full scene |

The paper uses three stages as the primary setting and also reports five-stage results. Dataset defaults reflect the reported observation that three stages work better on the smaller Indian/Pavia datasets and five stages work better on the larger Chikusei/Xiongan datasets. Override this with `--stage` for an exact ablation or checkpoint.

## Training

The defaults reproduce the paper protocol: 64×64 HR patches, 4× spatial ratio, batch size 10, 100 iterations per epoch, Adam, initial learning rate `2e-4`, and cosine decay to `1e-6`.

```bash
python train_put.py --dataset indian --data-root /path/to/Indian --output runs
python train_put.py --dataset pavia --data-root /path/to/Pavia --output runs
python train_put.py --dataset chikusei --data-root /path/to/Chikusei --output runs
python train_put.py --dataset xiongan --data-root /path/to/Xiongan --output runs
python train_put.py --dataset ln1 --data-root /path/to/Liaoning-1 --output runs
python train_put.py --dataset ln2 --data-root /path/to/Liaoning-2 --output runs
```

Checkpoints and JSONL logs are written under `runs/<dataset>/<stage>stages/`. Resume with `--resume runs/.../latest.pth`.

## Evaluation and inference

Evaluate a simulated/Wald split:

```bash
python test_put.py \
  --dataset chikusei \
  --data-root /path/to/Chikusei \
  --checkpoint runs/chikusei/5stages/best.pth \
  --split val
```

Run overlap-tiled inference on a real Liaoning scene:

```bash
python test_put.py \
  --dataset ln1 \
  --data-root /path/to/Liaoning-1 \
  --checkpoint runs/ln1/3stages/best.pth \
  --split full --tile-size 128 --overlap 16
```

Predictions are saved as MATLAB files with an `HSI` variable in H×W×B layout. Full-reference validation also writes `metrics.json`. By default, metrics use the same `imgvision==0.1.7.3` backend as the archived experiment scripts; a documented NumPy/scikit-image implementation is retained as a fallback.

### Original checkpoints

The final experiment scripts called this network `zab0512` and instantiated `architecture/Net0512.py`; the public class is now named `PUT`. Parameter names were retained, and `ZAB0512` remains an alias, so original checkpoints can be loaded through the provided checkpoint helper. The `--stage` value must match the checkpoint depth. One Chikusei variant contains weights for an inconsistent, inactive `lms` initialiser branch; the loader reports and discards only those six dead-branch tensors, then applies strict matching to every active PUT tensor.

The file named `zab2.py` is byte-identical across all six server directories, but the final training/test scripts and available checkpoints target `Net0512.py`. It is preserved under `legacy/` for provenance rather than presented as the published final implementation.

## Citation

```bibtex
@article{wang2026putpdn,
  title   = {Physics-Constrained Fusion and Prior-Guided Classification Framework for Remote Sensing Perception},
  author  = {Wang, Bin and Xiong, Xingchuang and Lian, Yusheng and Yu, Kun and Liu, Zilong},
  journal = {Information Fusion},
  year    = {2026},
  pages   = {104624},
  doi     = {10.1016/j.inffus.2026.104624}
}
```

## Before making the repository public

- add and validate the final PDN implementation, or rename/scope the repository explicitly as PUT-only;
- publish checkpoint files only after confirming distribution rights;
- add dataset download links and acknowledge each dataset's own licence;
- choose and add a software licence. No software licence is granted by this preparation alone.

See [`docs/SOURCE_PROVENANCE.md`](docs/SOURCE_PROVENANCE.md) for the source audit and consolidation decisions.
