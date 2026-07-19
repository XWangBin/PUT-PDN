# Dataset preparation

The code does not redistribute data. `--data-root` must point to one directory with the exact files below.

## Expected layouts

```text
Chikusei/
├── Chikusei.mat       # HDF5 dataset key: chikusei
└── GT.npy             # held-out 512×512×128 test cube

Indian/
└── 19920612_AVIRIS_IndianPine_NS-line.tif

Pavia/
├── Train_Pavia.h5     # keys: GT, MS, PAN; NCHW arrays
└── Test_Pavia.h5      # keys: GT, MS, PAN; NCHW arrays

Xiongan/
└── xantrain.npy

Liaoning-1/
└── LN01.mat           # keys: HSI (H×W×166), PAN (HR H×W)

Liaoning-2/
└── LN02.mat           # keys: HSI (H×W×144), PAN (HR H×W)
```

## Simulated-data protocol

The loaders reproduce the operations in the six archived experiment directories:

- spatial ratio: 4;
- HR patch: 64×64;
- patch stride: 24 (Pavia uses the pre-generated HDF5 patches);
- LR-HSI simulation: Gaussian blur with kernel `2r+1`, sigma `0.666r`, then ratio-aligned decimation;
- PAN simulation: mean over bands 60–79 for Chikusei, 15–54 for Indian, and 0–79 for Xiongan;
- Chikusei, Indian, and Xiongan cubes are divided by their global maximum; prepared Pavia and Liaoning arrays retain their stored scale.

Original held-out areas:

- Indian: rows 1700–2211, columns 80–591;
- Chikusei: the separate `GT.npy` cube;
- Xiongan: rows 1750–2261, columns 650–1161;
- Pavia: `Test_Pavia.h5`.

As in the source loaders, the Indian and Xiongan test regions are replaced in the training scene before patch extraction; the Chikusei training exclusion is also preserved. Review this protocol if preparing a new benchmark split.

## Liaoning protocol

For training/validation, the stored HSI is treated as the Wald-protocol HR target. PAN is resized to the HSI size, and HSI is downsampled by 4. For `--split full`, the native stored HSI is the LR-HSI observation and the native PAN is the HR-PAN observation; no reference metric is computed.

## Paper training settings

| Dataset | Epochs | Iterations/epoch | Batch | Initial LR |
|---|---:|---:|---:|---:|
| Indian | 100 | 100 | 10 | 2e-4 |
| Pavia | 100 | 100 | 10 | 2e-4 |
| Chikusei | 300 | 100 | 10 | 2e-4 |
| Xiongan | 300 | 100 | 10 | 2e-4 |
| Liaoning-1 | 100 | 100 | 10 | 2e-4 |
| Liaoning-2 | 100 | 100 | 10 | 2e-4 |

The public loader intentionally validates shapes and alignment more strictly than the research scripts. This makes malformed files fail early instead of silently producing shifted LR/HR pairs.
