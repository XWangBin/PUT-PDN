# Implementation notes

Audit date: 2026-07-19.

## Consolidation

The six dataset-specific experiment directories under `/home/user/wangbin_118/` were audited in read-only mode and consolidated into one maintained PUT implementation. Dataset constants, hard-coded paths, device IDs, and duplicated experiment wrappers were replaced with explicit command-line arguments and shared loaders.

The public model keeps the trained active-parameter layout required by the released PUT checkpoints. One Chikusei checkpoint also contains six tensors from an inactive experimental initialisation branch that was not used by the final `model(lr_hsi, pan)` workflow. The checkpoint loader reports and discards only those inactive tensors, then performs strict matching for every active PUT parameter.

## Included

- one dataset-agnostic PUT model;
- loaders for Indian, Pavia, Chikusei, Xiongan, Liaoning-1, and Liaoning-2;
- the paper training objective and schedule;
- direct and overlap-tiled inference;
- the paper-compatible `imgvision==0.1.7.3` metric backend;
- reproducibility and checkpoint-audit tools.

## Excluded

- duplicated comparison-method implementations;
- absolute server paths and fixed GPU IDs;
- logs, caches, generated result images, and dataset binaries;
- checkpoints, pending an explicit public-release decision;
- the classification implementation, because a complete runnable PDN source was not present in the audited fusion directories.

The source experiment directories were not modified.
