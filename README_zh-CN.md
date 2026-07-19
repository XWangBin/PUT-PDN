# PUT-PDN

论文 **“Physics-Constrained Fusion and Prior-Guided Classification Framework for Remote Sensing Perception”** 的官方代码整理版本。论文已发表于 *Information Fusion*（2026），文章号 104624：[https://doi.org/10.1016/j.inffus.2026.104624](https://doi.org/10.1016/j.inffus.2026.104624)。

[English README](README.md)

![本文框架总体架构](assets/figure1_architecture.png)

**图 1.** 本文提出的物理约束融合与先验引导分类框架。

## 方法简介

**Physics-Constrained Unfolding Transformer（PUT）** 从低空间分辨率 HSI 和高空间分辨率 PAN 中重建高空间分辨率 HSI，主要包括：

- 基于传感器观测模型的物理约束展开过程；
- 用于高效数据保真更新的截断高阶 Neumann 展开；
- 用于 PAN 条件空间-光谱协同表征学习的 Spatial-Spectral Synergy Block；
- 可学习的空间与光谱退化算子；
- PAN 和 LR-HSI 两个观测域的一致性监督。

当前仓库包含完整的 PUT 融合实现。已审计的融合实验目录中没有完整可运行的 PDN 分类源码，因此本次开源准备暂未包含 PDN。

## 论文结果复测

已使用当前公开版 PUT 代码、论文 checkpoint 和原始测试数据，在 NVIDIA Tesla V100S GPU 上重新计算论文指标。指标后端与原实验保持一致，采用 `imgvision==0.1.7.3`。

| 数据集 | 阶段数 | PSNR | SSIM | SAM | ERGAS | RMSE | 与论文一致 |
|---|---:|---:|---:|---:|---:|---:|:---:|
| Indian | 3 | 60.01 | 0.9974 | 1.1906 | 0.5911 | 0.0020 | 是 |
| Pavia | 3 | 36.12 | 0.9742 | 4.2595 | 3.2078 | 0.0206 | 是 |
| Chikusei | 5 | 45.68 | 0.9914 | 2.7261 | 4.1207 | 0.0057 | 是 |
| Xiongan | 5 | 47.75 | 0.9961 | 0.9653 | 0.6360 | 0.0049 | 是 |

全部 30 个复测指标单元格在四舍五入至论文展示精度后均与已发表表格一致。完整精度结果和复测环境见 [`results/reproduction_20260719.json`](results/reproduction_20260719.json)。PUT 与 FusionNet、PanFormer、LGPConv、PMACNet、WFANet、GPPNN、LGTEUN、DISPNet 和 SSUNNet 的完整对比见 [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md)。

## 安装

已验证环境为 Python 3.9 和 PyTorch 2.1。请先安装与 CUDA 匹配的 PyTorch，再安装本项目：

```bash
conda create -n put-pdn python=3.9 -y
conda activate put-pdn
# 从 pytorch.org 安装与 CUDA 匹配的 PyTorch。
pip install -e .
```

完整依赖版本见 [`environment.yml`](environment.yml)。

## 数据

本仓库不分发数据。请按照 [`docs/DATASETS.md`](docs/DATASETS.md) 准备文件，并通过 `--data-root` 指定对应数据目录。

| 参数名 | 波段数 | 训练轮数 | 默认阶段数 | 实验方式 | 数据来源与论文引用 |
|---|---:|---:|---:|---|---|
| `indian` | 220 | 100 | 3 | 仿真 | [Purdue PURR 数据](https://doi.org/10.4231/R7RX991C) [43] |
| `pavia` | 102 | 100 | 3 | 仿真/预处理 HDF5 | [Pavia Centre 场景](https://www.ehu.eus/ccwintco/index.php?redirect=no&title=Hyperspectral_Remote_Sensing_Scenes#Pavia_Centre_and_University) · [文献 [44]](https://doi.org/10.1109/JSTARS.2022.3220974) |
| `chikusei` | 128 | 300 | 5 | 仿真 | [作者下载页及引用说明](https://naotoyokoya.com/Download.html) [45] |
| `xiongan` | 93 | 300 | 5 | 仿真 | [数据论文与来源页](https://doi.org/10.11834/jrs.20209065) [46] |
| `ln1` | 166 | 100 | 3 | Wald 与真实全图 | [LISA 数据发布页](https://drive.google.com/drive/folders/1JLCCB6ld5R49HDLN5SsMISx1d0fuqRjO) · [文献 [47]](https://doi.org/10.1109/JSTARS.2025.3562278) |
| `ln2` | 144 | 100 | 3 | Wald 与真实全图 | [LISA 数据发布页](https://drive.google.com/drive/folders/1JLCCB6ld5R49HDLN5SsMISx1d0fuqRjO) · [文献 [47]](https://doi.org/10.1109/JSTARS.2025.3562278) |

论文以三阶段为主要设置，同时报告五阶段结果。可以通过 `--stage` 指定展开深度。

## 训练

默认训练设置与论文一致：HR patch 为 64×64、空间倍率 4、batch size 10、每轮 100 次迭代、Adam、初始学习率 `2e-4`，并余弦退火至 `1e-6`。

```bash
python train_put.py --dataset indian --data-root /path/to/Indian --output runs
python train_put.py --dataset pavia --data-root /path/to/Pavia --output runs
python train_put.py --dataset chikusei --data-root /path/to/Chikusei --output runs
python train_put.py --dataset xiongan --data-root /path/to/Xiongan --output runs
python train_put.py --dataset ln1 --data-root /path/to/Liaoning-1 --output runs
python train_put.py --dataset ln2 --data-root /path/to/Liaoning-2 --output runs
```

训练目标为：

```text
L = L1(预测HR-HSI, 真实HR-HSI)
  + 0.1 L1(重建PAN, PAN)
  + 0.1 L1(重建LR-HSI, LR-HSI)
```

checkpoint 和 JSONL 日志保存于 `runs/<dataset>/<stage>stages/`，可用 `--resume` 恢复训练。

## 评估

评估仿真或 Wald 协议划分：

```bash
python test_put.py \
  --dataset chikusei \
  --data-root /path/to/Chikusei \
  --checkpoint /path/to/put_checkpoint.pth \
  --stage 5 --split val
```

只计算指标而不保存预测图时可增加 `--metrics-only`。Liaoning 真实场景可采用重叠分块推理：

```bash
python test_put.py \
  --dataset ln1 \
  --data-root /path/to/Liaoning-1 \
  --checkpoint /path/to/put_checkpoint.pth \
  --stage 3 --split full --tile-size 128 --overlap 16
```

预测结果保存为 MATLAB 文件，其中变量 `HSI` 按 H×W×B 排列。验证时将 PSNR、SSIM、SAM、ERGAS 和 RMSE 写入 `metrics.json`。

## 目录结构

```text
PUT-PDN/
├── assets/figure1_architecture.png
├── docs/
│   ├── BENCHMARKS.md
│   ├── DATASETS.md
│   └── IMPLEMENTATION_NOTES.md
├── results/reproduction_20260719.json
├── src/put_pdn/
│   ├── models/put.py
│   ├── data.py
│   ├── train.py
│   ├── evaluate.py
│   ├── inference.py
│   └── metrics.py
├── tools/
├── train_put.py
└── test_put.py
```

## 引用

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

代码整理范围见 [`docs/IMPLEMENTATION_NOTES.md`](docs/IMPLEMENTATION_NOTES.md)。

## 数据集致谢

衷心感谢论文所用全部数据集的创建者与提供方：感谢 Baumgardner、Biehl、Landgrebe 及 Purdue University Research Repository 提供 Indian 数据；感谢 Paolo Gamba 教授及 University of Pavia 提供 Pavia 场景；感谢岑奕等作者及参与单位提供雄安新区马蹄湾村数据集；特别感谢 LISA 团队与 Jia 等作者提供 **Liaoning-1 和 Liaoning-2 两个数据集**。使用这些数据时，请引用上表链接的原始文献。

按照 Chikusei 数据提供方的要求，保留其标准英文致谢语：“The authors gratefully acknowledge Space Application Laboratory, Department of Advanced Interdisciplinary Studies, the University of Tokyo for providing the hyperspectral data.” 同时感谢 Naoto Yokoya 博士和 Akira Iwasaki 教授公开该数据集；其许可为 [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/)。
