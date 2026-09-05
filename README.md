<div align="center">

<h1>RPT</h1>

<h3>Few-Shot Medical Image Segmentation via a<br />Region-Enhanced Prototypical Transformer</h3>

<p><strong>MICCAI 2023</strong> · PyTorch implementation</p>

<p>
  <a href="https://github.com/YazhouZhu19/RPT-fixed/actions/workflows/ci.yml"><img src="https://github.com/YazhouZhu19/RPT-fixed/actions/workflows/ci.yml/badge.svg" alt="CI status" /></a>
  <img src="https://img.shields.io/badge/Python-3.9-3776AB?logo=python&logoColor=white" alt="Python 3.9" />
  <img src="https://img.shields.io/badge/PyTorch-1.10.2-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch 1.10.2" />
  <img src="https://img.shields.io/badge/MICCAI-2023-6F42C1" alt="MICCAI 2023" />
</p>

<p>
  <a href="#overview">Overview</a> ·
  <a href="#installation">Installation</a> ·
  <a href="#data-and-checkpoints">Data</a> ·
  <a href="#training">Training</a> ·
  <a href="#evaluation">Evaluation</a> ·
  <a href="#citation">Citation</a>
</p>

</div>

<p align="center">
  <img src="./overview.png" width="94%" alt="Overview of the Region-Enhanced Prototypical Transformer architecture" />
</p>

## Overview

**RPT** is a few-shot medical image segmentation framework designed to reduce the effect of large intra-class variation and support/query bias. It subdivides foreground support features into regional prototypes, filters interference through a self-selection mechanism, and progressively refines the prototypes with stacked **Bias-alleviated Transformer (BaT)** blocks.

The method was evaluated on three medical image datasets spanning abdominal CT, abdominal MRI, and cardiac MRI.

- **Regional prototypes** preserve local semantic detail that may be lost in a single global prototype.
- **Self-selection** suppresses distracting support and query regions before prototype matching.
- **Iterative refinement** produces a more reliable global prototype for final segmentation.

> [!NOTE]
> This is a maintained and hardened version of the [original RPT repository](https://github.com/YazhouZhu19/RPT). The model design is unchanged; the updates focus on protocol correctness, safer data loading, portable execution, reproducible evaluation, tests, and CI.

<details>
<summary><strong>Paper abstract</strong></summary>

Automated segmentation of large volumes of medical images is often limited by the availability of fully annotated data and by organ appearance differences across patients and acquisition protocols. RPT introduces regional foreground prototypes and a self-selection mechanism inside the BaT block to reduce these sources of bias. Stacking BaT blocks iteratively optimizes the regional prototypes and yields a refined global representation for few-shot segmentation. Experiments on three public medical image datasets demonstrate consistent improvements over previous few-shot medical image segmentation methods.

</details>

## What's improved in this version

| Area | Improvements |
| --- | --- |
| Evaluation protocol | Deterministic five-fold ordering, a dedicated final support case, and corrected Setting 2 label exclusion |
| Data integrity | Case-ID pairing for images, labels, and supervoxels; shape validation; finite-value checks; bounded episode sampling |
| Portability | Relative checkpoint paths, CPU/CUDA device handling, configurable pretrained weights, and portable launch scripts |
| Inference | Full-depth NIfTI predictions with source spatial metadata preserved and per-run result files |
| Quality | Regression tests, Python and shell checks, dependency manifests, and GitHub Actions CI |

## Installation

The reference environment uses **Python 3.9**, **PyTorch 1.10.2**, and **torchvision 0.11.3**.

```bash
git clone https://github.com/YazhouZhu19/RPT-fixed.git
cd RPT-fixed

python3.9 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For training, download the [DeepLabV3 ResNet-101 encoder weights](https://download.pytorch.org/models/deeplabv3_resnet101_coco-586e9e4e.pth) and place them at:

```text
checkpoints/deeplabv3_resnet101_coco-586e9e4e.pth
```

You can use another local file by overriding Sacred's `pretrained_weights` setting. The raw CHAOS DICOM conversion workflow additionally requires the `dcm2nii` system command.

## Data and checkpoints

Preprocessing follows the protocol introduced by [Ouyang et al.](https://github.com/cheng-01037/Self-supervised-Fewshot-Medical-Image-Segmentation/tree/2f2a22b74890cb9ad5e56ac234ea02b9f1c7a535). Ready-to-use data and trained RPT checkpoints are available below.

| Dataset | Modality | Preprocessed data + supervoxels | Trained RPT checkpoints |
| --- | --- | --- | --- |
| CHAOS-T2 | Abdominal MRI | [Download](https://drive.google.com/drive/folders/1elxzn67Hhe0m1PvjjwLGls6QbkIQr1m1?usp=share_link) | [Setting 1](https://drive.google.com/drive/folders/1gp2Hp4EPBOKwIbVN4l3QyfWeAN8l4jJj?usp=drive_link) · [Setting 2](https://drive.google.com/drive/folders/1RQ0B0XQfOIwoO-7R2sUG7h7dea6v4TJa?usp=drive_link) |
| SABS | Abdominal CT | [Download](https://drive.google.com/drive/folders/1pgm9sPE6ihqa2OuaiSz7X8QhXKkoybv5?usp=share_link) | [Setting 1](https://drive.google.com/drive/folders/1xXK8_1fQVQyRoL1N49RN7ZW3H10E-Y5-?usp=drive_link) · [Setting 2](https://drive.google.com/drive/folders/1EZamwmnh8DkJ51J3VJbC0Mn2vhdE-cGt?usp=drive_link) |
| CMR | Cardiac MRI | [Download](https://drive.google.com/drive/folders/1aaU5KQiKOZelfVOpQxxfZNXKNkhrcvY2?usp=share_link) | [Download](https://drive.google.com/drive/folders/1czW-1mMOdaouI5PBPBNXI8cLbt9jJ2xq?usp=drive_link) |

Place the extracted files in the following layout. Every image, label, and supervoxel file must share the same numeric case ID.

```text
data/
├── CHAOST2/
│   ├── chaos_MR_T2_normalized/
│   │   ├── image_<case>.nii.gz
│   │   └── label_<case>.nii.gz
│   └── supervoxels_5000/
│       └── superpix-MIDDLE_<case>.nii.gz
├── SABS/
│   ├── sabs_CT_normalized/
│   └── supervoxels_5000/
└── CMR/
    ├── cmr_MR_normalized/
    └── supervoxels_1000/
```

### Generating supervoxels

If you preprocess the original datasets yourself, compile the bundled 3D Felzenszwalb extension and generate supervoxels from the repository root:

```bash
cd data/supervoxels
python setup.py build_ext --inplace
python generate_supervoxels.py --dataset CHAOST2
cd ../..
```

Valid dataset values are `CHAOST2`, `SABS`, and `CMR`. Dataset-specific normalization and slice-index utilities are under `data/<dataset>/`.

## Experimental protocol

- All launchers run **five-fold cross-validation** using folds `0` through `4`.
- **Setting 1** permits training on all available training labels.
- **Setting 2** excludes the evaluation labels from training slices.
- Evaluation reserves the final case in each ordered fold as support; keep `supp_idx=-1`.
- The current episodic implementation supports one way and one query. Training supports one or more shots.
- Evaluation requires `batch_size=1` because query volumes may have different slice counts.

## Training

Run the launcher matching the target dataset and setting. Each script trains all five folds and writes Sacred run metadata, logs, and snapshots below its experiment directory.

| Dataset | Setting 1 | Setting 2 |
| --- | --- | --- |
| CHAOS-T2 | `bash scripts/train_CHAOST2_setting1.sh` | `bash scripts/train_CHAOST2_setting2.sh` |
| SABS | `bash scripts/train_SABS_setting1.sh` | `bash scripts/train_SABS_setting2.sh` |
| CMR | `bash scripts/train_CMR.sh` | — |

For example:

```bash
bash scripts/train_CHAOST2_setting1.sh
```

The default scripts use GPU `0`, 40,000 optimization steps, and checkpoints every 5,000 steps. Adjust the variables at the top of a launcher or invoke Sacred directly when running a custom experiment:

```bash
python train.py with \
  dataset=CHAOST2 \
  eval_fold=0 \
  test_label='[1,2,3,4]' \
  exclude_label=None \
  path.log_dir=./exps_on_CHAOST2_fewshot_setting1
```

## Evaluation

Run the evaluation launcher that matches the dataset and training setting:

| Dataset | Setting 1 | Setting 2 |
| --- | --- | --- |
| CHAOS-T2 | `bash scripts/test_CHAOST2_setting1.sh` | `bash scripts/test_CHAOST2_setting2.sh` |
| SABS | `bash scripts/test_SABS_setting1.sh` | `bash scripts/test_SABS_setting2.sh` |
| CMR | `bash scripts/test_CMR.sh` | — |

The launchers infer the expected checkpoint location. Override it with `RELOAD_MODEL_PATH` when evaluating a downloaded or relocated checkpoint:

```bash
RELOAD_MODEL_PATH=/absolute/path/to/model.pth \
  bash scripts/test_CHAOST2_setting1.sh
```

Evaluation writes the mean Dice score to `results.txt` and stores full-volume predictions under `interm_preds/`. Predictions use NIfTI compression and preserve the spacing, origin, and direction of each source volume.

## Tests and CI

Install the development dependencies and run the regression suite:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q tests
```

GitHub Actions runs lightweight checks on every push and pull request:

- Python syntax compilation and static analysis
- Shell-script syntax validation
- Dataset protocol regression tests

## Repository layout

```text
RPT-fixed/
├── models/          # RPT, encoder, and attention modules
├── dataloaders/     # Episodic train/test datasets and transforms
├── data/            # Dataset preprocessing and supervoxel tools
├── scripts/         # Reproducible train/test launchers
├── tests/           # Lightweight regression tests
├── visualization/   # Prediction visualization utilities
├── config.py        # Sacred experiment configuration
├── train.py         # Training entry point
├── test.py          # Evaluation entry point
└── utils.py         # Metrics and shared utilities
```

## Acknowledgements

This implementation builds on [SSL-ALPNet](https://github.com/cheng-01037/Self-supervised-Fewshot-Medical-Image-Segmentation), [ADNet](https://github.com/sha168/ADNet), and [Q-Net](https://github.com/ZJLAB-AMMI/Q-Net). We thank their authors for making their work available.

## Citation

If you find RPT useful in your research, please cite:

```bibtex
@inproceedings{zhu2023few,
  title        = {Few-Shot Medical Image Segmentation via a Region-Enhanced Prototypical Transformer},
  author       = {Zhu, Yazhou and Wang, Shidong and Xin, Tong and Zhang, Haofeng},
  booktitle    = {International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages        = {271--280},
  year         = {2023},
  organization = {Springer}
}
```

---

<div align="center">
  <sub>Research code for reproducible few-shot medical image segmentation.</sub>
</div>
