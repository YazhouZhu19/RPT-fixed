
<div align="center">

<h1>Few-Shot Medical Image Segmentation via a Region-enhanced Prototypical Transformer </h1>

</div>

<p align="center"><img width="95%" src="./overview.png" />


## Abstract
Automated segmentation of large volumes of medical images is often plagued by the limited availability of fully annotated data and the diversity of organ surface properties resulting from the use of different acquisition protocols for different patients. In this paper, we introduce a more promising few-shot learning-based method named Region-enhanced Prototypical Transformer (RPT) to mitigate the effects of large intra-class diversity/bias. First, a subdivision strategy is introduced to produce a collection of regional prototypes from the foreground of the support prototype. Second, a self-selection mechanism is proposed to incorporate into the Bias-alleviated Transformer (BaT) block to suppress or remove interferences present in the query prototype and regional support prototypes. By stacking BaT blocks, the proposed RPT can iteratively optimize the generated regional prototypes and finally produce rectified and more accurate global prototypes for Few-Shot Medical Image Segmentation (FSMS). Extensive experiments are conducted on three publicly available medical image datasets, and the obtained results show consistent improvements compared to state-of-the-art FSMS methods.   


# Getting started

### Dependencies
The reference environment uses Python 3.9. Install the Python dependencies with:
```
python3 -m pip install -r requirements.txt
```

The CHAOS DICOM conversion script also requires `dcm2nii` to be installed on the system.

Pre-processing is performed according to [Ouyang et al.](https://github.com/cheng-01037/Self-supervised-Fewshot-Medical-Image-Segmentation/tree/2f2a22b74890cb9ad5e56ac234ea02b9f1c7a535) and we follow the procedure on their github repository.


The trained models can be downloaded by:
1) [trained models for CHAOS under Setting 1](https://drive.google.com/drive/folders/1gp2Hp4EPBOKwIbVN4l3QyfWeAN8l4jJj?usp=drive_link)
2) [trained models for CHAOS under Setting 2](https://drive.google.com/drive/folders/1RQ0B0XQfOIwoO-7R2sUG7h7dea6v4TJa?usp=drive_link)
3) [trained models for SABS under Setting 1](https://drive.google.com/drive/folders/1xXK8_1fQVQyRoL1N49RN7ZW3H10E-Y5-?usp=drive_link)
4) [trained models for SABS under Setting 2](https://drive.google.com/drive/folders/1EZamwmnh8DkJ51J3VJbC0Mn2vhdE-cGt?usp=drive_link)
5) [trained models for CMR](https://drive.google.com/drive/folders/1czW-1mMOdaouI5PBPBNXI8cLbt9jJ2xq?usp=drive_link)



The pre-processed data and supervoxels can be downloaded by:
1) [Pre-processed CHAOS-T2 data and supervoxels](https://drive.google.com/drive/folders/1elxzn67Hhe0m1PvjjwLGls6QbkIQr1m1?usp=share_link)
2) [Pre-processed SABS data and supervoxels](https://drive.google.com/drive/folders/1pgm9sPE6ihqa2OuaiSz7X8QhXKkoybv5?usp=share_link)
3) [Pre-processed CMR data and supervoxels](https://drive.google.com/drive/folders/1aaU5KQiKOZelfVOpQxxfZNXKNkhrcvY2?usp=share_link)
### Training
1. From `data/supervoxels`, compile the extensions with `python setup.py build_ext --inplace`, then run `python generate_supervoxels.py --dataset SABS` (or `CHAOST2`/`CMR`).
2. Download the [DeepLabV3 ResNet-101 weights](https://download.pytorch.org/models/deeplabv3_resnet101_coco-586e9e4e.pth) into `checkpoints/`. A different local path can be supplied as Sacred's `pretrained_weights` setting.
3. Run one of the concrete launchers in `scripts/`, for example `bash scripts/train_CHAOST2_setting1.sh`.

The current episodic implementation is one-way and one-query. Training supports one or more shots; evaluation uses `batch_size=1` because query volumes have different depths.

### Validation

Run the lightweight regression suite with:

```
python3 -m pip install numpy==1.22.0 pytest
python3 -m pytest -q tests
```

### Inference
Run the matching launcher in `scripts/`, for example `bash scripts/test_CHAOST2_setting1.sh`. The final entry of every fold is reserved as its support case (`supp_idx=-1`). To override the inferred checkpoint path, set `RELOAD_MODEL_PATH` before launching the script.

### Acknowledgement
Our code is based the works: [SSL-ALPNet](https://github.com/cheng-01037/Self-supervised-Fewshot-Medical-Image-Segmentation), [ADNet](https://github.com/sha168/ADNet) and [QNet](https://github.com/ZJLAB-AMMI/Q-Net)

## Citation
```bibtex
@inproceedings{zhu2023few,
  title={Few-Shot Medical Image Segmentation via a Region-Enhanced Prototypical Transformer},
  author={Zhu, Yazhou and Wang, Shidong and Xin, Tong and Zhang, Haofeng},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={271--280},
  year={2023},
  organization={Springer}
}
```
