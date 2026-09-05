"""Generate 3D Felzenszwalb supervoxels for an RPT dataset."""

import argparse
import glob
import os
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy.ndimage import binary_fill_holes
from skimage.measure import label

from felzenszwalb_3d import felzenszwalb_3d


DATASETS = {
    'SABS': ('SABS/sabs_CT_normalized', 5000),
    'CHAOST2': ('CHAOST2/chaos_MR_T2_normalized', 5000),
    'CMR': ('CMR/cmr_MR_normalized', 1000),
}


def foreground_mask_2d(image, threshold):
    mask = image > threshold
    components = label(mask)
    if components.max() == 0:
        return np.zeros_like(mask, dtype=bool)
    largest = components == np.argmax(np.bincount(components.flat)[1:]) + 1
    return binary_fill_holes(largest)


def mask_supervoxels(segmentation, foreground_mask):
    segmentation = segmentation.copy()
    segmentation[segmentation == 0] = segmentation.max() + 1
    segmentation = segmentation.astype(np.int32, copy=False)
    segmentation[~foreground_mask] = 0
    return segmentation


def generate(dataset, n_supervoxels=None, mode='MIDDLE', foreground_threshold=10):
    relative_input, default_n_supervoxels = DATASETS[dataset]
    if n_supervoxels is None:
        n_supervoxels = default_n_supervoxels

    data_root = Path(__file__).resolve().parents[1]
    input_dir = data_root / relative_input
    output_dir = data_root / dataset / f'supervoxels_{n_supervoxels}'
    output_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(
        glob.glob(os.path.join(input_dir, 'image*')),
        key=lambda path: int(path.rsplit('_', 1)[-1].split('.nii.gz')[0]),
    )
    if not images:
        raise ValueError(f'No normalized images found in {input_dir}')

    for image_path in images:
        source_image = sitk.ReadImage(image_path)
        image = sitk.GetArrayFromImage(source_image)
        intensity_range = np.ptp(image)
        if intensity_range == 0:
            raise ValueError(f'Cannot generate supervoxels for constant image {image_path}')
        image = 255 * (image - image.min()) / intensity_range

        spacing_xyz = source_image.GetSpacing()
        spacing_zyx = (spacing_xyz[2], spacing_xyz[1], spacing_xyz[0])
        segmentation = felzenszwalb_3d(
            image, min_size=n_supervoxels, sigma=0, spacing=spacing_zyx
        )

        foreground = np.stack([
            foreground_mask_2d(image_slice, foreground_threshold)
            for image_slice in image
        ])
        segmentation = mask_supervoxels(segmentation, foreground)

        output_image = sitk.GetImageFromArray(segmentation)
        output_image.CopyInformation(source_image)
        case_id = os.path.basename(image_path).rsplit('_', 1)[-1].split('.nii.gz')[0]
        output_path = output_dir / f'superpix-{mode}_{case_id}.nii.gz'
        sitk.WriteImage(output_image, str(output_path), True)
        print(f'Case {case_id} saved to {output_path}')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=DATASETS, default='SABS')
    parser.add_argument('--n-supervoxels', type=int, default=None)
    parser.add_argument('--mode', default='MIDDLE')
    parser.add_argument('--foreground-threshold', type=float, default=10)
    return parser.parse_args()


if __name__ == '__main__':
    cli_args = parse_args()
    generate(
        dataset=cli_args.dataset,
        n_supervoxels=cli_args.n_supervoxels,
        mode=cli_args.mode,
        foreground_threshold=cli_args.foreground_threshold,
    )
