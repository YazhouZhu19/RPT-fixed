"""
Dataset for Training and Test
Extended from ADNet code by Hansen et al.
"""
import torch
from torch.utils.data import Dataset
import torchvision.transforms as deftfx
import glob
import os
import SimpleITK as sitk
import random
import numpy as np
from . import image_transforms as myit
from .dataset_specifics import get_excluded_slice_indices, get_folds
from .datasets import _align_case_paths, _contiguous_runs, _standardize_volume


def _episode_candidates(labels, class_ids, excluded_slices, required, min_size):
    excluded = set(excluded_slices.tolist())
    candidates = {}
    for class_id in class_ids:
        class_mask = labels == class_id
        slice_indices = np.flatnonzero(class_mask.sum(axis=(1, 2)) > 0)
        slice_indices = np.array(
            [slice_idx for slice_idx in slice_indices if slice_idx not in excluded],
            dtype=np.int64,
        )
        episodes = []
        for run in _contiguous_runs(slice_indices):
            for start in range(0, len(run) - required + 1):
                episode = run[start:start + required]
                if max(np.count_nonzero(class_mask[slice_idx]) for slice_idx in episode) >= min_size:
                    episodes.append(episode.copy())
        if episodes:
            candidates[class_id] = episodes
    return candidates


class TestDataset(Dataset):

    def __init__(self, args):

        # reading the paths
        if args['dataset'] == 'CMR':
            data_dir = os.path.join(args['data_dir'], 'cmr_MR_normalized')
        elif args['dataset'] == 'CHAOST2':
            data_dir = os.path.join(args['data_dir'], 'chaos_MR_T2_normalized')
        elif args['dataset'] == 'SABS':
            data_dir = os.path.join(args['data_dir'], 'sabs_CT_normalized')
        else:
            raise ValueError(f"Dataset: {args['dataset']} not found")

        self.image_dirs = glob.glob(os.path.join(data_dir, 'image*'))
        label_dirs = glob.glob(os.path.join(data_dir, 'label*'))
        _, (self.image_dirs, self.label_dirs) = _align_case_paths(self.image_dirs, label_dirs)

        # remove test fold!
        self.FOLD = get_folds(args['dataset'])
        fold_indices = self.FOLD[args['eval_fold']]
        self.image_dirs = [self.image_dirs[idx] for idx in fold_indices]
        self.label_dirs = [self.label_dirs[idx] for idx in fold_indices]

        # split into support/query
        support_idx = args.get('supp_idx', -1)
        if support_idx < 0:
            support_idx += len(self.image_dirs)
        if support_idx != len(self.image_dirs) - 1:
            raise ValueError('supp_idx must select the dedicated final support case')
        self.support_dir = self.image_dirs.pop(support_idx)
        self.support_label_dir = self.label_dirs.pop(support_idx)
        self.label = None

    def __len__(self):
        return len(self.image_dirs)

    def __getitem__(self, idx):
        if self.label is None:
            raise ValueError('Set TestDataset.label before iterating over query volumes')

        img_path = self.image_dirs[idx]
        img = sitk.GetArrayFromImage(sitk.ReadImage(img_path))
        img = _standardize_volume(img)
        img = np.stack(3 * [img], axis=1)

        lbl = sitk.GetArrayFromImage(sitk.ReadImage(self.label_dirs[idx]))
        lbl[lbl == 200] = 1
        lbl[lbl == 500] = 2
        lbl[lbl == 600] = 3
        lbl = 1 * (lbl == self.label)

        sample = {'id': img_path}

        # Evaluation protocol.
        idx = lbl.sum(axis=(1, 2)) > 0
        sample['image'] = torch.from_numpy(img[idx])
        sample['label'] = torch.from_numpy(lbl[idx])

        return sample

    def get_support_index(self, n_shot, C):
        """
        Selecting intervals according to Ouyang et al.
        """
        if n_shot < 1:
            raise ValueError('n_shot must be at least 1')
        if n_shot == 1:
            pcts = [0.5]
        else:
            half_part = 1 / (n_shot * 2)
            part_interval = (1.0 - 1.0 / n_shot) / (n_shot - 1)
            pcts = [half_part + part_interval * ii for ii in range(n_shot)]

        return (np.array(pcts) * C).astype('int')

    def getSupport(self, label=None, all_slices=True, N=None):
        if label is None:
            raise ValueError('Need to specify label class!')

        img_path = self.support_dir
        img = sitk.GetArrayFromImage(sitk.ReadImage(img_path))
        img = _standardize_volume(img)
        img = np.stack(3 * [img], axis=1)

        lbl = sitk.GetArrayFromImage(sitk.ReadImage(self.support_label_dir))
        lbl[lbl == 200] = 1
        lbl[lbl == 500] = 2
        lbl[lbl == 600] = 3
        lbl = 1 * (lbl == label)

        sample = {}
        if all_slices:
            sample['image'] = torch.from_numpy(img)
            sample['label'] = torch.from_numpy(lbl)
        else:
            # select N labeled slices
            if N is None:
                raise ValueError('Need to specify number of labeled slices!')
            idx = lbl.sum(axis=(1, 2)) > 0
            if not idx.any():
                raise ValueError(f'Support case has no slices for label {label}')
            idx_ = self.get_support_index(N, idx.sum())

            sample['image'] = torch.from_numpy(img[idx][idx_])
            sample['label'] = torch.from_numpy(lbl[idx][idx_])

        return sample


class TrainDataset(Dataset):

    def __init__(self, args):
        self.n_shot = args['n_shot']
        self.n_way = args['n_way']
        self.n_query = args['n_query']
        self.n_sv = args['n_sv']
        self.max_iter = args['max_iter']
        self.read = True  # read images before get_item
        self.train_sampling = 'neighbors'
        self.min_size = args['min_size']  # 200
        self.test_label = args['test_label']
        self.exclude_label = args['exclude_label']
        self.use_gt = args['use_gt']
        if self.n_shot < 1:
            raise ValueError('n_shot must be at least 1')
        if self.n_way != 1 or self.n_query != 1:
            raise ValueError('RPT training currently supports only n_way=1 and n_query=1')

        # reading the paths (leaving the reading of images into memory to __getitem__)
        if args['dataset'] == 'CMR':
            self.image_dirs = glob.glob(os.path.join(args['data_dir'], 'cmr_MR_normalized/image*'))
            self.label_dirs = glob.glob(os.path.join(args['data_dir'], 'cmr_MR_normalized/label*'))
        elif args['dataset'] == 'CHAOST2':
            self.image_dirs = glob.glob(os.path.join(args['data_dir'], 'chaos_MR_T2_normalized/image*'))
            self.label_dirs = glob.glob(os.path.join(args['data_dir'], 'chaos_MR_T2_normalized/label*'))
        elif args['dataset'] == 'SABS':
            self.image_dirs = glob.glob(os.path.join(args['data_dir'], 'sabs_CT_normalized/image*'))
            self.label_dirs = glob.glob(os.path.join(args['data_dir'], 'sabs_CT_normalized/label*'))
        else:
            raise ValueError(f"Dataset: {args['dataset']} not found")

        self.sprvxl_dirs = glob.glob(os.path.join(args['data_dir'], 'supervoxels_' + str(args['n_sv']), 'super*'))
        _, aligned_paths = _align_case_paths(self.image_dirs, self.label_dirs, self.sprvxl_dirs)
        self.image_dirs, self.label_dirs, self.sprvxl_dirs = aligned_paths

        # remove test fold!
        self.FOLD = get_folds(args['dataset'])
        self.image_dirs = [elem for idx, elem in enumerate(self.image_dirs) if idx not in self.FOLD[args['eval_fold']]]
        self.label_dirs = [elem for idx, elem in enumerate(self.label_dirs) if idx not in self.FOLD[args['eval_fold']]]
        self.sprvxl_dirs = [elem for idx, elem in enumerate(self.sprvxl_dirs) if
                            idx not in self.FOLD[args['eval_fold']]]

        # read images
        if self.read:
            self.images = {}
            self.labels = {}
            self.sprvxls = {}
            for image_dir, label_dir, sprvxl_dir in zip(self.image_dirs, self.label_dirs, self.sprvxl_dirs):
                self.images[image_dir] = sitk.GetArrayFromImage(sitk.ReadImage(image_dir))
                self.labels[label_dir] = sitk.GetArrayFromImage(sitk.ReadImage(label_dir))
                self.sprvxls[sprvxl_dir] = sitk.GetArrayFromImage(sitk.ReadImage(sprvxl_dir))

        required = (self.n_shot * self.n_way) + self.n_query
        self.episode_candidates = {}
        for pat_idx in range(len(self.image_dirs)):
            gt = self.labels[self.label_dirs[pat_idx]]
            lbl = gt if self.use_gt else self.sprvxls[self.sprvxl_dirs[pat_idx]]
            unique = np.setdiff1d(np.unique(lbl), [0])
            if self.use_gt:
                unique = np.setdiff1d(unique, self.test_label)
            excluded = get_excluded_slice_indices(gt, self.exclude_label)
            candidates = _episode_candidates(lbl, unique, excluded, required, self.min_size)
            if len(candidates) >= 2:
                self.episode_candidates[pat_idx] = candidates
        if not self.episode_candidates:
            raise RuntimeError('No cases contain two valid classes for outside-class episodes')

    def __len__(self):
        return self.max_iter

    def gamma_tansform(self, img):
        gamma_range = (0.5, 1.5)
        gamma = np.random.rand() * (gamma_range[1] - gamma_range[0]) + gamma_range[0]
        cmin = img.min()
        irange = (img.max() - cmin + 1e-5)

        img = img - cmin + 1e-5
        img = irange * np.power(img * 1.0 / irange, gamma)
        img = img + cmin

        return img

    def geom_transform(self, img, mask):

        affine = {'rotate': 5, 'shift': (5, 5), 'shear': 5, 'scale': (0.9, 1.2)}
        alpha = 10
        sigma = 5
        order = 3

        tfx = []
        tfx.append(myit.RandomAffine(affine.get('rotate'),
                                     affine.get('shift'),
                                     affine.get('shear'),
                                     affine.get('scale'),
                                     affine.get('scale_iso', True),
                                     order=order))
        tfx.append(myit.ElasticTransform(alpha, sigma))
        transform = deftfx.Compose(tfx)

        if len(img.shape) > 4:
            n_shot = img.shape[1]
            for shot in range(n_shot):
                cat = np.concatenate((img[0, shot], mask[:, shot])).transpose(1, 2, 0)
                cat = transform(cat).transpose(2, 0, 1)
                img[0, shot] = cat[:3, :, :]
                mask[:, shot] = np.rint(cat[3:, :, :])

        else:
            for q in range(img.shape[0]):
                cat = np.concatenate((img[q], mask[q][None])).transpose(1, 2, 0)
                cat = transform(cat).transpose(2, 0, 1)
                img[q] = cat[:3, :, :]
                mask[q] = np.rint(cat[3:, :, :].squeeze())

        return img, mask

    def __getitem__(self, idx):

        # sample patient idx
        pat_idx = random.choice(list(self.episode_candidates))

        if self.read:
            # get image/supervoxel volume from dictionary
            img = self.images[self.image_dirs[pat_idx]]
            gt = self.labels[self.label_dirs[pat_idx]]
            sprvxl = self.sprvxls[self.sprvxl_dirs[pat_idx]]
        else:
            # read image/supervoxel volume into memory
            img = sitk.GetArrayFromImage(sitk.ReadImage(self.image_dirs[pat_idx]))
            gt = sitk.GetArrayFromImage(sitk.ReadImage(self.label_dirs[pat_idx]))
            sprvxl = sitk.GetArrayFromImage(sitk.ReadImage(self.sprvxl_dirs[pat_idx]))

        # normalize
        img = _standardize_volume(img)

        # chose training label
        if self.use_gt:
            lbl = gt.copy()
        else:
            lbl = sprvxl.copy()
        # lbl is label numpy

        # sample two different valid classes (gt/supervoxel)
        candidates = self.episode_candidates[pat_idx]
        cls_idx, cls_outside_idx = random.sample(list(candidates), 2)
        sample = random.choice(candidates[cls_idx])
        sample_outside = random.choice(candidates[cls_outside_idx])
        lbl_cls = 1 * (lbl == cls_idx)
        lbl_cls_outside = 1 * (lbl == cls_outside_idx)

        # invert order
        if np.random.random(1) > 0.5:
            sample = sample[::-1]  # successive slices (inverted)

        sup_lbl = lbl_cls[sample[:self.n_shot * self.n_way]][None,]  # n_way * (n_shot * C) * H * W
        qry_lbl = lbl_cls[sample[self.n_shot * self.n_way:]]  # n_qry * C * H * W

        sup_img = img[sample[:self.n_shot * self.n_way]][None,]  # n_way * (n_shot * C) * H * W
        sup_img = np.stack((sup_img, sup_img, sup_img), axis=2)
        qry_img = img[sample[self.n_shot * self.n_way:]]  # n_qry * C * H * W
        qry_img = np.stack((qry_img, qry_img, qry_img), axis=1)

        # invert order
        if np.random.random(1) > 0.5:
            sample_outside = sample_outside[::-1]  # successive outside slices (inverted)

        sup_lbl_outside = lbl_cls_outside[sample_outside[:self.n_shot * self.n_way]][None,]
        qry_lbl_outside = lbl_cls_outside[sample_outside[self.n_shot * self.n_way:]]

        sup_img_outside = img[sample_outside[:self.n_shot * self.n_way]][None,]
        sup_img_outside = np.stack((sup_img_outside, sup_img_outside, sup_img_outside), axis=2)
        qry_img_outside = img[sample_outside[self.n_shot * self.n_way:]]
        qry_img_outside = np.stack((qry_img_outside, qry_img_outside, qry_img_outside), axis=1)

        # gamma transform
        if np.random.random(1) > 0.5:
            qry_img = self.gamma_tansform(qry_img)
        else:
            sup_img = self.gamma_tansform(sup_img)

        # geom transform
        if np.random.random(1) > 0.5:
            qry_img, qry_lbl = self.geom_transform(qry_img, qry_lbl)
        else:
            sup_img, sup_lbl = self.geom_transform(sup_img, sup_lbl)

        # gamma transform for outside classes
        if np.random.random(1) > 0.5:
            qry_img_outside = self.gamma_tansform(qry_img_outside)
        else:
            sup_img_outside = self.gamma_tansform(sup_img_outside)

        # geom transform for outside classes
        if np.random.random(1) > 0.5:
            qry_img_outside, qry_lbl_outside = self.geom_transform(qry_img_outside, qry_lbl_outside)
        else:
            sup_img_outside, sup_lbl_outside = self.geom_transform(sup_img_outside, sup_lbl_outside)

        sample = {'support_images': sup_img,
                  'support_fg_labels': sup_lbl,
                  'query_images': qry_img,
                  'query_labels': qry_lbl,
                  'support_images_outside': sup_img_outside,
                  'support_outside_labels': sup_lbl_outside,
                  'query_images_outside': qry_img_outside,
                  'query_outside_labels': qry_lbl_outside,
                  'selected_class': cls_idx}

        return sample
