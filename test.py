#!/usr/bin/env python
"""
For evaluation
"""
import inspect
import logging
import os
import random
import shutil
from collections.abc import Mapping

import numpy as np
import SimpleITK as sitk
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from models.fewshot import FewShotSeg
from dataloaders.datasets import TestDataset
from dataloaders.dataset_specifics import get_label_names
from utils import Scores
from config import ex


@ex.automain
def main(_run, _config, _log):
    if _run.observers:
        os.makedirs(f'{_run.observers[0].dir}/interm_preds', exist_ok=True)
        for source_file, _ in _run.experiment_info['sources']:
            os.makedirs(os.path.dirname(f'{_run.observers[0].dir}/source/{source_file}'),
                        exist_ok=True)
            _run.observers[0].save_file(source_file, f'source/{source_file}')
        shutil.rmtree(f'{_run.observers[0].basedir}/_sources')

        # Set up logger -> log to .txt
        file_handler = logging.FileHandler(os.path.join(_run.observers[0].dir, 'logger.log'))
        file_handler.setLevel('INFO')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
        file_handler.setFormatter(formatter)
        _log.handlers.append(file_handler)
        run_id = os.path.basename(_run.observers[0].dir)
        _log.info(f'Run "{_config["exp_str"]}" with ID "{run_id}"')

    # Deterministic setting for reproduciablity.
    if _config['seed'] is not None:
        random.seed(_config['seed'])
        np.random.seed(_config['seed'])
        torch.manual_seed(_config['seed'])
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(_config['seed'])
        cudnn.deterministic = True

    if _config['batch_size'] != 1:
        raise ValueError('Evaluation requires batch_size=1 because volumes have variable slice counts')
    if torch.cuda.is_available():
        torch.cuda.set_device(device=_config['gpu_id'])
        device = torch.device(f'cuda:{_config["gpu_id"]}')
        cudnn.enabled = True
        cudnn.benchmark = _config['seed'] is None
    else:
        device = torch.device('cpu')
        _log.warning('CUDA is unavailable; evaluation will run on CPU and may be very slow.')
    torch.set_num_threads(1)

    _log.info('Create model...')
    if not _config['reload_model_path']:
        raise ValueError('reload_model_path must point to a trained RPT checkpoint')
    model = FewShotSeg(pretrained_weights=None).to(device)
    load_kwargs = {'map_location': 'cpu'}
    if 'weights_only' in inspect.signature(torch.load).parameters:
        load_kwargs['weights_only'] = True
    checkpoint = torch.load(_config['reload_model_path'], **load_kwargs)
    if isinstance(checkpoint, Mapping) and 'state_dict' in checkpoint:
        checkpoint = checkpoint['state_dict']
    if not isinstance(checkpoint, Mapping):
        raise TypeError('The model checkpoint must contain a state-dict mapping')
    checkpoint = {
        (key[len('module.'):] if key.startswith('module.') else key): value
        for key, value in checkpoint.items()
    }
    model.load_state_dict(checkpoint)

    _log.info('Load data...')
    data_config = {
        'data_dir': _config['path'][_config['dataset']]['data_dir'],
        'dataset': _config['dataset'],
        'n_shot': _config['n_shot'],
        'n_way': _config['n_way'],
        'n_query': _config['n_query'],
        'n_sv': _config['n_sv'],
        'max_iter': _config['max_iters_per_load'],
        'eval_fold': _config['eval_fold'],
        'min_size': _config['min_size'],
        'max_slices': _config['max_slices'],
        'supp_idx': _config['supp_idx'],
    }
    test_dataset = TestDataset(data_config)
    test_loader = DataLoader(test_dataset,
                             batch_size=_config['batch_size'],
                             shuffle=False,
                             num_workers=_config['num_workers'],
                             pin_memory=device.type == 'cuda',
                             drop_last=False)

    # Get unique labels (classes).
    labels = get_label_names(_config['dataset'])

    # Loop over classes.
    class_dice = {}
    class_iou = {}

    _log.info('Starting validation...')
    for label_val, label_name in labels.items():

        # Skip BG class.
        if label_name == 'BG':
            continue
        elif label_val not in _config['test_label']:
            continue

        _log.info(f'Test Class: {label_name}')

        # Get support sample + mask for current class.
        support_sample = test_dataset.getSupport(label=label_val, all_slices=False, N=_config['n_part'])

        test_dataset.label = label_val

        # Test.
        with torch.no_grad():
            model.eval()

            # Unpack support data.
            support_image = [support_sample['image'][[i]].float().to(device) for i in
                             range(support_sample['image'].shape[0])]  # n_shot x 3 x H x W, support_image is a list {3X(1, 3, 256, 256)}
            support_fg_mask = [support_sample['label'][[i]].float().to(device) for i in
                               range(support_sample['image'].shape[0])]  # n_shot x H x W

            # Loop through query volumes.
            scores = Scores()
            for i, sample in enumerate(test_loader):  # this "for" loops 4 times

                # Unpack query data.
                query_image = sample['image'][0].float().to(device)
                query_label = sample['label'][0].long()  # C x H x W
                query_id = sample['id'][0].split('image_')[1][:-len('.nii.gz')]

                # Compute output.
                # Match support slice and query sub-chunck.
                query_pred = torch.zeros_like(query_label, device='cpu')
                C_q = sample['image'].shape[1]    # slice number of query img

                idx_ = np.linspace(0, C_q, _config['n_part'] + 1).astype('int')
                for sub_chunck in range(_config['n_part']):  # n_part = 3
                    support_image_s = [support_image[sub_chunck]]  # 1 x 3 x H x W
                    support_fg_mask_s = [support_fg_mask[sub_chunck]]  # 1 x H x W
                    query_image_s = query_image[idx_[sub_chunck]:idx_[sub_chunck + 1]]  # C' x 3 x H x W
                    query_pred_s = []
                    for slice_idx in range(query_image_s.shape[0]):
                        _pred_s, _, _, _, _ = model(
                            [support_image_s], [support_fg_mask_s], [query_image_s[[slice_idx]]], train=False
                        )
                        query_pred_s.append(_pred_s)
                    if not query_pred_s:
                        continue
                    query_pred_s = torch.cat(query_pred_s, dim=0)
                    query_pred_s = query_pred_s.argmax(dim=1).cpu()  # C x H x W
                    query_pred[idx_[sub_chunck]:idx_[sub_chunck + 1]] = query_pred_s

                # Record scores.
                scores.record(query_pred, query_label)

                # Log.
                _log.info(
                    f'Tested query volume: {sample["id"][0][len(_config["path"][_config["dataset"]]["data_dir"]):]}.')
                _log.info(f'Dice score: {scores.patient_dice[-1].item()}')

                # Save predictions.
                file_name = os.path.join(f'{_run.observers[0].dir}/interm_preds',
                                         f'prediction_{query_id}_{label_name}.nii.gz')
                num_slices = int(sample['num_slices'][0])
                slice_indices = sample['slice_indices'][0].long()
                full_prediction = torch.zeros(
                    (num_slices, *query_pred.shape[-2:]), dtype=torch.uint8
                )
                full_prediction[slice_indices] = query_pred.to(torch.uint8)
                itk_pred = sitk.GetImageFromArray(full_prediction.numpy())
                itk_pred.CopyInformation(sitk.ReadImage(sample['id'][0]))
                sitk.WriteImage(itk_pred, file_name, True)
                _log.info(f'{query_id} has been saved. ')

            # Log class-wise results
            class_dice[label_name] = torch.tensor(scores.patient_dice).mean().item()
            class_iou[label_name] = torch.tensor(scores.patient_iou).mean().item()
            _log.info(f'Test Class: {label_name}')
            _log.info(f'Mean class IoU: {class_iou[label_name]}')
            _log.info(f'Mean class Dice: {class_dice[label_name]}')

    _log.info('Final results...')
    _log.info(f'Mean IoU: {class_iou}')
    _log.info(f'Mean Dice: {class_dice}')

    if not class_dice:
        raise ValueError('No evaluation classes matched test_label')
    mean_dice = sum(class_dice.values()) / len(class_dice)
    results_path = os.path.join(_run.observers[0].dir, 'results.txt')
    with open(results_path, 'w') as file:
        file.write(f'{mean_dice}\n')

    _log.info(f'Whole mean Dice: {mean_dice}')
    _log.info('End of validation.')
    return 1
