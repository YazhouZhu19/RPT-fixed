#!/usr/bin/env python
import os
import random
import logging
import shutil
import math

import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from models.fewshot import FewShotSeg
from dataloaders.datasets import TrainDataset
from config import ex


@ex.automain
def main(_run, _config, _log):
    if _run.observers:
        # Set up source folder
        os.makedirs(f'{_run.observers[0].dir}/snapshots', exist_ok=True)
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

    if torch.cuda.is_available():
        torch.cuda.set_device(device=_config['gpu_id'])
        device = torch.device(f'cuda:{_config["gpu_id"]}')
        cudnn.enabled = True
        cudnn.benchmark = _config['seed'] is None
    else:
        device = torch.device('cpu')
        _log.warning('CUDA is unavailable; training will run on CPU and may be very slow.')
    torch.set_num_threads(1)

    _log.info('Create model...')
    model = FewShotSeg(pretrained_weights=_config['pretrained_weights'])
    model = model.to(device)
    model.train()

    _log.info('Set optimizer...')
    optimizer = torch.optim.SGD(model.parameters(), **_config['optim'])
    lr_milestones = list(range(
        _config['max_iters_per_load'], _config['n_steps'], _config['max_iters_per_load']
    ))
    scheduler = MultiStepLR(optimizer, milestones=lr_milestones, gamma=_config['lr_step_gamma'])

    my_weight = torch.tensor([_config['bg_wt'], 1.0], dtype=torch.float32, device=device)
    criterion = nn.NLLLoss(ignore_index=_config['ignore_label'], weight=my_weight)

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
        'test_label': _config['test_label'],
        'exclude_label': _config['exclude_label'],
        'use_gt': _config['use_gt'],
    }
    train_dataset = TrainDataset(data_config)
    train_loader = DataLoader(train_dataset,
                              batch_size=_config['batch_size'],
                              shuffle=True,
                              num_workers=_config['num_workers'],
                              pin_memory=device.type == 'cuda',
                              drop_last=True)

    if len(train_loader) == 0:
        raise ValueError('Training loader is empty; reduce batch_size or increase max_iters_per_load')
    n_sub_epochs = math.ceil(_config['n_steps'] / len(train_loader))
    log_loss = {'total_loss': 0, 'query_loss': 0, 'align_loss': 0, 'thresh_loss': 0}

    i_iter = 0
    _log.info('Start training...')

    for sub_epoch in range(n_sub_epochs):
        _log.info(f'This is epoch "{sub_epoch}" of "{n_sub_epochs}" epochs.')

        for _, sample in enumerate(train_loader):
            # Prepare episode data.
            support_batch = sample['support_images']
            support_mask_batch = sample['support_fg_labels']
            query_batch = sample['query_images']
            query_label_batch = sample['query_labels']
            support_images = [
                [support_batch[:, way, shot].float().to(device) for shot in range(support_batch.shape[2])]
                for way in range(support_batch.shape[1])
            ]
            support_fg_mask = [
                [support_mask_batch[:, way, shot].float().to(device) for shot in range(support_mask_batch.shape[2])]
                for way in range(support_mask_batch.shape[1])
            ]
            query_images = [
                query_batch[:, query].float().to(device) for query in range(query_batch.shape[1])
            ]
            query_labels = query_label_batch.long().to(device).flatten(0, 1)

            # Compute outputs and losses.
            query_pred, periphery_loss, align_loss, mse_loss, qry_loss = model(support_images, support_fg_mask,
                                                                               query_images, query_labels, train=True)

            query_loss = criterion(torch.log(torch.clamp(query_pred, torch.finfo(torch.float32).eps,
                                                         1 - torch.finfo(torch.float32).eps)), query_labels)

            # bd_loss = criterion_bd(query_pred, query_labels)
            # dice_loss = criterion_dice(query_pred, query_labels)

            loss = query_loss + 0.1 * periphery_loss + align_loss + 0.1 * mse_loss + qry_loss

            # Compute gradient and do SGD step.
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Log loss
            query_loss = query_loss.item()
            # align_loss = align_loss.detach().data.cpu().numpy()

            _run.log_scalar('total_loss', loss.item())
            _run.log_scalar('query_loss', query_loss)

            log_loss['total_loss'] += loss.item()
            log_loss['query_loss'] += query_loss

            # Print loss and take snapshots.
            if (i_iter + 1) % _config['print_interval'] == 0:
                total_loss = log_loss['total_loss'] / _config['print_interval']
                query_loss = log_loss['query_loss'] / _config['print_interval']

                log_loss['total_loss'] = 0
                log_loss['query_loss'] = 0

                _log.info(f'step {i_iter + 1}: total_loss: {total_loss}, query_loss: {query_loss},'
                          )

            if (i_iter + 1) % _config['save_snapshot_every'] == 0:
                _log.info('###### Taking snapshot ######')
                torch.save(model.state_dict(),
                           os.path.join(f'{_run.observers[0].dir}/snapshots', f'{i_iter + 1}.pth'))

            i_iter += 1
            if i_iter >= _config['n_steps']:
                break
        if i_iter >= _config['n_steps']:
            break

    _log.info('End of training.')
    return 1
