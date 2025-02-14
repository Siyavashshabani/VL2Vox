# -*- coding: utf-8 -*-
#
# Developed by Haozhe Xie <cshzxie@gmail.com>

import json
import numpy as np
import logging
import torch
import torch.backends.cudnn
import torch.utils.data

import utils.data_loaders
import utils.data_transforms
import utils.helpers

from models.encoder import Encoder
from models.decoder import Decoder
from models.refiner import Refiner
from models.merger import Merger
from utils.average_meter import AverageMeter
from transformers import AutoProcessor, FlavaModel
import torch
###############################################################################################################



def ensure_fixed_shape(tensor, target_length=202):
    """
    Ensures the sequence length of the tensor is fixed to `target_length`.
    Pads or truncates as needed.

    Args:
        tensor (torch.Tensor): Input tensor of shape [B, N, D].
        target_length (int): Desired sequence length.
    
    Returns:
        torch.Tensor: Tensor of shape [B, target_length, D].
    """
    batch_size, seq_length, feature_dim = tensor.shape

    if seq_length < target_length:
        # Pad the tensor to the right
        padding = torch.zeros((batch_size, target_length - seq_length, feature_dim), device=tensor.device)
        tensor = torch.cat([tensor, padding], dim=1)
    elif seq_length > target_length:
        # Truncate the tensor
        tensor = tensor[:, :target_length, :]

    return tensor


def test_flava_dino(cfg,vlm2pix, epoch_idx=-1, test_data_loader=None, test_writer=None):
    # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
    torch.backends.cudnn.benchmark = True

    # Load taxonomies of dataset
    taxonomies = []
    with open(cfg.DATASETS[cfg.DATASET.TEST_DATASET.upper()].TAXONOMY_FILE_PATH, encoding='utf-8') as file:
        taxonomies = json.loads(file.read())
    taxonomies = {t['taxonomy_id']: t for t in taxonomies}

    # Set up data loader
    if test_data_loader is None:
        # Set up data augmentation
        IMG_SIZE = cfg.CONST.IMG_H, cfg.CONST.IMG_W
        CROP_SIZE = cfg.CONST.CROP_IMG_H, cfg.CONST.CROP_IMG_W
        test_transforms = utils.data_transforms.Compose([
            utils.data_transforms.CenterCrop(IMG_SIZE, CROP_SIZE),
            utils.data_transforms.RandomBackground(cfg.TEST.RANDOM_BG_COLOR_RANGE),
            utils.data_transforms.Normalize(mean=cfg.DATASET.MEAN, std=cfg.DATASET.STD),
            utils.data_transforms.ToTensor(),
        ])

        dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TEST_DATASET](cfg)
        test_data_loader = torch.utils.data.DataLoader(dataset=dataset_loader.get_dataset(
            utils.data_loaders.DatasetType.TEST, cfg.CONST.N_VIEWS_RENDERING, test_transforms),
                                                       batch_size=1,
                                                       num_workers=cfg.CONST.NUM_WORKER,
                                                       pin_memory=True,
                                                       shuffle=False)


    # Set up loss functions
    bce_loss = torch.nn.BCELoss()

    # Testing loop
    n_samples = len(test_data_loader)
    test_iou = dict()
    vl_encoder_losses = AverageMeter()
    refiner_losses = AverageMeter()
    dino_encoder_losses = AverageMeter()
    fusion_losses = AverageMeter()

    # Switch models to evaluation mode
    vlm2pix.eval()

    for sample_idx, (taxomony_class, taxonomy_id, sample_name, rendering_images, ground_truth_volume) in enumerate(test_data_loader):
        taxonomy_id = taxonomy_id[0] if isinstance(taxonomy_id[0], str) else taxonomy_id[0].item()
        sample_name = sample_name[0]

        with torch.no_grad():
            # Get data from data loader
            rendering_images = utils.helpers.var_or_cuda(rendering_images)
            ground_truth_volume = utils.helpers.var_or_cuda(ground_truth_volume)


            vl_encoder_loss, dino_encoder_loss, refiner_loss, fusion_loss, generated_volume = vlm2pix(rendering_images,ground_truth_volume, taxomony_class )


            # Append loss and accuracy to average metrics
            vl_encoder_losses.update(vl_encoder_loss.item())
            refiner_losses.update(refiner_loss.item())
            dino_encoder_losses.update(dino_encoder_loss.item())
            fusion_losses.update(fusion_loss.item())

            # IoU per sample
            sample_iou = []
            for th in cfg.TEST.VOXEL_THRESH:
                _volume = torch.ge(generated_volume, th).float()
                intersection = torch.sum(_volume.mul(ground_truth_volume)).float()
                union = torch.sum(torch.ge(_volume.add(ground_truth_volume), 1)).float()
                sample_iou.append((intersection / union).item())

            # IoU per taxonomy
            if taxonomy_id not in test_iou:
                test_iou[taxonomy_id] = {'n_samples': 0, 'iou': []}
            test_iou[taxonomy_id]['n_samples'] += 1
            test_iou[taxonomy_id]['iou'].append(sample_iou)

            # Append generated volumes to TensorBoard
            if test_writer and sample_idx < 3:
                # Volume Visualization
                rendering_views = utils.helpers.get_volume_views(generated_volume.cpu().numpy())
                print("rendering_views------------------------:", rendering_views.shape)
                print("sample_idx-----------------------------:", sample_idx)
                print("epoch_idx-----------------------------:", epoch_idx)

                test_writer.add_image('Model%02d/Reconstructed' % sample_idx, rendering_views, epoch_idx)
                rendering_views = utils.helpers.get_volume_views(ground_truth_volume.cpu().numpy())
                test_writer.add_image('Model%02d/GroundTruth' % sample_idx, rendering_views, epoch_idx)

            # Print sample loss and IoU
            logging.info('Test[%d/%d] Taxonomy = %s Sample = %s VLDLoss = %.4f RLoss = %.4f  DinoLoss = %.4f IoU = %s' % #FuLoss = %.4f
                         (sample_idx + 1, n_samples, taxonomy_id, sample_name, 
                          vl_encoder_loss.item(), refiner_loss.item(),  dino_encoder_loss.item(), #fusion_loss.item(), 
                          ['%.4f' % si for si in sample_iou]))
        # break
    # Output testing results
    mean_iou = []
    for taxonomy_id in test_iou:
        test_iou[taxonomy_id]['iou'] = np.mean(test_iou[taxonomy_id]['iou'], axis=0)
        mean_iou.append(test_iou[taxonomy_id]['iou'] * test_iou[taxonomy_id]['n_samples'])
    mean_iou = np.sum(mean_iou, axis=0) / n_samples

    # Print header
    print('============================ TEST RESULTS ============================')
    print('Taxonomy', end='\t')
    print('#Sample', end='\t')
    print('Baseline', end='\t')
    for th in cfg.TEST.VOXEL_THRESH:
        print('t=%.2f' % th, end='\t')
    print()
    # Print body
    for taxonomy_id in test_iou:
        print('%s' % taxonomies[taxonomy_id]['taxonomy_name'].ljust(8), end='\t')
        print('%d' % test_iou[taxonomy_id]['n_samples'], end='\t')
        if 'baseline' in taxonomies[taxonomy_id]:
            print('%.4f' % taxonomies[taxonomy_id]['baseline']['%d-view' % cfg.CONST.N_VIEWS_RENDERING], end='\t\t')
        else:
            print('N/a', end='\t\t')

        for ti in test_iou[taxonomy_id]['iou']:
            print('%.4f' % ti, end='\t')
        print()
    # Print mean IoU for each threshold
    print('Overall ', end='\t\t\t\t')
    for mi in mean_iou:
        print('%.4f' % mi, end='\t')
    print('\n')

    # Add testing results to TensorBoard
    max_iou = np.max(mean_iou)
    if test_writer is not None:
        test_writer.add_scalar('vl_encoder_losses/EpochLoss', vl_encoder_losses.avg, epoch_idx)
        test_writer.add_scalar('dino_encoder_losses/EpochLoss', dino_encoder_losses.avg, epoch_idx)
        test_writer.add_scalar('fusion_losses/EpochLoss', fusion_losses.avg, epoch_idx)
        test_writer.add_scalar('Refiner/EpochLoss', refiner_losses.avg, epoch_idx)
        test_writer.add_scalar('Refiner/IoU', max_iou, epoch_idx)

    return max_iou
