import os
import logging
import random
import torch
import torch.backends.cudnn
import torch.utils.data

import pix2vox_utils.data_loaders
import pix2vox_utils.data_transforms
import pix2vox_utils.helpers

from datetime import datetime as dt
from tensorboardX import SummaryWriter
from time import time

from core.test_flava import test_net_flava
# from models.encoder import Encoder
# from models.decoder import Decoder
# from models.refiner import Refiner
# from models.merger import Merger
from pix2vox_utils.average_meter import AverageMeter


############################################################################# add flava 
import torch
import torch.nn as nn
from transformers import AutoProcessor, FlavaModel
from torch.utils.data import Dataset, DataLoader
from models.VL2Vox import VL2Pix
from models.VL2VoxDino import VL2VoxDino


def train_net_flava_finetune(cfg):
    # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
    torch.backends.cudnn.benchmark = True

    print("Start the train_net code -------------------------------------------------")
    # Set up data augmentation
    IMG_SIZE = cfg.CONST.IMG_H, cfg.CONST.IMG_W
    CROP_SIZE = cfg.CONST.CROP_IMG_H, cfg.CONST.CROP_IMG_W
    train_transforms = pix2vox_utils.data_transforms.Compose([
        pix2vox_utils.data_transforms.RandomCrop(IMG_SIZE, CROP_SIZE),
        pix2vox_utils.data_transforms.RandomBackground(cfg.TRAIN.RANDOM_BG_COLOR_RANGE),
        pix2vox_utils.data_transforms.ColorJitter(cfg.TRAIN.BRIGHTNESS, cfg.TRAIN.CONTRAST, cfg.TRAIN.SATURATION),
        pix2vox_utils.data_transforms.RandomNoise(cfg.TRAIN.NOISE_STD),
        pix2vox_utils.data_transforms.Normalize(mean=cfg.DATASET.MEAN, std=cfg.DATASET.STD),
        pix2vox_utils.data_transforms.RandomFlip(),
        pix2vox_utils.data_transforms.RandomPermuteRGB(),
        pix2vox_utils.data_transforms.ToTensor(),
    ])
    val_transforms = pix2vox_utils.data_transforms.Compose([
        pix2vox_utils.data_transforms.CenterCrop(IMG_SIZE, CROP_SIZE),
        pix2vox_utils.data_transforms.RandomBackground(cfg.TEST.RANDOM_BG_COLOR_RANGE),
        pix2vox_utils.data_transforms.Normalize(mean=cfg.DATASET.MEAN, std=cfg.DATASET.STD),
        pix2vox_utils.data_transforms.ToTensor(),
    ])

    # Set up data loader
    train_dataset_loader = pix2vox_utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TRAIN_DATASET](cfg)
    val_dataset_loader = pix2vox_utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TEST_DATASET](cfg)
    train_data_loader = torch.utils.data.DataLoader(dataset=train_dataset_loader.get_dataset(
    pix2vox_utils.data_loaders.DatasetType.TRAIN, cfg.CONST.N_VIEWS_RENDERING, train_transforms),
                                                    batch_size=cfg.CONST.BATCH_SIZE,
                                                    num_workers=cfg.CONST.NUM_WORKER,
                                                    pin_memory=True,
                                                    shuffle=True,
                                                    drop_last=True)
    val_data_loader = torch.utils.data.DataLoader(dataset=val_dataset_loader.get_dataset(
        pix2vox_utils.data_loaders.DatasetType.VAL, cfg.CONST.N_VIEWS_RENDERING, val_transforms),
                                                  batch_size=1,
                                                  num_workers=cfg.CONST.NUM_WORKER,
                                                  pin_memory=True,
                                                  shuffle=False)

        # Set up loss functions
    bce_loss = torch.nn.BCELoss()
    # # Set up networks
    # encoder = Encoder(cfg)
    # decoder = Decoder(cfg)
    # refiner = Refiner(cfg)
    # merger = Merger(cfg)
    # convert_layer = ConvertLayer()
    # vlm2pix = VL2Pix(cfg)

    logging.info('Recovering from %s ...' % (cfg.CONST.WEIGHTS))
    # checkpoint = torch.load(cfg.CONST.WEIGHTS)
    if cfg.CONST.MODEL=="VLM2pix":
        vlm2pix = VL2Pix(cfg, bce_loss)
    elif cfg.CONST.MODEL=="VL2pixDino":
        vlm2pix = VL2VoxDino(cfg, bce_loss)
        print("VL2pixDino load!!!---------------------------")
    else:
        raise Exception("Incorrect model name!")
    
    logging.debug('Parameters in Encoder: %d.' % (pix2vox_utils.helpers.count_parameters(vlm2pix.encoder)))
    logging.debug('Parameters in Decoder: %d.' % (pix2vox_utils.helpers.count_parameters(vlm2pix.decoder)))
    logging.debug('Parameters in Refiner: %d.' % (pix2vox_utils.helpers.count_parameters(vlm2pix.refiner)))
    logging.debug('Parameters in Merger: %d.' %  (pix2vox_utils.helpers.count_parameters(vlm2pix.merger)))
    logging.debug('Parameters in convert_layer: %d.' % (pix2vox_utils.helpers.count_parameters(vlm2pix.convert_layer)))

    # # Initialize processor and model
    # flava_processor = AutoProcessor.from_pretrained("facebook/flava-full")
    # flava_model = FlavaModel.from_pretrained("facebook/flava-full") #.cuda()
    # Freeze FLAVA parameters
    # Set up solver
    if cfg.TRAIN.POLICY == 'adam':
        convert_layer_solver = torch.optim.Adam(vlm2pix.convert_layer.parameters(),
                                          lr=cfg.TRAIN.CONVERTER_LEARNING_RATE,
                                          betas=cfg.TRAIN.BETAS)
        decoder_solver = torch.optim.Adam(vlm2pix.decoder.parameters(),
                                          lr=cfg.TRAIN.DECODER_LEARNING_RATE,
                                          betas=cfg.TRAIN.BETAS)
        refiner_solver = torch.optim.Adam(vlm2pix.refiner.parameters(),
                                          lr=cfg.TRAIN.REFINER_LEARNING_RATE,
                                          betas=cfg.TRAIN.BETAS)
    elif cfg.TRAIN.POLICY == 'sgd':
        convert_layer_solver = torch.optim.SGD(vlm2pix.convert_layer.parameters(),
                                         lr=cfg.TRAIN.CONVERTER_LEARNING_RATE,
                                         momentum=cfg.TRAIN.MOMENTUM)
        decoder_solver = torch.optim.SGD(vlm2pix.decoder.parameters(),
                                         lr=cfg.TRAIN.DECODER_LEARNING_RATE,
                                         momentum=cfg.TRAIN.MOMENTUM)
        refiner_solver = torch.optim.SGD(vlm2pix.refiner.parameters(),
                                         lr=cfg.TRAIN.REFINER_LEARNING_RATE,
                                         momentum=cfg.TRAIN.MOMENTUM)
    else:
        raise Exception('[FATAL] %s Unknown optimizer %s.' % (dt.now(), cfg.TRAIN.POLICY))

    # Set up learning rate scheduler to decay learning rates dynamically
    if cfg.TRAIN=="LR":
        convert_layer_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(convert_layer_solver,
                                                                milestones=cfg.TRAIN.MERGER_LR_MILESTONES,
                                                                gamma=cfg.TRAIN.GAMMA)
        decoder_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(decoder_solver,
                                                                    milestones=cfg.TRAIN.DECODER_LR_MILESTONES,
                                                                    gamma=cfg.TRAIN.GAMMA)
        refiner_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(refiner_solver,
                                                                milestones=cfg.TRAIN.REFINER_LR_MILESTONES,
                                                                gamma=cfg.TRAIN.GAMMA)
    
    elif  cfg.TRAIN.SCHEDULAR=="Cosine":
        convert_layer_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(convert_layer_solver,
                                                            T_max=cfg.TRAIN.T_MAX,
                                                            eta_min=cfg.TRAIN.ETA_MIN )
        decoder_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(decoder_solver,
                                                            T_max=cfg.TRAIN.T_MAX,
                                                            eta_min=cfg.TRAIN.ETA_MIN )
        refiner_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(refiner_solver,
                                                            T_max=cfg.TRAIN.T_MAX,
                                                            eta_min=cfg.TRAIN.ETA_MIN )  
                 
    # Load pretrained model if exists
    init_epoch = 0
    best_iou = -1
    best_epoch = -1

    # Summary writer for TensorBoard
    output_dir = os.path.join(cfg.DIR.OUT_PATH, '%s', dt.now().isoformat())
    cfg.DIR.LOGS = output_dir % 'logs'
    cfg.DIR.CHECKPOINTS = output_dir % 'checkpoints'
    train_writer = SummaryWriter(os.path.join(cfg.DIR.LOGS, 'train'))
    val_writer = SummaryWriter(os.path.join(cfg.DIR.LOGS, 'test'))

    # Training loop
    for epoch_idx in range(init_epoch, cfg.TRAIN.NUM_EPOCHS):
        # Tick / tock
        epoch_start_time = time()

        # Batch average meterics
        batch_time = AverageMeter()
        data_time = AverageMeter()
        encoder_losses = AverageMeter()
        refiner_losses = AverageMeter()

        # # switch models to training mode
        vlm2pix.convert_layer.train()
        vlm2pix.decoder.train()
        vlm2pix.refiner.train()
        # vlm2pix.flava_model.train()

        batch_end_time = time()
        n_batches = len(train_data_loader)
        for batch_idx, (taxomony_class, taxonomy_names, sample_names, rendering_images,
                        ground_truth_volumes) in enumerate(train_data_loader):
            # Measure data time
            data_time.update(time() - batch_end_time)

            # print("pass data to model--------------------------------------------------------------")
            encoder_loss, refiner_loss, generated_volumes = vlm2pix(rendering_images.cuda(),ground_truth_volumes.cuda(), taxomony_class )
    
            # Backward the loss on convert_layer 
            vlm2pix.convert_layer.zero_grad()
            vlm2pix.decoder.zero_grad()
            vlm2pix.refiner.zero_grad()
            # vlm2pix.flava_model.zero_grad()


            if any(param.requires_grad for param in vlm2pix.convert_layer.parameters()):
                encoder_loss.backward(retain_graph=True) 
            if any(param.requires_grad for param in vlm2pix.decoder.parameters()):
                encoder_loss.backward(retain_graph=True) 
            if any(param.requires_grad for param in vlm2pix.refiner.parameters()):
                refiner_loss.backward(retain_graph=True) 
            # if any(param.requires_grad for param in vlm2pix.flava_model.parameters()):
            #     encoder_loss.backward(retain_graph=True) 
                               
            convert_layer_solver.step()
            decoder_solver.step()
            refiner_solver.step()


            # Append loss to average metrics
            encoder_losses.update(encoder_loss.item())
            refiner_losses.update(refiner_loss.item())
            # Append loss to TensorBoard
            n_itr = epoch_idx * n_batches + batch_idx
            train_writer.add_scalar('EncoderDecoder/BatchLoss', encoder_loss.item(), n_itr)
            train_writer.add_scalar('Refiner/BatchLoss', refiner_loss.item(), n_itr)

            # Tick / tock
            batch_time.update(time() - batch_end_time)
            batch_end_time = time()
            logging.info(
                '[Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) EDLoss = %.4f RLoss = %.4f' %
                (epoch_idx + 1, cfg.TRAIN.NUM_EPOCHS, batch_idx + 1, n_batches, batch_time.val, data_time.val,
                encoder_loss.item(), refiner_loss.item())
            )
            
        # # Adjust learning rate
        convert_layer_lr_scheduler.step()
        decoder_lr_scheduler.step()

        # Append epoch loss to TensorBoard
        train_writer.add_scalar('EncoderDecoder/EpochLoss', encoder_losses.avg, epoch_idx + 1)
        train_writer.add_scalar('Refiner/EpochLoss', refiner_losses.avg, epoch_idx + 1)

        # Tick / tock
        epoch_end_time = time()
        logging.info('[Epoch %d/%d] EpochTime = %.3f (s) EDLoss = %.4f RLoss = %.4f' %
                     (epoch_idx + 1, cfg.TRAIN.NUM_EPOCHS, epoch_end_time - epoch_start_time, encoder_losses.avg,
                      refiner_losses.avg))

        # Update Rendering Views
        if cfg.TRAIN.UPDATE_N_VIEWS_RENDERING:
            n_views_rendering = random.randint(1, cfg.CONST.N_VIEWS_RENDERING)
            train_data_loader.dataset.set_n_views_rendering(n_views_rendering)
            logging.info('Epoch [%d/%d] Update #RenderingViews to %d' %
                         (epoch_idx + 2, cfg.TRAIN.NUM_EPOCHS, n_views_rendering))

        # Validate the training models conditionally
        if (epoch_idx < 50 and epoch_idx % 10 == 0) or \
        (50 <= epoch_idx <= 100 and epoch_idx % 2 == 0) or \
        (epoch_idx > 100):
            iou = test_net_flava(cfg, vlm2pix, epoch_idx + 1, val_data_loader, val_writer)


        # Save weights to file
        if (epoch_idx + 1) % cfg.TRAIN.SAVE_FREQ == 0 or iou > best_iou:
            file_name = 'checkpoint-epoch-%03d.pth' % (epoch_idx + 1)
            if iou > best_iou:
                best_iou = iou
                best_epoch = epoch_idx
                file_name = 'checkpoint-best.pth'

            output_path = os.path.join(cfg.DIR.CHECKPOINTS, file_name)
            if not os.path.exists(cfg.DIR.CHECKPOINTS):
                os.makedirs(cfg.DIR.CHECKPOINTS)

            checkpoint = {
                'epoch_idx': epoch_idx,
                'best_iou': best_iou,
                'best_epoch': best_epoch,
                'vlm2pix_state_dict': vlm2pix.state_dict(),
            }
            # if cfg.NETWORK.USE_REFINER:
            #     checkpoint['refiner_state_dict'] = refiner.state_dict()
            # if cfg.NETWORK.USE_MERGER:
            #     checkpoint['merger_state_dict'] = merger.state_dict()

            torch.save(checkpoint, output_path)
            logging.info('Saved checkpoint to %s ...' % output_path)

    # Close SummaryWriter for TensorBoard
    train_writer.close()
    val_writer.close()


# python3 runner_VLM2Vox_v2.py --weights=./pre_train/Pix2Vox++-A-ShapeNet.pth