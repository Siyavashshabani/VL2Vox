#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Developed by Haozhe Xie <cshzxie@gmail.com>

import logging
import matplotlib
import numpy as np
import os
import sys
# Fix problem: no $DISPLAY environment variable
matplotlib.use('Agg')

from argparse import ArgumentParser
from pprint import pprint

from config import cfg
from core.train import train_net, train_net_flava
from core.test_flava import test_net_flava
import torch
from models.VL2Vox import VL2Pix


def get_args_from_command_line():
    parser = ArgumentParser(description='Parser of Runner of Pix2Vox')
    parser.add_argument('--gpu',
                        dest='gpu_id',
                        help='GPU device id to use [cuda0]',
                        default=cfg.CONST.DEVICE,
                        type=str)
    parser.add_argument('--rand', dest='randomize', help='Randomize (do not use a fixed seed)', action='store_true')
    parser.add_argument('--test', dest='test', help='Test neural networks', action='store_true')
    parser.add_argument('--batch-size',
                        dest='batch_size',
                        help='name of the net',
                        default=cfg.CONST.BATCH_SIZE,
                        type=int)
    parser.add_argument('--epoch', dest='epoch', help='number of epoches', default=cfg.TRAIN.NUM_EPOCHS, type=int)
    parser.add_argument('--weights', dest='weights', help='Initialize network from the weights file', default=None)
    parser.add_argument('--out', dest='out_path', help='Set output path', default=cfg.DIR.OUT_PATH)
    args = parser.parse_args()
    return args


def main():
    # Get args from command line
    args = get_args_from_command_line()

    if args.gpu_id is not None:
        cfg.CONST.DEVICE = args.gpu_id
    if not args.randomize:
        np.random.seed(cfg.CONST.RNG_SEED)
    if args.batch_size is not None:
        cfg.CONST.BATCH_SIZE = args.batch_size
    if args.epoch is not None:
        cfg.TRAIN.NUM_EPOCHS = args.epoch
    if args.out_path is not None:
        cfg.DIR.OUT_PATH = args.out_path
    if args.weights is not None:
        cfg.CONST.WEIGHTS = args.weights
        if not args.test:
            cfg.TRAIN.RESUME_TRAIN = True

    # Print config
    print('Use config:')
    pprint(cfg)

    # Set GPU to use
    if type(cfg.CONST.DEVICE) == str:
        os.environ["CUDA_VISIBLE_DEVICES"] = cfg.CONST.DEVICE

    # # Start train/test process
    # if not args.test:
    #     train_net_flava(cfg) #train_net
    # else:
    if 'WEIGHTS' in cfg.CONST and os.path.exists(cfg.CONST.WEIGHTS):
        bce_loss = torch.nn.BCELoss()
        vlm2pix = VL2Pix(cfg, bce_loss)
        checkpoint = torch.load(cfg.CONST.WEIGHTS)
        # print("checkpoint.keys()------------------------------------------:",checkpoint.keys())
        # print(checkpoint["best_iou"])

        # Load the entire model state dictionary
        vlm2pix.load_state_dict(checkpoint["vlm2pix_state_dict"])   
        # Print the keys to see the structure
        print("Keys in vlm2pix_state_dict---------------------------------:")
        vlm2pix_state_dict = checkpoint["vlm2pix_state_dict"]
        for key in vlm2pix_state_dict.keys():
            print(key)             
        test_net_flava(cfg, vlm2pix)
    else:
        logging.error('Please specify the file path of checkpoint.')
        sys.exit(2)


if __name__ == '__main__':
    if sys.version_info < (3, 0):
        raise Exception("Please follow the installation instruction on 'https://github.com/hzxie/Pix2Vox'")

    logging.basicConfig(format='[%(levelname)s] %(asctime)s %(message)s', level=logging.DEBUG)
    main()

## how to test the model: python3 test_VLM2Vox_v2.py --test --weights ./output/checkpoints/2024-12-05T12:30:06.383139/checkpoint-best.pth
## and in the config.py file, we should set the CONST.STATE on "Test"
