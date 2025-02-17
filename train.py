import logging
import matplotlib
import numpy as np
import os
import sys
matplotlib.use('Agg')

from argparse import ArgumentParser
from pprint import pprint

from config import cfg
from core.train_flava import train_net_flava
from core.test import test_net
from core.train_flava_finetune import train_net_flava_finetune
from core.train_flava_dino import train_flava_dino


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


    pprint(cfg)

    # Set GPU to use
    if type(cfg.CONST.DEVICE) == str:
        os.environ["CUDA_VISIBLE_DEVICES"] = cfg.CONST.DEVICE


    ## stat training 
    if cfg.Train_decoder:
        if cfg.CONST.MODEL=="VLM2pix":
            train_net_flava_finetune(cfg) #train_net
        elif cfg.CONST.MODEL=="VL2pixDino":
            train_flava_dino(cfg)
        else:
            raise Exception("Incorrect model name!")
        


if __name__ == '__main__':
    if sys.version_info < (3, 0):
        raise Exception("Please follow the installation instruction on 'https://github.com/hzxie/Pix2Vox'")

    logging.basicConfig(format='[%(levelname)s] %(asctime)s %(message)s', level=logging.DEBUG)
    main()



## How to run the code: python3 runner_VLM2Vox_v2.py --weights=./pre_train/Pix2Vox++-A-ShapeNet.pth