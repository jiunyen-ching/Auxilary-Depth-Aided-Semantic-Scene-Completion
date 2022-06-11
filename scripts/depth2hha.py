import numpy as np
import cv2
import os
import argparse
from tqdm import tqdm
import helper_functions as hf

import math
from hha_utils.rgbd_util import *
from hha_utils.getCameraParam import *
from hha_utils.getHHA import *

parser = argparse.ArgumentParser(description='Converting depth to HHA map')
parser.add_argument('--dataset', '-d', required=True, type=str)
args = parser.parse_args()

TRAIN_PATH  = '../data/depthbin_eval/depthbin/%strain' % args.dataset
VAL_PATH    = '../data/depthbin_eval/depthbin/%stest' % args.dataset

filelist = []
for path in [TRAIN_PATH, VAL_PATH]:
    if not os.path.exists(os.path.join(path,'hha')):
        os.mkdir(os.path.join(path,'hha'))

    for file in os.listdir(path):
        if file.endswith('.bin'):
            filelist.append(os.path.join(path, file[:-4] + '.png'))

for file in tqdm(filelist):
    depth = hf._read_bitshift(file)
    camera_matrix = getCameraParam('color')
    hha = getHHA(camera_matrix, depth / 1000, depth / 1000)

    # Save image
    dirname, filename = os.path.dirname(file), os.path.basename(file)
    cv2.imwrite(os.path.join(dirname, 'hha', filename[:-4] + '_hha.png'), hha)
