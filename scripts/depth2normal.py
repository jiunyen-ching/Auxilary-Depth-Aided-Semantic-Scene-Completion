import numpy as np
import cv2
import os
import argparse
from tqdm import tqdm
import helper_functions as hf

parser = argparse.ArgumentParser(description='Converting depth to normal map')
parser.add_argument('--dataset', '-d', required=True, type=str)
args = parser.parse_args()

TRAIN_PATH  = '../data/depthbin_eval/depthbin/%strain' % args.dataset
VAL_PATH    = '../data/depthbin_eval/depthbin/%stest' % args.dataset

filelist = []
for path in [TRAIN_PATH, VAL_PATH]:
    if not os.path.exists(os.path.join(path,'normal')):
        os.mkdir(os.path.join(path,'normal'))

    for file in os.listdir(path):
        if file.endswith('.bin'):
            filelist.append(os.path.join(path, file[:-4] + '.png'))

for file in tqdm(filelist):
    normal = hf._gen_normal(file)

    # Save image
    dirname, filename = os.path.dirname(file), os.path.basename(file)
    cv2.imwrite(os.path.join(dirname, 'normal', filename[:-4] + '_normal.png'), normal)
