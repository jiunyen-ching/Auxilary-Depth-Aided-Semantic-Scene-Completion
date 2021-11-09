import h5py
import cv2
import os
import numpy as np
from struct import *
from helper_functions import * # test if this works

dataset, mode = 'NYU', 'train'
base_path   = '/home/mmu/Downloads/sscnet/depthbin_eval/depthbin'

bin_dir     = os.path.join(base_path, '%s%s' % (dataset,mode))
depth_dir   = os.path.join(base_path, '%s%s' % (dataset,mode))
output_dir  = os.path.join(base_path, '%s%s_mapping' % (dataset,mode))

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

filelist        = [x[:12] for x in os.listdir(bin_dir) if x.endswith('.bin')]
processed_files = [x[:12] for x in os.listdir(output_dir) if x.endswith('_mapping.npz')]
to_process      = [x for x in filelist if x not in processed_files]
to_process.sort()

for filename in to_process:
    bin_path = os.path.join(bin_dir, filename + '.bin')
    depth_path = os.path.join(depth_dir, filename + '.png')

    im_depth = read_bitshift(depth_path, return_flat=False, return_float=True)

    mapping = calculate_mapping_vectorize(bin_path, im_depth, return_flat=False)

    np.savez_compressed(os.path.join(output_dir, filename + '_mapping.npz'),
                        mapping=mapping)

    print('Done processing:', filename)
