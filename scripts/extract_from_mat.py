import numpy as np
import h5py
import cv2
import os
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Extracting images from NYUv2 .mat file')
parser.add_argument('--type', '-t', required=True, type=str)
args = parser.parse_args()

MAT_FILE    = '../data/temp/nyu_depth_v2_labeled.mat'
TRAIN_PATH  = '../data/depthbin_eval/depthbin/NYUtrain'
VAL_PATH    = '../data/depthbin_eval/depthbin/NYUtest'

train_prefix = [int(x[3:7])-1 for x in os.listdir(TRAIN_PATH) if x.endswith('.bin')]
val_prefix   = [int(x[3:7])-1 for x in os.listdir(VAL_PATH) if x.endswith('.bin')]

f = h5py.File(MAT_FILE, 'r')

if args.type == 'color':
    output_paths = [os.path.join(TRAIN_PATH, args.type), os.path.join(VAL_PATH, args.type)]
    for output_path in output_paths:
        if not os.path.exists(output_path):
            os.mkdir(output_path)

    color = np.array(f['images'])
    color = np.transpose(color, (0,3,2,1))

    for idx in tqdm(range(len(color))):
        img = cv2.cvtColor(color[idx],cv2.COLOR_RGB2BGR)
        if idx in train_prefix:
            img_file = os.path.join(output_paths[0], 'NYU%04d_0000_color.jpg' % (idx+1))
        else:
            img_file = os.path.join(output_paths[1], 'NYU%04d_0000_color.jpg' % (idx+1))
        cv2.imwrite(img_file, img)

elif args.type == 'label':
    output_paths, labels = [], ['semantic', 'instance']
    for label in labels:
        output_paths.append(os.path.join(TRAIN_PATH, label))
        output_paths.append(os.path.join(VAL_PATH, label))
        for output_path in output_paths:
            if not os.path.exists(output_path):
                os.mkdir(output_path)

    semantic = np.array(f['labels'])
    semantic = np.transpose(semantic, (0,2,1))
    instances = np.array(f['instances'])
    instances = np.transpose(instances, (0,2,1))

    for idx in tqdm(range(len(semantic))):
        instance_idx = 0
        instance_output = np.zeros((480,640), dtype=np.uint8)
        for instance in np.unique(instances[idx]):
            if instance == 0:
                continue
            for label in np.unique(semantic[idx]):
                if label == 0:
                    continue

                instance_np = (instances[idx] == instance).astype(np.uint8)
                semantic_np = (semantic[idx] == label).astype(np.uint8)
                instance_found = instance_np * semantic_np

                if np.any(instance_found) == True:
                    instance_found = np.where(instance_found == 1)
                    instance_idx += 1
                    instance_output[instance_found] = instance_idx

        # output_path[0] = TRAIN_PATH | semantic
        # output_path[1] = VAL_PATH   | semantic
        # output_path[2] = TRAIN_PATH | instance
        # output_path[3] = VAL_PATH   | instance
        if idx in train_prefix:
            semantic_file = os.path.join(output_paths[0], 'NYU%04d_0000_semantic.png' % (idx+1))
            instance_file = os.path.join(output_paths[2], 'NYU%04d_0000_instance.png' % (idx+1))
        else:
            semantic_file = os.path.join(output_paths[1], 'NYU%04d_0000_semantic.png' % (idx+1))
            instance_file = os.path.join(output_paths[3], 'NYU%04d_0000_instance.png' % (idx+1))
        cv2.imwrite(semantic_file, semantic[idx])
        cv2.imwrite(instance_file, instance_output)
