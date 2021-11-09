import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def normalize_rgb(rgb):
    
    assert rgb.shape[0] == 3, 'Make sure image is channel-first'

    # Using ImageNet's normalization values
    MEAN = 255 * np.array([0.485, 0.456, 0.406])
    STD = 255 * np.array([0.229, 0.224, 0.225])

    rgb = rgb.astype(np.float32)
    rgb = (img - MEAN[:, None, None]) / STD[:, None, None] # broadcasting
    
    return rgb
  
r"""
Add RGB and pixel-to-voxel mapping to EdgeNet's processed npz files
"""

dataset, mode = 'NYU', 'test'
base_path = '/home/mmu/Downloads/Dataset/NYUedge'

npz_path     = os.path.join(base_path, '%s%s' % (dataset,mode))
paint_path   = os.path.join(base_path, '%s%s_paint_post' % (dataset,mode))
mapping_path = os.path.join(base_path, '%s%s_mapping' % (dataset,mode))
output_path  = os.path.join(base_path, '%s%s_preproc_new' % (dataset,mode))
if not os.path.exists(output_path):
    os.mkdir(output_path)

filelist   = [x[:12] for x in os.listdir(npz_path)]
processed  = [x[:12] for x in os.listdir(output_path)]
to_process = [x for x in filelist if x not in processed]
to_process.sort()
print("No. of files to process:", len(to_process))

for filename in to_process:
    npz     = np.load(os.path.join(npz_path, filename + '.npz'))
    tsdf    = npz['tsdf']
    edges   = npz['edges']
    lbl     = npz['lbl']
    weights = npz['weights']
    vol     = npz['vol']
    
    rgb = Image.open(os.path.join(paint_path, filename + '_color.jpg'))
    rgb = np.array(rgb)
    rgb = np.transpose(rgb, (2,0,1))
    rgb = normalize_rgb(rgb)
    
    mapping = np.load(os.path.join(mapping_path, filename + '_mapping.npz'))
    mapping = mapping['mapping']
    mapping = np.expand_dims(mapping, axis=0) # channel-first
    
    np.savez_compressed(os.path.join(output_path, filename + '.npz'),
                       tsdf    = tsdf,
                       edges   = edges,
                       lbl     = lbl,
                       weights = weights,
                       vol     = vol,
                       rgb     = rgb,
                       mapping = mapping)
    
    print('Done processing:', filename)
    
