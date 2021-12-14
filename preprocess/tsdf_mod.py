import os
import numpy as np
import math

# Scale values (large remain large, small become smaller)
def mod_v1(filepath):
    filename = os.path.basename(filepath)
    folder_name = os.path.basename(os.path.dirname(filepath))
    root_folder = os.path.dirname(os.path.dirname(filepath))

    file = np.load(filepath)
    tsdf = file['tsdf']
    edges = file['edges']
    lbl = file['lbl']
    weights = file['weights']
    vol = file['vol']
        
    output_path = os.path.join(root_folder + '_v1', folder_name)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for value in np.unique(tsdf):
        replace = np.where(tsdf == value)
        
        sign = value/np.abs(value)
        
        try:
            val = 0.92 - np.abs(value)
            val = math.log(val, 0.3)/5
            val = sign * val
            tsdf[replace] == val
            
        except ValueError:
            print("found surface")
            
    for value in np.unique(edges):
        replace = np.where(edges == value)
        
        sign = value/np.abs(value)
        
        try:
            val = 0.92 - np.abs(value)
            val = math.log(val, 0.3)/5
            val = sign * val
            edges[replace] == val
            
        except ValueError:
            print("found surface")

    np.savez_compressed(os.path.join(output_path, filename),
                       tsdf=tsdf,
                       edges=edges,
                       lbl=lbl,
                       weights=weights,
                       vol=vol)

def mod_v3(filepath):
    mode = 'test'

    avoxel_path = 'F:/research/datasets/NYUCADamodal/NYUCAD%s_amodal' % (mode)
    save_path = 'F:/research/datasets/NYUCADamodal/v3/NYUCAD%s_amodal' % (mode)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for file in os.listdir(avoxel_path):
        _avoxel = os.path.join(avoxel_path, file)
        # print(_avoxel)

        avoxel = np.load(_avoxel)
        atsdf = avoxel['tsdf']

        # Modification 3
        # 1. Foreground scaled to small values
        # 2. Background values are left as default (per distance transform calculation)
        # 3. Make all values non-negative
        fg = np.unique(atsdf)
        for val in fg:
            if val < 0 or val == 1:
                continue
            replace = np.where(atsdf == val)
            _val = 0.99 - val
            _val = math.log(_val, 0.3)/10
            atsdf[replace] = _val

        atsdf = np.abs(atsdf)

        _save_path = os.path.join(save_path, file)
        print("Saving to", _save_path)

        np.savez_compressed(_save_path,
                            tsdf=atsdf,
                            edges=avoxel['edges'],
                            lbl=avoxel['lbl'],
                            weights=avoxel['weights'],
                            vol=avoxel['vol'])

