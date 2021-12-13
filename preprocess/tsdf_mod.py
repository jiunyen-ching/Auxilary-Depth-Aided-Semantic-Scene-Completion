import os
import numpy as np
import math

# Scale values (large remain large, small become smaller)
def replace_v1(filepath):
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
