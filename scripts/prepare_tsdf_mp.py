"""
Run this script to convert depth to TSDF (240x144x240) and optionally prepare GT label (60x36x60) and weights (60x36x60) for training
"""

import numpy as np
import os
from struct import *
# import math
# import cv2
import multiprocessing
# import cc3d

# from tqdm import tqdm
# import time

import sys
sys.path.append("./vis_utils")
import helper_functions as hf
from scene_functions import *
from scene_templates import *

np.set_printoptions(suppress=True)

# Processing parameters
target_tsdf, target_label = 240, 60
label_and_weights = False
write_ply = False

# Map 36(+1) classes to 11(+1) classes
class_mapping = hf._get_class_map()
params = hf._init_params(target_tsdf=target_tsdf, target_label=60)
for k,v in params.items():
    print(k,':',v)

path = "/media/viprlab/Deuxième Disque SSD/ching/datasets/depthbin_eval/depthbin/NYUCAD"
path = os.path.join(path, 'NYUCADtrain', 'preprocessed_ching')
# path = os.path.join(path, 'NYUCADtest', 'preprocessed_ching')
save_paths = ['tsdf_{}'.format(target_tsdf),
              # 'label_{}'.format(target_label),
              # 'weights_{}'.format(target_label)
             ]

for save_path in save_paths:
    save_path = os.path.join(path, save_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

# Check for processed files and only process those that aren't
processed = [int(x[3:7]) for x in os.listdir(save_paths[0])] # check in tsdf folder
file_idxs = [int(x[3:7]) for x in os.listdir(os.path.join(path,'bin'))]
file_idxs = [x for x in file_idxs if x not in processed]
file_idxs.sort()
print("Processing {} files".format(len(file_idxs)))

def multiprocess(file_idx):

    bin_file            = os.path.join(path,'bin','NYU{:04d}_0000.bin'.format(file_idx))
    depth_file          = os.path.join(path,'depth','NYU{:04d}_0000_depth.png'.format(file_idx))
    _3d_semantic_file   = os.path.join(path,'3D_semantic_seg','NYU{:04d}_0000.npz'.format(file_idx))
    
    # Read binfile
    origin, cam_pose, _ = hf._read_bin(bin_file, return_voxels=False)
    vox_label = np.load(_3d_semantic_file)
    vox_label = vox_label['label_hr']

    # 37(+1) classes -> 11(+1) classes
    _255 = np.where(vox_label == 255) # keep location of voxels outside of room
    vox_label[_255] = 0 # temporary set 255 to 0
    vox_label = np.take(class_mapping, vox_label)
    vox_label[_255] = 255 # put back vox label 255

    # Count object voxels
    object_count = (vox_label > 3) * (vox_label < 255)
    object_count = np.sum(object_count)
    # print("Object count:", object_count)

    assert object_count > 0, "Skip file {} since object count is zero".format(file_idx)

    # Compute TSDF:
    # |-- 1. Pixel-to-Voxel
    depth_data = hf._read_bitshift(depth_file, return_as_float=True)
    mapping = hf._2Dto3D(depth_data, origin, cam_pose, params, return_as='1D', mapping_as='voxel')
    _map = np.where(mapping != -1)

    vox_binary = np.zeros([params["tsdf_vox_num"]])
    vox_binary[mapping[_map]] = 1

    # |-- 2. Squared Distance Transform (Array version)
    default_value = 255
    vox_tsdf = np.ones(params["tsdf_vox_num"], dtype=np.float32) * default_value

    # Set surface voxels in TSDF to 0
    vox_tsdf[mapping[_map]] = 0

    # All other voxels to process
    vox_idxs = np.where(vox_tsdf == default_value)[0]

    z = ((vox_idxs / (params["tsdf_vox_size"][0] * params["tsdf_vox_size"][1])) % params["tsdf_vox_size"][2]).astype(np.int32)
    y = ((vox_idxs / params["tsdf_vox_size"][0]) % params["tsdf_vox_size"][1]).astype(np.int32)
    x = (vox_idxs % params["tsdf_vox_size"][0]).astype(np.int32)

    zyx = np.stack((z,y,x), axis=1)
    point_cam_x, point_cam_y, point_cam_z = hf._3Dto2D(zyx, origin, cam_pose, params, return_at_point_cam=True)

    # filter by point_cam_z
    mask = point_cam_z > 0
    vox_idxs, point_cam_x, point_cam_y, point_cam_z = vox_idxs[mask], point_cam_x[mask], point_cam_y[mask], point_cam_z[mask]

    # this implementation works like C++'s roundf - +/-.5 rounding away from zero
    cam_K = params["cam_K_virt"]
    pixel_x = hf._round_half_up(cam_K[0] * point_cam_x / point_cam_z + cam_K[2]).astype(np.int32)
    pixel_y = hf._round_half_up(cam_K[4] * point_cam_y / point_cam_z + cam_K[5]).astype(np.int32)
    # print("Pixel x and y (C++ ver.)"); print(pixel_x, pixel_y)

    # filter by pixel_x and pixel_y
    mask = (pixel_x >= 0) * (pixel_x < 640) * (pixel_y >= 0) * (pixel_y < 480)
    vox_idxs, point_cam_z, pixel_x, pixel_y = vox_idxs[mask], point_cam_z[mask], pixel_x[mask], pixel_y[mask]
    # print(pixel_x, pixel_y)

    point_depth = depth_data[pixel_y, pixel_x]

    # too near/too far
    too_near = np.where(point_depth < 0.5)
    vox_tsdf[vox_idxs[too_near]] = 1.0

    too_far = np.where(point_depth > 8.0)
    vox_tsdf[vox_idxs[too_far]] = 1.0

    mask = (point_depth >= 0.5) * (point_depth <= 8.0)
    vox_idxs, point_cam_z, point_depth = vox_idxs[mask], point_cam_z[mask], point_depth[mask]

    # should do missing depth first?
    # missing depth
    missing_depth = np.where(np.round(point_depth) == 0)
    vox_tsdf[vox_idxs[missing_depth]] = -1.0

    mask = np.round(point_depth) > 0
    vox_idxs, point_cam_z, point_depth = vox_idxs[mask], point_cam_z[mask], point_depth[mask]

    # calculate sign
    diff = abs(point_depth - point_cam_z)

    small_diff = np.where(diff < 0.0001)
    sign = 1 # prevent NaN
    vox_tsdf[vox_idxs[small_diff]] = sign

    big_diff = np.where(diff >= 0.0001)
    sign = (point_depth - point_cam_z) / abs(point_depth - point_cam_z)
    vox_tsdf[vox_idxs[big_diff]] = sign[big_diff]

    # Use numpy arrays to calculate distance from nearest surface voxel
    _z = ((vox_idxs / (params["tsdf_vox_size"][0] * params["tsdf_vox_size"][1])) % params["tsdf_vox_size"][2]).astype(np.int32)
    _y = ((vox_idxs / params["tsdf_vox_size"][0]) % params["tsdf_vox_size"][1]).astype(np.int32)
    _x = (vox_idxs % params["tsdf_vox_size"][0]).astype(np.int32)

    bin_idxs = np.where(vox_binary == 1)[0]
    _zbin = ((bin_idxs / (params["tsdf_vox_size"][0] * params["tsdf_vox_size"][1])) % params["tsdf_vox_size"][2]).astype(np.int32)
    _ybin = ((bin_idxs / params["tsdf_vox_size"][0]) % params["tsdf_vox_size"][1]).astype(np.int32)
    _xbin = (bin_idxs % params["tsdf_vox_size"][0]).astype(np.int32)

    split_size = 50
    split_count = np.ceil(vox_idxs.shape[0] / split_size).astype(np.uint32)
    
    start = 0
    for i in range(split_count):

        stop = start + split_size

        # sample a small portion of remaining voxels to perform tsdf calculation
        z = _z[start:stop]
        y = _y[start:stop]
        x = _x[start:stop]

        # Instead of iterating thru z-,y-,x-axes within search range of each voxel to find surface voxels
        # and the corresponding nearest distance, we directly pick voxels:
        # 1. within search range of voxels (inside split count)
        # 2. only taking known surface voxels into consideration

        # set a range of zbin voxels to compare distances to
        low_z, top_z = np.min(z), np.max(z)
        low_z, top_z = max(0, low_z-params["tsdf_search_region"]), min(params["tsdf_vox_size"][0], top_z+params["tsdf_search_region"]+1)

        # set a range of ybin voxels to compare distances to
        low_y, top_y = np.min(y), np.max(y)
        low_y, top_y = max(0, low_y-params["tsdf_search_region"]), min(params["tsdf_vox_size"][1], top_y+params["tsdf_search_region"]+1)

        # set a range of xbin voxels to compare distances to
        low_x, top_x = np.min(x), np.max(x)
        low_x, top_x = max(0, low_x-params["tsdf_search_region"]), min(params["tsdf_vox_size"][2], top_x+params["tsdf_search_region"]+1)

        mask = (_zbin >= low_z) * (_zbin < top_z) * (_ybin >= low_y) * (_ybin < top_y) * (_xbin >= low_x) * (_xbin < top_x)
        zbin, ybin, xbin = _zbin[mask], _ybin[mask], _xbin[mask]

        # now we have all the voxels within search range, we can start searching for nearest distance for
        # each voxel inside split count

        start = stop

        # just checking only 1 of zbin/ybin/xbin should be sufficient because if there's no surface voxel in vicinity,
        # zbin, ybin, xbin should all be empty arrays
        if xbin.size == 0 or ybin.size == 0 or zbin.size == 0:
            continue

        zbin = zbin[np.newaxis,...]
        ybin = ybin[np.newaxis,...]
        xbin = xbin[np.newaxis,...]

        # shape = split size * no. of voxels found within vicinity
        z = np.repeat(z, repeats=zbin.shape[1])
        y = np.repeat(y, repeats=ybin.shape[1])
        x = np.repeat(x, repeats=xbin.shape[1])

        # keep 'step' before modifying using np.repeat. this variable is used to find index inside split_size
        step = zbin.shape[1]

        zbin = np.repeat(zbin, repeats=z.shape[0]//zbin.shape[1], axis=0).reshape(-1)
        ybin = np.repeat(ybin, repeats=y.shape[0]//ybin.shape[1], axis=0).reshape(-1)
        xbin = np.repeat(xbin, repeats=x.shape[0]//xbin.shape[1], axis=0).reshape(-1)

        distances = np.sqrt((z-zbin)**2 + (y-ybin)**2 + (x-xbin)**2) / params["tsdf_search_region"]
        distances = distances.reshape((-1,step))
        distances = np.min(distances, axis=1)

        idx = i * split_size + np.arange(len(distances))
        sign = vox_tsdf[vox_idxs[idx]] / abs(vox_tsdf[vox_idxs[idx]])

        distances = np.minimum(distances, abs(vox_tsdf[vox_idxs[idx]]))
        vox_tsdf[vox_idxs[idx]] = distances * sign

    if label_and_weights:
        out_vox_binary = np.zeros(params["out_vox_num"], dtype=np.uint8) # (downsampled) surface occupancy - not used
        out_vox_tsdf   = np.zeros(params["out_vox_num"], dtype=np.float32) # (downsampled) tsdf
        out_vox_label  = np.zeros(params["out_vox_num"], dtype=np.uint8) # (downsampled) ground truth label

        # Threshold for downsampled voxel to be considered free space
        empty_thresh = int(0.95 * (params["down_scale"] ** 3))

        vox_idxs = np.arange(params["out_vox_num"], dtype=np.int32)
        _z = ((vox_idxs / (params["out_vox_size"][0] * params["out_vox_size"][1])) % params["out_vox_size"][2]).astype(np.int32)
        _y = ((vox_idxs / params["out_vox_size"][0]) % params["out_vox_size"][1]).astype(np.int32)
        _x = (vox_idxs % params["out_vox_size"][0]).astype(np.int32)

        for vox_idx in vox_idxs:

            outside_room_count = 0
            free_space_count = 0
            not_surface_count = 0

            z = _z[vox_idx]
            y = _y[vox_idx]
            x = _x[vox_idx]

            z = np.array(range(z*params["down_scale"], (z+1)*params["down_scale"]))
            y = np.array(range(y*params["down_scale"], (y+1)*params["down_scale"]))
            x = np.array(range(x*params["down_scale"], (x+1)*params["down_scale"]))

            z = np.repeat(z, repeats=params["down_scale"]*params["down_scale"])
            y = np.repeat(y, repeats=params["down_scale"])
            y = np.repeat(y[np.newaxis,...], repeats=params["down_scale"], axis=0).reshape(-1)
            x = np.repeat(x[np.newaxis,...], repeats=params["down_scale"]*params["down_scale"], axis=0).reshape(-1)

            iidx = z * 240 * 144 + y * 240 + x
            # print(iidx)

            label_val = vox_label[iidx]
            # print(label_val)

            free_space_count = np.sum(label_val == 0)
            # print(free_space_count)

            outside_room_count = np.sum(label_val == 255)
            # print(outside_room_count)

            not_surface_count = (vox_binary[iidx] == 0) + (vox_label[iidx] == 255) # logical_OR
            not_surface_count = np.sum(not_surface_count)
            # print(not_surface_count)

            if free_space_count + outside_room_count > empty_thresh:
                unique_element, count = np.unique(label_val, return_counts=True)
                out_vox_label[vox_idx] = unique_element[np.argmax(count)]
                # print(out_vox_label[vox_idx])
            else:
                # filter out '0' and '255' before using np.argmax
                label_val = label_val[label_val != 0]
                label_val = label_val[label_val != 255] # test with and without filtering 255 (edgenet includes 255, satnet does not)
                unique_element, count = np.unique(label_val, return_counts=True)
                out_vox_label[vox_idx] = unique_element[np.argmax(count)]

            # downsampled surface voxels (not used)
            if not_surface_count > empty_thresh:
                out_vox_binary[vox_idx] = 0
            else:
                out_vox_binary[vox_idx] = 1

            # Mark weights for training
            tsdf_sum = np.sum(vox_tsdf[iidx])
            out_vox_tsdf[vox_idx] = tsdf_sum / (params["down_scale"] ** 3)

        out_vox_weights = np.zeros(params["out_vox_num"], dtype=np.float32) # (60*36*60)

        mark_for_training = (out_vox_label > 0) * (out_vox_label < 255)
        out_vox_weights[mark_for_training] = 1 # an arbitrary value to indicate occupied voxels

        mark_for_background = (out_vox_tsdf < 0) * (out_vox_label < 255)
        mark_for_background = ~mark_for_training * mark_for_background # invert mark_for_training to exclude it from marking background
        out_vox_weights[mark_for_background] = 2 # an arbitrary value to indicate background

        out_vox_label[out_vox_label == 255] = 0

        # Reshape label and weights
        out_vox_label = out_vox_label.reshape(params["out_vox_size"][0],
                                              params["out_vox_size"][1],
                                              params["out_vox_size"][2]) 
        out_vox_weights = out_vox_weights.reshape(params["out_vox_size"][0],
                                                  params["out_vox_size"][1],
                                                  params["out_vox_size"][2]) 

    # Flip TSDF
    vox_tsdf[vox_tsdf > 1] = 1 # 2000 and 255 become 1

    sign = np.zeros_like(vox_tsdf)

    small = np.where(abs(vox_tsdf) < 0.001)
    sign[small] = 1

    big = np.where(abs(vox_tsdf) >= 0.001)
    sign[big] = vox_tsdf[big] / abs(vox_tsdf[big])

    vox_tsdf = sign * np.maximum(0.001, 1-abs(vox_tsdf))

    # Reshape tsdf
    vox_tsdf = vox_tsdf.reshape(params["tsdf_vox_size"][0], 
                                params["tsdf_vox_size"][1],
                                params["tsdf_vox_size"][2])

    if write_ply:
        # Write ply files for visualization
        # file = "/home/mmu/Desktop/NYU{:04d}_0000.ply".format(file_idx)
    
        voxels, ply_file, vox_coords, vox_count = get_scene_properties(file, 'same', 'tsdf', vox_tsdf, vox_res='hr', lite=True)
        scene_writer(voxels, ply_file, vox_coords, vox_count, color_tsdf, 'tsdf', True)
    
        if label_and_weights:
            voxels, ply_file, vox_coords, vox_count = get_scene_properties(file, 'same', 'weights', out_vox_weights, vox_res='lr', lite=True)
            scene_writer(voxels, ply_file, vox_coords, vox_count, color_weights, 'weights', True)
            
            voxels, ply_file, vox_coords, vox_count = get_scene_properties(file, 'same', 'semantic', out_vox_label, vox_res='lr', lite=True)
            scene_writer(voxels, ply_file, vox_coords, vox_count, color_suncg, 'semantic', True)

    # Save file(s)
    processed_file = os.path.join(path, 'tsdf_{}'.format(target_tsdf), 'NYU{:04d}_0000.npz'.format(file_idx))
    np.savez_compressed(processed_file,
                        tsdf=vox_tsdf[...,np.newaxis])

    if label_and_weights:
        processed_file = os.path.join(path, 'weights_{}'.format(target_label), 'NYU{:04d}_0000.npz'.format(file_idx))
        np.savez_compressed(processed_file,
                            weights=out_vox_weights[...,np.newaxis])
        
        processed_file = os.path.join(path, 'label_{}'.format(target_label), 'NYU{:04d}_0000.npz'.format(file_idx))
        np.savez_compressed(processed_file,
                            label=out_vox_label)

if __name__ == '__main__':
    pool = multiprocessing.Pool(processes = multiprocessing.cpu_count()-1)
    list(tqdm(pool.imap(multiprocess, file_idxs), total=len(file_idxs)))
    pool.close()
    pool.join()
