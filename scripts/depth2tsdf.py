import numpy as np
import cv2
import os
import argparse
from tqdm import tqdm
import helper_functions as hf
# from struct import *
import math
import time
np.set_printoptions(suppress=True)

parser = argparse.ArgumentParser(description='Converting depth to TSDF')
parser.add_argument('--dataset', '-d', required=True, type=str)
parser.add_argument('--file', '-f', required=False, type=str)
args = parser.parse_args()

TRAIN_PATH  = '../data/depthbin_eval/depthbin/%strain' % args.dataset
VAL_PATH    = '../data/depthbin_eval/depthbin/%stest' % args.dataset

filelist = []
for path in [TRAIN_PATH, VAL_PATH]:
    if not os.path.exists(os.path.join(path,'TSDF')):
        os.mkdir(os.path.join(path,'TSDF'))

    for file in os.listdir(path):
        if file.endswith('.bin'):
            filelist.append(os.path.join(path, file[:-4]))

# Map 36(+1) classes to 11(+1) classes
segmentation_class_map = hf._get_class_map()

# Parameters (put inside .py file later)
img_height, img_width = 480, 640
img_scale = 1.0
vox_margin = 0.24
vox_unit = 0.02
vox_size = (240,144,240)
cam_K = np.array([518.8579 / img_scale, 0., img_width / (2 * img_scale),
                  0., 518.8579 / img_scale, img_height / (2 * img_scale),
                  0., 0., 1.], dtype=np.float32)
search_region = int(vox_margin/vox_unit)

for _file in filelist:
    if args.file: # if process a single file
        if os.path.basename(args.file)[:-4] != os.path.basename(_file):
            continue
        file = args.file
    file = _file # temporary
    print(file)

    origin, cam_pose, vox_label = hf._read_bin(file + '.bin')

    # 36(+1) classes -> 11(+1) classes
    _255 = np.where(vox_label == 255) # keep location of voxels outside of room
    vox_label[_255] = 0 # temporary set 255 to 0
    vox_label = np.take(segmentation_class_map, vox_label).astype(np.uint8) # int8 -> uint8 (otherwise 255 will overflow)
    vox_label[_255] = 255 # put back vox label 255

    # Count objects
    object_count = (vox_label > 3) * (vox_label < 255)
    object_count = np.sum(object_count) # count no. of True's

    class ObjectCountError(Exception):
        pass
    if object_count > 0:
        pass # if processing more than 1 file, consider using continue
    else:
        raise ObjectCountError("Object count is 0 or less")

    # Compute TSDF:
    # |-- 1. Pixel-to-Voxel
    depth_data = hf._read_bitshift(file + '.png', return_as_float=True)
    mapping = hf._2Dto3D(file + '.bin', depth_data, return_as='1D', mapping_as='voxel') # pixel-to-voxel mapping
    vox_surface = np.where(mapping != -1)

    vox_binary = np.zeros((240 * 144 *240), dtype=np.uint8) # 0: empty, 1: surface
    vox_binary[mapping[vox_surface]] = 1

    # |-- 2. Squared Distance Transform (Array version)
    default_value = 255
    vox_tsdf = np.ones((240 * 144 * 240), dtype=np.float32) * default_value # what value to put as default?

    # Set surface voxels in TSDF to 0
    vox_tsdf[mapping[vox_surface]] = 0
    vox_idxs = np.where(vox_tsdf == default_value)[0]

    z = ((vox_idxs / (240 * 144)) % 240)
    y = ((vox_idxs / 240) % 144)
    x = (vox_idxs % 240)

    z = z.astype(np.int32)
    y = y.astype(np.int32)
    x = x.astype(np.int32)

    point_base_x = z * vox_unit + origin[0]
    point_base_y = x * vox_unit + origin[1]
    point_base_z = y * vox_unit + origin[2]

    point_base_x = point_base_x - cam_pose[0 * 4 + 3]
    point_base_y = point_base_y - cam_pose[1 * 4 + 3]
    point_base_z = point_base_z - cam_pose[2 * 4 + 3]

    point_cam_x = cam_pose[0 * 4 + 0] * point_base_x + cam_pose[1 * 4 + 0] * point_base_y + cam_pose[2 * 4 + 0] * point_base_z
    point_cam_y = cam_pose[0 * 4 + 1] * point_base_x + cam_pose[1 * 4 + 1] * point_base_y + cam_pose[2 * 4 + 1] * point_base_z
    point_cam_z = cam_pose[0 * 4 + 2] * point_base_x + cam_pose[1 * 4 + 2] * point_base_y + cam_pose[2 * 4 + 2] * point_base_z

    # filter by point_cam_z
    vox_idxs    = vox_idxs[point_cam_z > 0]
    point_cam_x = point_cam_x[point_cam_z > 0]
    point_cam_y = point_cam_y[point_cam_z > 0]
    point_cam_z = point_cam_z[point_cam_z > 0]

    # this implementation works like C++'s roundf - +/-.5 rounding away from zero
    pixel_x = hf._round_half_up(cam_K[0] * point_cam_x / point_cam_z + cam_K[2]).astype(np.int32)
    pixel_y = hf._round_half_up(cam_K[4] * point_cam_y / point_cam_z + cam_K[5]).astype(np.int32)
    print("Pixel x and y (C++ ver.)"); print(pixel_x, pixel_y)

    # Filter invalid pixels (outside FOV)
    outside_fov = np.where(pixel_x < 0)
    vox_tsdf[vox_idxs[outside_fov]] = 2000
    outside_fov = np.where(pixel_x >= 640)
    vox_tsdf[vox_idxs[outside_fov]] = 2000
    outside_fov = np.where(pixel_y < 0)
    vox_tsdf[vox_idxs[outside_fov]] = 2000
    outside_fov = np.where(pixel_y >= 480)
    vox_tsdf[vox_idxs[outside_fov]] = 2000

    # filter by pixel_x and pixel_y
    vox_idxs    = vox_idxs[pixel_x >= 0]
    point_cam_z = point_cam_z[pixel_x >= 0]
    pixel_y     = pixel_y[pixel_x >= 0]
    pixel_x     = pixel_x[pixel_x >= 0]

    vox_idxs    = vox_idxs[pixel_x < 640]
    point_cam_z = point_cam_z[pixel_x < 640]
    pixel_y     = pixel_y[pixel_x < 640]
    pixel_x     = pixel_x[pixel_x < 640]

    vox_idxs    = vox_idxs[pixel_y >= 0]
    point_cam_z = point_cam_z[pixel_y >= 0]
    pixel_x     = pixel_x[pixel_y >= 0]
    pixel_y     = pixel_y[pixel_y >= 0]

    vox_idxs    = vox_idxs[pixel_y < 480]
    point_cam_z = point_cam_z[pixel_y < 480]
    pixel_x     = pixel_x[pixel_y < 480]
    pixel_y     = pixel_y[pixel_y < 480]

    point_depth = depth_data[pixel_y, pixel_x]

    # Filter invalid pixels (too near/too far)
    too_near = np.where(point_depth < 0.5)
    vox_tsdf[vox_idxs[too_near]] = 1.0
    vox_idxs    = vox_idxs[point_depth >= 0.5]
    point_cam_z = point_cam_z[point_depth >= 0.5]
    point_depth = point_depth[point_depth >= 0.5]

    too_far = np.where(point_depth > 8.0)
    vox_tsdf[vox_idxs[too_far]] = 1.0
    vox_idxs    = vox_idxs[point_depth <= 8.0]
    point_cam_z = point_cam_z[point_depth <= 8.0]
    point_depth = point_depth[point_depth <= 8.0]

    # missing depth
    missing_depth = np.where(np.round(point_depth) == 0)
    vox_tsdf[vox_idxs[missing_depth]] = -1.0
    vox_idxs    = vox_idxs[np.round(point_depth) > 0]
    point_cam_z = point_cam_z[np.round(point_depth) > 0]
    point_depth = point_depth[np.round(point_depth) > 0]

    # calculate sign
    diff = abs(point_depth - point_cam_z)

    small_diff = np.where(diff < 0.0001)
    sign = 1 # prevent NaN
    vox_tsdf[vox_idxs[small_diff]] = sign

    big_diff = np.where(diff >= 0.0001)
    sign = (point_depth - point_cam_z) / abs(point_depth - point_cam_z)
    vox_tsdf[vox_idxs[big_diff]] = sign[big_diff]

    # Calculate TSDF
    vox_tsdf_np = vox_tsdf.copy()

    start = time.time()

    _z = (vox_idxs / (240 * 144)) % 240
    _y = (vox_idxs / 240) % 144
    _x = (vox_idxs % 240)

    _z = _z.astype(np.int32)
    _y = _y.astype(np.int32)
    _x = _x.astype(np.int32)

    bin_idxs = np.where(vox_binary == 1)[0]

    _zbin = (bin_idxs / (240 * 144)) % 240
    _ybin = (bin_idxs / 240) % 144
    _xbin = (bin_idxs % 240)

    _zbin = _zbin.astype(np.int32)
    _ybin = _ybin.astype(np.int32)
    _xbin = _xbin.astype(np.int32)

    split_size = 1000
    split_count = np.ceil(vox_idxs.shape[0] / split_size).astype(np.uint32)
    print("How many voxels to process:", vox_idxs.shape[0])
    print("How many splits:", vox_idxs.shape[0] / split_size, '->', split_count)

    start = 0
    for i in tqdm(range(split_count)):

        stop = start + split_size

        # sample a small portion of remaining voxels to perform tsdf calculation
        z = _z[start:stop]
        y = _y[start:stop]
        x = _x[start:stop]

        # set a range of zbin voxels to compare distances to
        low_z, top_z = np.min(z), np.max(z)
        low_z, top_z = max(0, low_z-12), min(240, top_z+12+1)

        xbin = _xbin[_zbin >= low_z]
        ybin = _ybin[_zbin >= low_z]
        zbin = _zbin[_zbin >= low_z]

        xbin = xbin[zbin < top_z]
        ybin = ybin[zbin < top_z]
        zbin = zbin[zbin < top_z]

        # set a range of ybin voxels to compare distances to
        low_y, top_y = np.min(y), np.max(y)
        low_y, top_y = max(0, low_y-12), min(144, top_y+12+1)

        xbin = xbin[ybin >= low_y]
        zbin = zbin[ybin >= low_y]
        ybin = ybin[ybin >= low_y] # modify ybin last because array is modified

        xbin = xbin[ybin < top_y]
        zbin = zbin[ybin < top_y]
        ybin = ybin[ybin < top_y] # modify xbin last because array is modified

        # set a range of xbin voxels to compare distances to
        low_x, top_x = np.min(x), np.max(x)
        low_x, top_x = max(0, low_x-12), min(240, top_x+12+1)

        ybin = ybin[xbin >= low_x]
        zbin = zbin[xbin >= low_x]
        xbin = xbin[xbin >= low_x] # modify xbin last because array is modified

        ybin = ybin[xbin < top_x]
        zbin = zbin[xbin < top_x]
        xbin = xbin[xbin < top_x] # modify xbin last because array is modified

        start = stop

        if xbin.size == 0 or ybin.size == 0 or zbin.size == 0: # just checking zbin should be sufficient since modifying xbin also affects zbin
            continue

        zbin = zbin[np.newaxis,...]
        ybin = ybin[np.newaxis,...]
        xbin = xbin[np.newaxis,...]

        z = np.repeat(z, repeats=zbin.shape[1])
        y = np.repeat(y, repeats=ybin.shape[1])
        x = np.repeat(x, repeats=xbin.shape[1])

        # keep 'step' before modifying using np.repeat
        # to find index inside split_size
        step = zbin.shape[1]

        zbin = np.repeat(zbin, repeats=z.shape[0]//zbin.shape[1], axis=0).reshape(-1)
        ybin = np.repeat(ybin, repeats=y.shape[0]//ybin.shape[1], axis=0).reshape(-1)
        xbin = np.repeat(xbin, repeats=x.shape[0]//xbin.shape[1], axis=0).reshape(-1)

        distances = np.sqrt((z-zbin)**2 + (y-ybin)**2 + (x-xbin)**2) / search_region
        # print('distances shape:', distances.shape)

        for j in range(0, len(distances), step):
            tsdf_value = distances[j:j+step]
            nearest_idx = np.argmin(tsdf_value)
            tsdf_value = tsdf_value[nearest_idx]

            idx = i * split_size + (j // step) # to access vox_idxs

            # print(idx, i, j)
            # print(j//step)

            sign = vox_tsdf_np[vox_idxs[idx]] / abs(vox_tsdf_np[vox_idxs[idx]])
            if tsdf_value < abs(vox_tsdf_np[vox_idxs[idx]]):
                vox_tsdf_np[vox_idxs[idx]] = tsdf_value * sign

    end = time.time()
    print("Elapsed = %s" % (end - start))
    print(np.unique(vox_tsdf_np))

    vox_tsdf = vox_tsdf_np.copy()

    # DownsampleLabel (Array version)
    # |-- 1. Downsample Kernel
    downsampling_scale = 4

    in_vox_size = np.array([240, 144, 240])
    in_vox_num = in_vox_size[0] * in_vox_size[1] * in_vox_size[2] # 240x144x240 = 8,294,400
    out_vox_size = in_vox_size // downsampling_scale # [60,36,60]
    out_vox_num = in_vox_num // (downsampling_scale ** 3) # 8,294,400 / (4*4*4) = 129,600

    out_vox_binary = np.zeros(out_vox_num, dtype=np.uint8) # (downsampled) surface occupancy - not used
    out_vox_tsdf = np.zeros(out_vox_num, dtype=np.float32) # (downsampled) tsdf
    out_vox_label = np.zeros(out_vox_num, dtype=np.uint8) # (downsampled) ground truth label

    # Threshold for downsampled voxel to be considered free space
    empty_thresh = int(0.95 * (downsampling_scale ** 3))


    vox_idx = np.array(range(out_vox_size[0] * out_vox_size[1] * out_vox_size[2]))

    _z = (vox_idx / (out_vox_size[0] * out_vox_size[1])) % out_vox_size[2]
    _y = (vox_idx / out_vox_size[0]) % out_vox_size[1]
    _x = (vox_idx % out_vox_size[0])

    _z = _z.astype(np.int32)
    _y = _y.astype(np.int32)
    _x = _x.astype(np.int32)


    for vox_idx in range(out_vox_size[0] * out_vox_size[1] * out_vox_size[2]):

        outside_room_count = 0
        free_space_count = 0
        not_surface_count = 0

        z = _z[vox_idx]
        y = _y[vox_idx]
        x = _x[vox_idx]

        z = np.array(range(z*4, (z+1)*4))
        y = np.array(range(y*4, (y+1)*4))
        x = np.array(range(x*4, (x+1)*4))

        z = np.repeat(z, repeats=4*4)
        y = np.repeat(y, repeats=4)
        y = np.repeat(y[np.newaxis,...], repeats=4, axis=0).reshape(-1)
        x = np.repeat(x[np.newaxis,...], repeats=4*4, axis=0).reshape(-1)


        iidx = z * 240 * 144 + y * 240 + x
        # print(iidx)

        label_val = vox_label[iidx]
        # print(label_val)

        free_space_count = len(np.where(label_val == 0)[0])
        # print(free_space_count)

        outside_room_count = len(np.where(label_val == 255)[0])
        # print(outside_room_count)

        not_surface_count = (vox_binary[iidx] == 0) + (vox_label[iidx] == 255) # logical_OR
        not_surface_count = np.count_nonzero(not_surface_count)
        # print(not_surface_count)

        if free_space_count + outside_room_count > empty_thresh:
            unique_element, count = np.unique(label_val, return_counts=True)
            out_vox_label[vox_idx] = unique_element[np.argmax(count)]
            # print(out_vox_label[vox_idx])
        else:
            # filter out '0' and '255' before using np.argmax
            label_val = label_val[label_val != 0]
            label_val = label_val[label_val != 255]
            unique_element, count = np.unique(label_val, return_counts=True)
            out_vox_label[vox_idx] = unique_element[np.argmax(count)]

        # downsampled surface voxels (not used)
        if not_surface_count > empty_thresh:
            out_vox_binary[vox_idx] = 0
        else:
            out_vox_binary[vox_idx] = 1

        tsdf_sum = np.sum(vox_tsdf[iidx])
        out_vox_tsdf[vox_idx] = tsdf_sum / (downsampling_scale ** 3)

    # |-- 2. Mark voxels for training (Array version)
    out_vox_weights = np.zeros(out_vox_num, dtype=np.uint8) # (60,36,60)

    mark_for_training = (out_vox_label > 0) * (out_vox_label < 255)
    mark_for_training = np.where(mark_for_training == True)
    out_vox_weights[mark_for_training] = 1

    mark_for_background = (out_vox_tsdf < 0) * (out_vox_label < 255)
    mark_for_background = np.where(mark_for_background == True)
    out_vox_weights[mark_for_background] = 1

    mark_for_foreground = np.where(out_vox_tsdf > 1)
    out_vox_weights[mark_for_foreground] = 0

    # Convert Class 255 to Class 0
    out_vox_label[out_vox_label == 255] = 0

    # Flip TSDF (240 x 144 x 240) (Array version)
    # _vox_tsdf = jit_vox_tsdf.copy() # to prevent re-running notebook in case of mistake
    _vox_tsdf = vox_tsdf_np.copy()
    _vox_tsdf[_vox_tsdf > 1] = 1

    sign = np.zeros_like(_vox_tsdf)

    small = np.where(abs(_vox_tsdf) < 0.001)
    sign[small] = 1

    big = np.where(abs(_vox_tsdf) >= 0.001)
    sign[big] = _vox_tsdf[big] / abs(_vox_tsdf[big])

    _vox_tsdf = sign * np.maximum(0.001, 1-abs(_vox_tsdf))
    print(np.unique(_vox_tsdf))

    dirname, filename = os.path.dirname(file), os.path.basename(file)
    # cv2.imwrite(os.path.join(dirname, 'TDSF', filename[:-4] + '_TSDF.npz'), normal)

    # Save TSDF
    tsdf_file = os.path.join(dirname, 'TSDF', filename + '_TSDF.npz')
    np.savez_compressed(tsdf_file,
                        tsdf=_vox_tsdf.reshape(240,144,240,1),
                        lbl=out_vox_label.reshape(60,36,60,1),
                        weights=out_vox_weights.reshape(60,36,60))
