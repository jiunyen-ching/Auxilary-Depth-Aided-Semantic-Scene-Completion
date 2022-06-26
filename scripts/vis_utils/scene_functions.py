import numpy as np
import os
from struct import *
import cv2

def write_vertex(f, loc, color, flip_lr):
    shift = 0.5

    if flip_lr:
        _x, _y, _z = loc[2], loc[1], loc[0]
    else:
        _x, _y, _z = loc[0], loc[1], loc[2]

    color_x, color_y, color_z = color[0], color[1], color[2]
    f.write('%f %f %f %d %d %d' % (_x - shift, _y - shift, _z - shift, color_x, color_y, color_z))
    f.write('\n%f %f %f %d %d %d' % (_x - shift, _y + shift, _z - shift, color_x, color_y, color_z))
    f.write('\n%f %f %f %d %d %d' % (_x + shift, _y + shift, _z - shift, color_x, color_y, color_z))
    f.write('\n%f %f %f %d %d %d' % (_x + shift, _y - shift, _z - shift, color_x, color_y, color_z))
    f.write('\n%f %f %f %d %d %d' % (_x - shift, _y - shift, _z + shift, color_x, color_y, color_z))
    f.write('\n%f %f %f %d %d %d' % (_x - shift, _y + shift, _z + shift, color_x, color_y, color_z))
    f.write('\n%f %f %f %d %d %d' % (_x + shift, _y + shift, _z + shift, color_x, color_y, color_z))
    f.write('\n%f %f %f %d %d %d' % (_x + shift, _y - shift, _z + shift, color_x, color_y, color_z))
    f.write('\n')

def write_face(f, voxel_count):
    counter = 0
    for i in range(voxel_count):
        f.write("4 %d %d %d %d\n" % (counter, counter + 1, counter + 2, counter + 3))
        f.write("4 %d %d %d %d\n" % (counter, counter + 4, counter + 5, counter + 1))
        f.write("4 %d %d %d %d\n" % (counter, counter + 3, counter + 7, counter + 4))
        f.write("4 %d %d %d %d\n" % (counter + 6, counter + 5, counter + 4, counter + 7))
        f.write("4 %d %d %d %d\n" % (counter + 6, counter + 7, counter + 3, counter + 2))
        f.write("4 %d %d %d %d\n" % (counter + 6, counter + 2, counter + 1, counter + 5))
        counter += 8

def get_scene_properties(input_path, output_path, type, voxels, iter=None, hr_or_lr=None):
    filename = os.path.basename(input_path)

    if output_path == 'same':
        output_path = os.path.dirname(input_path)

    if type == 'tsdf':
        ply_file = os.path.join(output_path, filename[:-4] + '_tsdf_%d.ply' % iter)
        unique = np.where(abs(voxels) > 0.8) # visualize voxels with absolute tsdf values higher than <threshold value; typically 0.8>

        # uncomment if want to visualize a range of values
        # mask = np.logical_and(abs(voxels) > 0, abs(voxels) < 255)
        # B = voxels[mask] # values
        # B = np.unique(B)
        # d = np.where(B[:, None] == voxels.ravel())[1]
        # unique = np.unravel_index(d, voxels.shape)

        voxel_count = len(unique[0])

    elif type == 'occupancy' or type == 'semantic':
        ply_file = os.path.join(output_path, filename[:-4] + '_%s_%s.ply' % (type, hr_or_lr))
        unique = np.where(voxels != 0)
        voxel_count = len(unique[0])

    elif type == 'weights':
        ply_file = os.path.join(output_path, filename[:-4] + '_weights.ply')
        unique = np.where(voxels != 0)
        voxel_count = len(unique[0])

    elif type == 'mapping':
        ply_file = os.path.join(output_path, filename[:-4] + '_mapping.ply')
        unique = np.where(voxels != 0)
        voxel_count = len(unique[0])

    return ply_file, unique, voxel_count

def get_bin_info(bin_file):
    with open(bin_file,'rb') as f:
        float_size = 4
        uint_size = 4
        total_count = 0
        cor = f.read(float_size*3)
        cors = unpack('fff',cor)
        cam = f.read(float_size*16)
        cams = unpack('ffffffffffffffff', cam)
    f.close()

    return cors, cams

def read_bitshift(depth_path, return_flat=False, return_float=False):
    depth = cv2.imread(depth_path, -1)
    lower_depth = depth >> 3
    higher_depth = (depth % 8) << 13
    real_depth = (lower_depth | higher_depth)

    if return_flat:
        real_depth = np.reshape(real_depth, (-1,))

    if return_float:
        real_depth = real_depth.astype(np.float32)/1000

    return real_depth

def do_mapping(bin_file, depth_img, _preprocessor):
    depth = cv2.imread(depth_img, -1)
    if _preprocessor == 'edgenet':
        lower_depth = depth >> 3
        higher_depth = (depth % 8) << 13
        real_depth = (lower_depth | higher_depth).astype(np.float32) / 1000
    else:
        real_depth = depth.astype(np.float32) / 1000
    real_depth = np.reshape(real_depth,(-1,))

    img_height, img_width = depth.shape
    img_scale = 1.0
    vox_unit = 0.02
    vox_size = (240,144,240)
    cam_K = np.array([518.8579 / img_scale, 0., img_width / (2 * img_scale),
                      0., 518.8579 / img_scale, img_height / (2 * img_scale),
                      0., 0., 1.], dtype=np.float32)

    # vox_origin, cam_pose, _ = get_bin_info(bin_file)
    vox_origin, cam_pose = get_bin_info(bin_file)

    depth_mapping = np.zeros((img_height * img_width), dtype=np.uint64)

    for i in range(len(real_depth)):
        pixel_x, pixel_y = i % 640, i // 640
        point_depth = real_depth[pixel_y * img_width + pixel_x]

        point_cam = np.zeros(3, dtype=float)
        point_cam[0] = (pixel_x - cam_K[2]) * point_depth/cam_K[0]
        point_cam[1] = (pixel_y - cam_K[5]) * point_depth/cam_K[4]
        point_cam[2] = point_depth

        point_base = np.zeros(3, dtype=float)
        point_base[0] = cam_pose[0 * 4 + 0] * point_cam[0] + cam_pose[0 * 4 + 1] * point_cam[1] + cam_pose[0 * 4 + 2] * point_cam[2];
        point_base[1] = cam_pose[1 * 4 + 0] * point_cam[0] + cam_pose[1 * 4 + 1] * point_cam[1] + cam_pose[1 * 4 + 2] * point_cam[2];
        point_base[2] = cam_pose[2 * 4 + 0] * point_cam[0] + cam_pose[2 * 4 + 1] * point_cam[1] + cam_pose[2 * 4 + 2] * point_cam[2];

        point_base[0] = point_base[0] + cam_pose[0 * 4 + 3];
        point_base[1] = point_base[1] + cam_pose[1 * 4 + 3];
        point_base[2] = point_base[2] + cam_pose[2 * 4 + 3];

        z = int(np.floor((point_base[0] - vox_origin[0]) / vox_unit));
        x = int(np.floor((point_base[1] - vox_origin[1]) / vox_unit));
        y = int(np.floor((point_base[2] - vox_origin[2]) / vox_unit));

        if x >= 0 and x < vox_size[0] and y >= 0 and y < vox_size[1] and z >= 0 and z < vox_size[2]:
            vox_idx = z * vox_size[0] * vox_size[1] + y * vox_size[0] + x;
            depth_mapping[pixel_y * img_width + pixel_x] = vox_idx;

    depth_mapping = np.reshape(depth_mapping, (img_height, img_width))
    print("Done mapping")

    return depth_mapping
