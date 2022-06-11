import numpy as np
import cv2
from struct import *

def _read_bin(bin_file):
    with open(bin_file,'rb') as f:
        float_size = 4
        # float_size = 8
        cor = f.read(float_size*3)
        cors = unpack('fff',cor) # origin in world coordinates
        # cors = unpack('ddd',cor) # origin in world coordinates

        cam = f.read(float_size*16)
        cams = unpack('ffffffffffffffff', cam) # camera pose
        # cams = unpack('dddddddddddddddd', cam) # camera pose

        uint_size = 4
        vox = f.read()
        numC = int(len(vox)/uint_size)
        checkVoxValIter = unpack('I'*numC, vox)
        checkVoxVal = checkVoxValIter[0::2]
        checkVoxIter = checkVoxValIter[1::2]
        voxels = [i for (val, repeat) in zip(checkVoxVal,checkVoxIter) for i in np.tile(val, repeat)]
        voxels = np.array(voxels, dtype=np.int16)

    return cors, cams, voxels

def _read_mat(mat_file, return_as_flat=True, return_as_float=False):
    with h5py.File(mat_file, 'r') as f:
        data = f['depth_mat']
        amodal_list = []
        for i in data:
            depth = np.transpose(i, (1,0))
            depth = np.ceil(depth*1000).astype(np.uint16)

            if return_as_flat == True:
                depth = np.reshape(depth, (-1,))
            if return_as_float == True:
                depth = depth.astype(np.float32)/1000

            amodal_list.append(depth)
        f.close()
    amodal_list = np.array(amodal_list)
    return amodal_list

def _read_bitshift(depth_path, return_as_flat=False, return_as_float=False):
    depth = cv2.imread(depth_path, -1)
    lower_depth = depth >> 3
    higher_depth = (depth % 8) << 13
    real_depth = (lower_depth | higher_depth)

    if return_as_flat:
        real_depth = np.reshape(real_depth, (-1,))
    if return_as_float:
        real_depth = real_depth.astype(np.float32) / 1000
    return real_depth

def _get_class_map():
    return np.array([ 0,  1,  2,  3,  4,
                     11,  5,  6,  7,  8,
                      8, 10, 10, 10, 11,
                     11,  9,  8, 11, 11,
                     11, 11, 11, 11, 11,
                     11, 11, 10, 10, 11,
                      8, 10, 11,  9, 11,
                     11, 11], dtype=np.uint8)

def _2Dto3D(bin_file, depth_img, return_as='3D', mapping_as='pcd'):
    # parameters
    img_height, img_width = 480, 640
    img_scale = 1.0
    vox_unit = 0.02
    vox_size = (240,144,240)
    cam_K = np.array([518.8579 / img_scale, 0., img_width / (2 * img_scale),
                      0., 518.8579 / img_scale, img_height / (2 * img_scale),
                      0., 0., 1.], dtype=np.float32)
    # cam_K = np.array([518.8579, 0, 325.58,
    #                   0, 519.4696, 253.74,
    #                   0, 0, 1], dtype=np.float32)

    vox_origin, cam_pose, _ = _read_bin(bin_file)

    depth_mapping = np.ones((img_height, img_width), dtype=np.int32) * -1
    mask = np.zeros_like(depth_img, dtype=np.bool_)

    img_y = np.repeat(np.expand_dims(np.arange(depth_img.shape[0]), axis=1), depth_img.shape[1], axis=1)
    img_x = np.repeat(np.expand_dims(np.arange(depth_img.shape[1]), axis=0), depth_img.shape[0], axis=0)

    point_cam_x = (img_x - cam_K[2]) * depth_img / cam_K[0]
    point_cam_y = (img_y - cam_K[5]) * depth_img / cam_K[4]
    point_cam_z = depth_img

    point_base_x = cam_pose[0 * 4 + 0] * point_cam_x + cam_pose[0 * 4 + 1] * point_cam_y + cam_pose[0 * 4 + 2] * point_cam_z
    point_base_y = cam_pose[1 * 4 + 0] * point_cam_x + cam_pose[1 * 4 + 1] * point_cam_y + cam_pose[1 * 4 + 2] * point_cam_z
    point_base_z = cam_pose[2 * 4 + 0] * point_cam_x + cam_pose[2 * 4 + 1] * point_cam_y + cam_pose[2 * 4 + 2] * point_cam_z

    point_base_x = point_base_x + cam_pose[0 * 4 + 3]
    point_base_y = point_base_y + cam_pose[1 * 4 + 3]
    point_base_z = point_base_z + cam_pose[2 * 4 + 3]

    if mapping_as == 'pcd':
        z = (point_base_x - vox_origin[0]) / vox_unit
        x = (point_base_y - vox_origin[1]) / vox_unit
        y = (point_base_z - vox_origin[2]) / vox_unit

    elif mapping_as == 'voxel':
        z = np.floor((point_base_x - vox_origin[0]) / vox_unit)
        x = np.floor((point_base_y - vox_origin[1]) / vox_unit)
        y = np.floor((point_base_z - vox_origin[2]) / vox_unit)

        z = z.astype(np.int32)
        x = x.astype(np.int32)
        y = y.astype(np.int32)

    for i in range((480*640)):
        pix_y, pix_x = i // 640, i % 640 # row, col

        if x[pix_y,pix_x] >= 0 and x[pix_y,pix_x] < vox_size[0] \
            and y[pix_y,pix_x] >= 0 and y[pix_y,pix_x] < vox_size[1] \
            and z[pix_y,pix_x] >= 0 and z[pix_y,pix_x] < vox_size[2]:

            vox_idx = z[pix_y,pix_x] * vox_size[0] * vox_size[1] \
                    + y[pix_y,pix_x] * vox_size[0] \
                    + x[pix_y,pix_x]

            depth_mapping[pix_y,pix_x] = vox_idx
            mask[pix_y,pix_x] = 1

    # each pixel contains voxel idx ranging from 0 -> 8,294,400
    # shape = (307200,)
    if return_as == '1D':
        return np.reshape(depth_mapping, (-1,))

    # each pixel contains voxel idx ranging from 0 -> 8,294,400
    # shape = (480,640)
    elif return_as == '2D':
        return depth_mapping

    # 3-channel image for 3 axes
    # each pixel contains voxel idx ranging from (0 -> 240), (0 -> 144), (0 -> 240) for x, y, z axes respectively
    elif return_as == '3D':
        mask = np.expand_dims(mask, axis=0)
        zxy = np.stack((z,x,y), axis=0)
        zxy = zxy * mask
        return zxy, mask

def _3Dto2D(bin_path, pcd):
    # parameters
    img_height, img_width = 480, 640
    img_scale = 1.0
    vox_unit = 0.02
    vox_size = (240,144,240)
    cam_K = np.array([518.8579 / img_scale, 0., img_width / (2 * img_scale),
                      0., 518.8579 / img_scale, img_height / (2 * img_scale),
                      0., 0., 1.], dtype=np.float32)

    # pcd = np.asarray(pcd.points)

    vox_origin, cam_pose = _read_bin(bin_path)

    point_base_x = pcd[:,0] * vox_unit + vox_origin[0]
    point_base_y = pcd[:,1] * vox_unit + vox_origin[1]
    point_base_z = pcd[:,2] * vox_unit + vox_origin[2]

    point_base_x = point_base_x - cam_pose[0 * 4 + 3]
    point_base_y = point_base_y - cam_pose[1 * 4 + 3]
    point_base_z = point_base_z - cam_pose[2 * 4 + 3]

    point_cam_x = cam_pose[0 * 4 + 0] * point_base_x + cam_pose[1 * 4 + 0] * point_base_y + cam_pose[2 * 4 + 0] * point_base_z
    point_cam_y = cam_pose[0 * 4 + 1] * point_base_x + cam_pose[1 * 4 + 1] * point_base_y + cam_pose[2 * 4 + 1] * point_base_z
    point_cam_z = cam_pose[0 * 4 + 2] * point_base_x + cam_pose[1 * 4 + 2] * point_base_y + cam_pose[2 * 4 + 2] * point_base_z

    pixel_x = cam_K[0] * (point_cam_x / point_cam_z) + cam_K[2]
    pixel_y = cam_K[4] * (point_cam_y / point_cam_z) + cam_K[5]

    pixel_x = np.round_(pixel_x).astype(np.int16)
    pixel_y = np.round_(pixel_y).astype(np.int16)

    return pixel_x, pixel_y

def _gen_normal(depth_path):

    def _details_and_fov(img_height, img_width, img_scale, vox_scale):
        vox_details = np.array([0.02 * vox_scale, 0.24], np.float32)
        camera_fov = np.array([518.8579 / img_scale, 0., img_width / (2 * img_scale),
                               0., 518.8579 / img_scale, img_height / (2 * img_scale),
                               0., 0., 1.], dtype=np.float32)
        return vox_details, camera_fov

    def _diff_vec(img, axis=0):
        img_diff = np.diff(img, 1, axis)
        img_diff_l = img_diff[1:, :] if axis == 0 else img_diff[:, 1:]
        img_diff_h = img_diff[:-1, :] if axis == 0 else img_diff[:, :-1]
        img_diff = img_diff_l + img_diff_h
        pad_tuple = ((1, 1), (0, 0), (0, 0)) if axis == 0 else ((0, 0), (1, 1), (0, 0))
        padded = np.lib.pad(img_diff, pad_tuple, 'edge')
        return padded

    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    lower_depth = depth >> 3
    higher_depth = (depth % 8) << 13
    real_depth = (lower_depth | higher_depth).astype(np.float32) / 1000
    _, fov = _details_and_fov(*real_depth.shape, 1, 1)

    img_x = np.repeat(np.expand_dims(np.arange(real_depth.shape[0]), axis=1), real_depth.shape[1], axis=1)
    img_y = np.repeat(np.expand_dims(np.arange(real_depth.shape[1]), axis=0), real_depth.shape[0], axis=0)
    point_cam_x = (img_x - fov[2]) * real_depth / fov[0]
    point_cam_y = (img_y - fov[5]) * real_depth / fov[4]
    points = np.stack([point_cam_x, point_cam_y, real_depth], axis=2)

    diff_y = _diff_vec(points, axis=0)
    diff_x = _diff_vec(points, axis=1)
    normal = np.cross(diff_x, diff_y)
    normal_factor = np.expand_dims(np.linalg.norm(normal, axis=2), axis=-1)
    normal = np.where((normal_factor == 0.) | np.isnan(normal_factor), (0, 0, 0), normal / normal_factor)
    normal = (np.clip((normal + 1) / 2, 0, 1) * 65535).astype(np.uint16)

    return normal

# C++ implementation of rounding away from zero
def _round_half_up(x):

    out = x.copy()
    mask = (out >= 0)
    np.add(out, 0.5, where=mask, out=out)
    np.floor(out, where=mask, out=out)
    np.invert(mask, out=mask)
    np.subtract(out, 0.5, where=mask, out=out)
    np.ceil(out, where=mask, out=out)

    return out

# def _round_half_up(x):
#     output = cp.zeros_like(x)
#
#     mask = np.where(x >= 0)
#     output[mask] = x[mask] + 0.5
#     output[mask] = cp.floor(output[mask])
#
#     mask = np.where(x < 0)
#     output[mask] = x[mask] - 0.5
#     output[mask] = cp.ceil(output[mask])
#
#     return output
