import numpy as np
import cv2
from struct import *

def get_bin_info(bin_file):
    with open(bin_file,'rb') as f:
        float_size = 4
        cor = f.read(float_size*3)
        cors = unpack('fff',cor) # origin in world coordinates
        cam = f.read(float_size*16)
        cams = unpack('ffffffffffffffff', cam) # camera pose
        f.close()
    return cors, cams

def read_mat(mat_file, return_as_flat=True, return_as_float=False):
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

def read_bitshift(depth_path, return_as_flat=False, return_as_float=False):
    depth = cv2.imread(depth_path, -1)
    lower_depth = depth >> 3
    higher_depth = (depth % 8) << 13
    real_depth = (lower_depth | higher_depth)

    if return_as_flat:
        real_depth = np.reshape(real_depth, (-1,))
    if return_as_float:
        real_depth = real_depth.astype(np.float32)/1000
    return real_depth

def calculate_mapping_vectorize(bin_file, depth_img_flat, return_as_flat=True):
    # parameters
    img_height, img_width = (480, 640)
    img_scale = 1.0
    vox_unit = 0.02
    vox_size = (240,144,240)
    cam_K = np.array([518.8579 / img_scale, 0., img_width / (2 * img_scale),
                      0., 518.8579 / img_scale, img_height / (2 * img_scale),
                      0., 0., 1.], dtype=np.float32)
    # cam_K = np.array([518.8579, 0, 325.58,
    #                   0, 519.4696, 253.74,
    #                   0, 0, 1], dtype=np.float32)
    
    vox_origin, cam_pose = get_bin_info(bin_file)

    real_depth = np.reshape(depth_img_flat, (img_height, img_width))

    depth_mapping = np.ones((img_height, img_width), dtype=np.int32) * -1
    mask = np.zeros_like(depth_img, dtype=np.bool_)

    img_y = np.repeat(np.expand_dims(np.arange(real_depth.shape[0]), axis=1), real_depth.shape[1], axis=1)
    img_x = np.repeat(np.expand_dims(np.arange(real_depth.shape[1]), axis=0), real_depth.shape[0], axis=0)

    point_cam_x = (img_x - cam_K[2]) * real_depth / cam_K[0]
    point_cam_y = (img_y - cam_K[5]) * real_depth / cam_K[4]
    point_cam_z = real_depth

    point_base_x = cam_pose[0 * 4 + 0] * point_cam_x + cam_pose[0 * 4 + 1] * point_cam_y + cam_pose[0 * 4 + 2] * point_cam_z;
    point_base_y = cam_pose[1 * 4 + 0] * point_cam_x + cam_pose[1 * 4 + 1] * point_cam_y + cam_pose[1 * 4 + 2] * point_cam_z;
    point_base_z = cam_pose[2 * 4 + 0] * point_cam_x + cam_pose[2 * 4 + 1] * point_cam_y + cam_pose[2 * 4 + 2] * point_cam_z;

    point_base_x = point_base_x + cam_pose[0 * 4 + 3];
    point_base_y = point_base_y + cam_pose[1 * 4 + 3];
    point_base_z = point_base_z + cam_pose[2 * 4 + 3];

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
        pix_x, pix_y = i // 640, i % 640

        if x[pix_x,pix_y] >= 0 and x[pix_x,pix_y] < vox_size[0] \
            and y[pix_x,pix_y] >= 0 and y[pix_x,pix_y] < vox_size[1] \
            and z[pix_x,pix_y] >= 0 and z[pix_x,pix_y] < vox_size[2]:

            vox_idx = z[pix_x,pix_y] * vox_size[0] * vox_size[1] \
                    + y[pix_x,pix_y] * vox_size[0] \
                    + x[pix_x,pix_y]
            
            depth_mapping[pix_x,pix_y] = vox_idx
            mask[pix_x,pix_y] = 1
            
    if return_as == '1D':
        return np.reshape(depth_mapping, (-1,))
    
    elif return_as == '2D':
        return depth_mapping

    elif return_as == '3D':
        mask = np.expand_dims(mask, axis=0)
        zxy = np.stack((z,x,y), axis=0)
        zxy = zxy * mask
        return zxy, mask
