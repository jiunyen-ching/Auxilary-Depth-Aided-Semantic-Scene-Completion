import numpy as np
import cv2
from struct import *
import h5py

def _init_params(target_tsdf=240, target_label=60):

    params = {
        "img_height"    : 480,
        "img_width"     : 640,
        "cam_K_virt"    : np.array([518.8579, 0., 320.,
                                    0., 518.8579, 240.,
                                    0., 0., 1.], dtype=np.float32),
        "cam_K_real"    : np.array([518.8579, 0, 325.58,
                                    0., 519.4696, 253.74,
                                    0., 0., 1.], dtype=np.float32),
    }

    # tsdf parameters
    params["tsdf_vox_scale"] = 240 // target_tsdf
    params["tsdf_vox_unit"] = 0.02 * params["tsdf_vox_scale"]
    params["tsdf_search_region"] = 12 // params["tsdf_vox_scale"]
    params["tsdf_vox_size"] = np.array([240,144,240]) // params["tsdf_vox_scale"]
    params["tsdf_vox_num"] = params["tsdf_vox_size"][0] * params["tsdf_vox_size"][1] * params["tsdf_vox_size"][2]

    # vox_label parameters
    params["down_scale"] = 240 // target_label
    params["in_vox_size"] = np.array([240,144,240])
    params["in_vox_num"] = params["in_vox_size"][0] * params["in_vox_size"][1] * params["in_vox_size"][2]
    params["out_vox_size"] = params["in_vox_size"] // params["down_scale"]
    params["out_vox_num"] = params["out_vox_size"][0] * params["out_vox_size"][1] * params["out_vox_size"][2]

    return params

def _get_class_map():
    return np.array([0, 1, 2, 3, 4, 11, 5, 6, 7, 8, 8, 10, 10, 10, 11, 11, 9, 8, 11, 11, 11, 11, 11, 11, 11, 11, 11, 10,
                     10, 11, 8, 10, 11, 9, 11, 11, 11, 12])
    
def _read_bin(bin_file, return_voxels=True):
    with open(bin_file,'rb') as f:
        float_size = 4
        cor = f.read(float_size*3)
        cors = unpack('fff',cor) # origin in world coordinates
        cam = f.read(float_size*16)
        cams = unpack('ffffffffffffffff', cam) # camera pose

        if return_voxels:
            uint_size = 4
            vox = f.read()
            numC = int(len(vox)/uint_size)
            checkVoxValIter = unpack('I'*numC, vox)
            checkVoxVal = checkVoxValIter[0::2]
            checkVoxIter = checkVoxValIter[1::2]
            voxels = [i for (val, repeat) in zip(checkVoxVal,checkVoxIter) for i in np.tile(val, repeat)]
            voxels = np.array(voxels, dtype=np.int16)
            return cors, cams, voxels
            
        return cors, cams, None

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

def _2Dto3D(depth_img, vox_origin, cam_pose, params, img_x=None, img_y=None, return_as=None, mapping_as='pcd', cam_type='virtual'):
    
    try:
        if not img_x.any() and not img_y.any():
            # print("Using default image coordinates")
            img_y = np.repeat(np.expand_dims(np.arange(depth_img.shape[0]), axis=1), depth_img.shape[1], axis=1)
            img_x = np.repeat(np.expand_dims(np.arange(depth_img.shape[1]), axis=0), depth_img.shape[0], axis=0)
    except ValueError:
        if not img_x and not img_y:
            # print("Using default image coordinates")
            img_y = np.repeat(np.expand_dims(np.arange(depth_img.shape[0]), axis=1), depth_img.shape[1], axis=1)
            img_x = np.repeat(np.expand_dims(np.arange(depth_img.shape[1]), axis=0), depth_img.shape[0], axis=0)
    except AttributeError:
        if not img_x and not img_y:
            # print("Using default image coordinates")
            img_y = np.repeat(np.expand_dims(np.arange(depth_img.shape[0]), axis=1), depth_img.shape[1], axis=1)
            img_x = np.repeat(np.expand_dims(np.arange(depth_img.shape[1]), axis=0), depth_img.shape[0], axis=0)

    if cam_type == 'virtual':
        cam_K = params["cam_K_virt"]
    else:
        cam_K = params["cam_K_real"]

    point_cam_x = (img_x - cam_K[2]) * depth_img / cam_K[0]
    point_cam_y = (img_y - cam_K[5]) * depth_img / cam_K[4]
    point_cam_z = depth_img

    point_base_x = cam_pose[0 * 4 + 0] * point_cam_x + cam_pose[0 * 4 + 1] * point_cam_y + cam_pose[0 * 4 + 2] * point_cam_z
    point_base_y = cam_pose[1 * 4 + 0] * point_cam_x + cam_pose[1 * 4 + 1] * point_cam_y + cam_pose[1 * 4 + 2] * point_cam_z
    point_base_z = cam_pose[2 * 4 + 0] * point_cam_x + cam_pose[2 * 4 + 1] * point_cam_y + cam_pose[2 * 4 + 2] * point_cam_z

    point_base_x = point_base_x + cam_pose[0 * 4 + 3]
    point_base_y = point_base_y + cam_pose[1 * 4 + 3]
    point_base_z = point_base_z + cam_pose[2 * 4 + 3]

    z = (point_base_x - vox_origin[0]) / params["tsdf_vox_unit"]
    x = (point_base_y - vox_origin[1]) / params["tsdf_vox_unit"]
    y = (point_base_z - vox_origin[2]) / params["tsdf_vox_unit"]

    if mapping_as == 'pcd':
        return z, y, x

    elif mapping_as == 'voxel':
        z = np.floor(z).astype(np.int32)
        y = np.floor(y).astype(np.int32)
        x = np.floor(x).astype(np.int32)
        # z = _round_half_up(z).astype(np.int32)
        # y = _round_half_up(y).astype(np.int32)
        # x = _round_half_up(x).astype(np.int32)

        assert return_as is not None, 'Please specify "return_as" format since mapping type is voxel'

        depth_mapping = np.ones((depth_img.shape[0], depth_img.shape[1]), dtype=np.int64) * -1
        mask = np.zeros_like(depth_img, dtype=np.bool_)

        for i in range((480*640)):
            pix_y, pix_x = i // 640, i % 640

            if x[pix_y,pix_x] >= 0 and x[pix_y,pix_x] < params["tsdf_vox_size"][0] \
                and y[pix_y,pix_x] >= 0 and y[pix_y,pix_x] < params["tsdf_vox_size"][1] \
                and z[pix_y,pix_x] >= 0 and z[pix_y,pix_x] < params["tsdf_vox_size"][2]:

                vox_idx = z[pix_y,pix_x] * params["tsdf_vox_size"][0] * params["tsdf_vox_size"][1] \
                        + y[pix_y,pix_x] * params["tsdf_vox_size"][0] \
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
    else:
        raise NotImplementedError

def _3Dto2D(points_3d, vox_origin, cam_pose, params, cam_type='virtual', return_at_point_cam=False, return_pixel_as_int=True):
    # points_3d can be either point cloud or voxels
    # np.where returns coordinates as "channel", "row", "col"
    point_base_x = pcd[:,0] * params["tsdf_vox_unit"] + vox_origin[0] # z - channel
    point_base_y = pcd[:,1] * params["tsdf_vox_unit"] + vox_origin[1] # x - col
    point_base_z = pcd[:,2] * params["tsdf_vox_unit"] + vox_origin[2] # y - row

    point_base_x = point_base_x - cam_pose[0 * 4 + 3]
    point_base_y = point_base_y - cam_pose[1 * 4 + 3]
    point_base_z = point_base_z - cam_pose[2 * 4 + 3]

    point_cam_x = cam_pose[0 * 4 + 0] * point_base_x + cam_pose[1 * 4 + 0] * point_base_y + cam_pose[2 * 4 + 0] * point_base_z
    point_cam_y = cam_pose[0 * 4 + 1] * point_base_x + cam_pose[1 * 4 + 1] * point_base_y + cam_pose[2 * 4 + 1] * point_base_z
    point_cam_z = cam_pose[0 * 4 + 2] * point_base_x + cam_pose[1 * 4 + 2] * point_base_y + cam_pose[2 * 4 + 2] * point_base_z

    if return_at_point_cam:
        return point_cam_x, point_cam_y, point_cam_z
    
    if cam_type == 'virtual':
        cam_K = params["cam_K_virt"]
    else:
        cam_K = params["cam_K_real"]
    pixel_x = cam_K[0] * (point_cam_x / point_cam_z) + cam_K[2]
    pixel_y = cam_K[4] * (point_cam_y / point_cam_z) + cam_K[5]

    if return_pixel_as_int:
        # pixel_x = np.round_(pixel_x).astype(np.int16)
        # pixel_y = np.round_(pixel_y).astype(np.int16)
        pixel_x = _round_half_up(pixel_x).astype(np.int16)
        pixel_y = _round_half_up(pixel_y).astype(np.int16)

    return pixel_x, pixel_y, point_cam_z

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
    
def _gen_hollow_array(array, connectivity, size=3):
    # Standardize shape
    # array = array.reshape((240,144,240))

    if connectivity == 6:
        kernel = np.array([[[0,0,0],
                            [0,1,0],
                            [0,0,0]],
                           [[0,1,0],
                            [1,1,1],
                            [0,1,0]],
                           [[0,0,0],
                            [0,1,0],
                            [0,0,0]]])
    elif connectivity == 18:
        kernel = np.array([[[0,1,0],
                            [1,1,1],
                            [0,1,0]],
                           [[1,1,1],
                            [1,1,1],
                            [1,1,1]],
                           [[0,1,0],
                            [1,1,1],
                            [0,1,0]]])
    elif connectivity == 26:
        assert size % 2 != 0, 'Enter an odd number' # Ensure odd-number dimensions
        kernel = np.ones((size,size,size), dtype=np.uint8)
    else:
        raise NotImplementedError

    pad_amount = kernel.shape[0] // 2 * 2
    offset = pad_amount // 2
    padded = np.zeros((array.shape[0] + pad_amount,
                       array.shape[1] + pad_amount,
                       array.shape[2] + pad_amount), dtype=np.bool_)
    padded[offset:-offset, offset:-offset, offset:-offset] = array

    # Mask to indicate voxel for removal
    mask = np.zeros_like(padded)

    # Consider only valid points
    _z, _y, _x = np.where(padded)
    valid_points = np.sum(padded)

    for i in range(int(valid_points)):
        z, y, x = _z[i], _y[i], _x[i]

        # target = padded[z-1:z+2, y-1:y+2, x-1:x+2]
        target = padded[z-offset:z+offset+1, y-offset:y+offset+1, x-offset:x+offset+1]
        target = kernel * target
        if target.sum() == kernel.sum():
            mask[z,y,x] = 1

    mask = mask[offset:-offset, offset:-offset, offset:-offset]
    array = array * (1 - mask)

    return array
