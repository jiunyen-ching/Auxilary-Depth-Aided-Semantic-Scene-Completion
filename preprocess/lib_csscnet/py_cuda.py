import ctypes
import numpy as np
import cv2
import os

from enum import Enum
class InputType(Enum):
   DEPTH_ONLY = 1
   DEPTH_COLOR = 2
   DEPTH_EDGES = 3

def get_segmentation_class_map():
    return np.array([0, 1, 2, 3, 4, 11, 5, 6, 7, 8, 8, 10, 10, 10, 11, 11, 9, 8, 11, 11,
                                   11, 11, 11, 11, 11, 11, 11, 10, 10, 11, 8, 10, 11, 9, 11, 11, 11], dtype=np.int32)
def get_class_names():
    return ["ceil.", "floor", "wall ", "wind.", "chair", "bed  ", "sofa ", "table", "tvs  ", "furn.", "objs."]

#nvcc --ptxas-options=-v --compiler-options '-fPIC' -o lib_csscnet.so --shared lib_csscnet.cu

_lib = ctypes.CDLL(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))),'src/lib_csscnet.so'))


_lib.ProcessColor.argtypes = (ctypes.c_char_p,
                              ctypes.c_void_p,
                              ctypes.c_void_p,
                              ctypes.c_void_p,
                              ctypes.c_int,
                              ctypes.c_void_p,
                              ctypes.c_void_p,
                              ctypes.c_void_p,
                              ctypes.c_void_p,
                              ctypes.c_void_p,
                              ctypes.c_void_p
                              )

_lib.ProcessEdges.argtypes = (ctypes.c_char_p,
                              ctypes.c_void_p,
                              ctypes.c_void_p,
                              ctypes.c_void_p,
                              ctypes.c_int,
                              ctypes.c_void_p,
                              ctypes.c_void_p,
                              ctypes.c_void_p,
                              ctypes.c_void_p,
                              ctypes.c_void_p,
                              ctypes.c_void_p
                              )

_lib.setup.argtypes = (ctypes.c_int,
                       ctypes.c_int,
                       ctypes.c_void_p,
                       ctypes.c_int,
                       ctypes.c_int,
                       ctypes.c_float,
                       ctypes.c_float)

def lib_sscnet_setup(device=0, num_threads=1024, K=None, frame_shape=(640, 480), v_unit=0.02, v_margin=0.24, debug=0):

    global _lib

    frame_width = frame_shape[0]
    frame_height = frame_shape[1]

    if K is None:
        K = np.array([518.8579, 0.0, frame_width / 2.0, 0.0, 518.8579, frame_height / 2.0, 0.0, 0.0, 1.0],dtype=np.float32)

    _lib.setup(ctypes.c_int(device),
              ctypes.c_int(num_threads),
              K.ctypes.data_as(ctypes.c_void_p),
              ctypes.c_int(frame_width),
              ctypes.c_int(frame_height),
              ctypes.c_float(v_unit),
              ctypes.c_float(v_margin),
              ctypes.c_int(debug)
              )

def process_color(file_prefix, voxel_shape, down_scale = 4):
    global _lib

    vox_origin = np.ones(3,dtype=np.float32)
    cam_pose = np.ones(16,dtype=np.float32)
    num_voxels = voxel_shape[0] * voxel_shape[1] * voxel_shape[2]
    vox_size = np.array([voxel_shape[0], voxel_shape[1], voxel_shape[2]], dtype=np.int32)
    segmentation_class_map = get_segmentation_class_map()
    segmentation_label = np.zeros(num_voxels//(down_scale*down_scale*down_scale), dtype=np.int32)

    vox_weights = np.zeros(num_voxels//(down_scale*down_scale*down_scale), dtype=np.float32)
    vox_vol = np.zeros(num_voxels//(down_scale*down_scale*down_scale), dtype=np.float32)

    depth_image = cv2.imread(file_prefix+'_depth.png', cv2.IMREAD_ANYDEPTH)
    rgb_image = cv2.imread(file_prefix+'_color.jpg', cv2.IMREAD_COLOR)

    vox_tsdf = np.zeros(num_voxels, dtype=np.float32)
    vox_rgb = np.zeros(3*num_voxels, dtype=np.float32)

    _lib.ProcessColor(ctypes.c_char_p(bytes(file_prefix+'.bin','utf-8')),
                      cam_pose.ctypes.data_as(ctypes.c_void_p),
                      vox_size.ctypes.data_as(ctypes.c_void_p),
                      vox_origin.ctypes.data_as(ctypes.c_void_p),
                      ctypes.c_int(down_scale),
                      segmentation_class_map.ctypes.data_as(ctypes.c_void_p),
                      depth_image.ctypes.data_as(ctypes.c_void_p),
                      rgb_image.ctypes.data_as(ctypes.c_void_p),
                      vox_tsdf.ctypes.data_as(ctypes.c_void_p),
                      vox_rgb.ctypes.data_as(ctypes.c_void_p),
                      vox_weights.ctypes.data_as(ctypes.c_void_p),
                      vox_vol.ctypes.data_as(ctypes.c_void_p),
                      segmentation_label.ctypes.data_as(ctypes.c_void_p)
                      )

    return vox_tsdf, vox_rgb, segmentation_label, vox_weights, vox_vol

# def process_edges(file_prefix, voxel_shape, down_scale = 4):
#     global _lib
#
#     vox_origin = np.ones(3,dtype=np.float32)
#     cam_pose = np.ones(16,dtype=np.float32)
#     num_voxels = voxel_shape[0] * voxel_shape[1] * voxel_shape[2]
#     vox_size = np.array([voxel_shape[0], voxel_shape[1], voxel_shape[2]], dtype=np.int32)
#     segmentation_class_map = get_segmentation_class_map()
#     segmentation_label = np.zeros(num_voxels//(down_scale*down_scale*down_scale), dtype=np.int32)
#
#     vox_weights = np.zeros(num_voxels//(down_scale*down_scale*down_scale), dtype=np.float32)
#     vox_vol = np.zeros(num_voxels//(down_scale*down_scale*down_scale), dtype=np.float32)
#
#     depth_image = cv2.imread(file_prefix+'_depth.png', cv2.IMREAD_ANYDEPTH)
#     rgb_image = cv2.imread(file_prefix+'_color.jpg', cv2.IMREAD_COLOR)
#     rgb_image = cv2.cvtColor(rgb_image,cv2.COLOR_BGR2RGB)
#     edges_image = cv2.Canny(rgb_image,100,200)
#
#     vox_tsdf = np.zeros(num_voxels, dtype=np.float32)
#     tsdf_edges = np.zeros(num_voxels, dtype=np.float32)
#     vox_edges = np.zeros(num_voxels, dtype=np.float32)
#
#     _lib.ProcessEdges(ctypes.c_char_p(bytes(file_prefix+'.bin','utf-8')),
#                       cam_pose.ctypes.data_as(ctypes.c_void_p),
#                       vox_size.ctypes.data_as(ctypes.c_void_p),
#                       vox_origin.ctypes.data_as(ctypes.c_void_p),
#                       ctypes.c_int(down_scale),
#                       segmentation_class_map.ctypes.data_as(ctypes.c_void_p),
#                       depth_image.ctypes.data_as(ctypes.c_void_p),
#                       edges_image.ctypes.data_as(ctypes.c_void_p),
#                       vox_tsdf.ctypes.data_as(ctypes.c_void_p),
#                       vox_edges.ctypes.data_as(ctypes.c_void_p),
#                       tsdf_edges.ctypes.data_as(ctypes.c_void_p),
#                       vox_weights.ctypes.data_as(ctypes.c_void_p),
#                       vox_vol.ctypes.data_as(ctypes.c_void_p),
#                       segmentation_label.ctypes.data_as(ctypes.c_void_p)
#                       )
#
#     return vox_tsdf, vox_edges, tsdf_edges, segmentation_label, vox_weights, vox_vol

def process_edges(file_prefix, voxel_shape, down_scale = 4):
    global _lib

    vox_origin = np.ones(3,dtype=np.float32)
    cam_pose = np.ones(16,dtype=np.float32)
    num_voxels = voxel_shape[0] * voxel_shape[1] * voxel_shape[2]
    vox_size = np.array([voxel_shape[0], voxel_shape[1], voxel_shape[2]], dtype=np.int32)
    segmentation_class_map = get_segmentation_class_map()
    segmentation_label = np.zeros(num_voxels//(down_scale*down_scale*down_scale), dtype=np.int32)

    vox_weights = np.zeros(num_voxels//(down_scale*down_scale*down_scale), dtype=np.float32)
    vox_vol = np.zeros(num_voxels//(down_scale*down_scale*down_scale), dtype=np.float32)

    depth_image = cv2.imread(file_prefix+'_depth.png', cv2.IMREAD_ANYDEPTH)
    rgb_image = cv2.imread(file_prefix+'_color.jpg', cv2.IMREAD_COLOR)
    rgb_image = cv2.cvtColor(rgb_image,cv2.COLOR_BGR2RGB)
    edges_image = cv2.Canny(rgb_image,100,200)

    vox_tsdf = np.zeros(num_voxels, dtype=np.float32)
    tsdf_edges = np.zeros(num_voxels, dtype=np.float32)
    vox_edges = np.zeros(num_voxels, dtype=np.float32)
    vox_grid_new = np.load('./path/to/surface_grid.npz') # need to flatten here

    _lib.ProcessEdges(ctypes.c_char_p(bytes(file_prefix+'.bin','utf-8')),
                      cam_pose.ctypes.data_as(ctypes.c_void_p),
                      vox_size.ctypes.data_as(ctypes.c_void_p),
                      vox_origin.ctypes.data_as(ctypes.c_void_p),
                      ctypes.c_int(down_scale),
                      segmentation_class_map.ctypes.data_as(ctypes.c_void_p),
                      depth_image.ctypes.data_as(ctypes.c_void_p),
                      edges_image.ctypes.data_as(ctypes.c_void_p),
                      vox_tsdf.ctypes.data_as(ctypes.c_void_p),
                      vox_edges.ctypes.data_as(ctypes.c_void_p),
                      tsdf_edges.ctypes.data_as(ctypes.c_void_p),
                      vox_weights.ctypes.data_as(ctypes.c_void_p),
                      vox_vol.ctypes.data_as(ctypes.c_void_p),
                      segmentation_label.ctypes.data_as(ctypes.c_void_p),
                      vox_grid_new.ctypes.data_as(ctypes.c_void_p)
                      )

    return vox_tsdf, vox_edges, tsdf_edges, segmentation_label, vox_weights, vox_vol

def process(file_prefix, voxel_shape, down_scale = 4, input_type=InputType.DEPTH_COLOR):
    if input_type == InputType.DEPTH_COLOR:
        return process_color(file_prefix, voxel_shape, down_scale=down_scale)
    elif input_type == InputType.DEPTH_EDGES:
        return process_edges(file_prefix, voxel_shape, down_scale=down_scale)
    elif input_type == InputType.DEPTH_ONLY:
        print("input type DEPTH ONLY not implemented yet")
        exit(-1)
