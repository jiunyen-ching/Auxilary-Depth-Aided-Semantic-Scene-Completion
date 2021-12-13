"""
Compare tsdf from SATNET and Edgenet - see if channel-wise gives problem
now edgenet
"""
import numpy as np
import os
import argparse
from scene_templates import *
from scene_functions import *

parser = argparse.ArgumentParser()
parser.add_argument('--input', help='Full input filepath', type=str, required=True, default="/home/mmu/Desktop/NYU0003_0000.npz")
parser.add_argument('--preprocessor', help='Dataset preprocessed by which author?', type=str, choices=['satnet','edgenet'], required=True)
parser.add_argument('--type', help='Type of voxel to visualize', type=str, choices=['tsdf','occupancy','semantic','gt','weights'], required=True, default='tsdf')
parser.add_argument('--output', help='Output folder to store output .ply', type=str, required=False, default='same')
args = parser.parse_args()

voxel = np.load(args.input)
tsdf_list = []
if args.preprocessor == 'edgenet':
    # try:
        # tsdf_1 = voxel['tsdf']
        # tsdf_1 = voxel['lbl_hr']
        # tsdf_2 = voxel['edges']
    # except KeyError:
    #     tsdf_1 = voxel['tsdf_a'] # tsdf_a and tsdf_b are my dataset
    #     tsdf_2 = voxel['tsdf_b']
    # occupancy_semantic = voxel['lbl']
    occupancy_semantic = voxel['scatter']
    # weights = voxel['weights']
    # tsdf_list.append((tsdf_1, tsdf_2))
    # tsdf_list.append((tsdf_1, None))
else:
    tsdf = voxel['arr_0']
    # occupancy_semantic = voxel['arr_1']
    # weights = voxel['arr_2']
    # mapping = voxel['arr_3']
    tsdf_list.append((tsdf, None))


if args.type == 'tsdf':
    tsdf_list = np.squeeze(tsdf_list, axis=0) # (1, 2, 240, 144, 240, 1) -> (2, 240, 144, 240, 1)
    for iter in range(2):
        tsdf = tsdf_list[iter]
        if len(np.shape(tsdf)) != 0: # satnet does not have 2nd tsdf
            tsdf = np.array(tsdf, dtype=np.float64)
            if tsdf.shape != (240,144,240):
                shape = np.array(tsdf.shape)
                axis = int(np.where(shape == 1)[0])
                tsdf = np.squeeze(tsdf, axis=axis) # (240, 144, 240, 1) -> (240, 144, 240)

            ply_file, unique, voxel_count = get_scene_properties(args.input, args.output, args.type, tsdf, iter)
            with open(ply_file, "w") as f:
                f.write(ply_template % (8*voxel_count, 6*voxel_count))
                print('Writing to', ply_file)

                for idx in range(voxel_count):
                    loc = unique[0][idx], unique[1][idx], unique[2][idx]

                    # if tsdf[loc] == 1: # occupied
                    #     color = color_tsdf[2]
                    #
                    # elif tsdf[loc] > 0 and tsdf[loc] < 1: # foreground
                    #     color = color_tsdf[0]
                    #
                    # else: # background
                    #     color = color_tsdf[3]
                    if tsdf[loc] > 0.8 and tsdf[loc] < 1:
                        color = color_tsdf[0]
                    elif tsdf[loc] < 0:
                        color = color_tsdf[1]
                    elif tsdf[loc] == 1:
                        color = color_tsdf[2]
                    write_vertex(f, loc, color, args.preprocessor)
                write_face(f, voxel_count)


elif args.type == 'occupancy' or args.type == 'semantic' or args.type == 'gt':
    # print(np.unique(occupancy_semantic))

    if len(occupancy_semantic.shape) != 3: # (x,y,z,1) -> (x,y,z) OR (1,1,x,y,z) -> (x,y,z)
        shape = np.array(occupancy_semantic.shape)
        print(shape)

        squeeze_axis = np.where(shape == 1)[0] # [j,k,240,144,240]
        for ax in range(len(occupancy_semantic.shape) - 3): # keep squeezing until we're left with (240,144,240)
            axis = squeeze_axis[ax] - ax # with every squeeze, the no. of dimensions reduce, so must keep updating by minus-ing the loop iter.
            occupancy_semantic = np.squeeze(occupancy_semantic, axis=axis)
    print(occupancy_semantic.shape)

    if args.type == 'gt':
        seg_class_map = [0, 1, 2, 3, 4, 11, 5, 6, 7, 8, 8, 10, 10, 10, 11, 11, 9, 8, 11, 11, 11, 11, 11, 11, 11, 11, 11, 10,
                        10, 11, 8, 10, 11, 9, 11, 11, 11, 12] # 37 elements
        if 255 in np.unique(occupancy_semantic):
            occupancy_semantic[occupancy_semantic == 255] = 0
        occupancy_semantic = np.reshape(occupancy_semantic, (-1))
        print(occupancy_semantic.shape)
        _occupancy_semantic = np.take(seg_class_map, occupancy_semantic)
        occupancy_semantic = np.reshape(_occupancy_semantic, (240,144,240))

    ply_file, unique, voxel_count = get_scene_properties(args.input, args.output, args.type, occupancy_semantic)
    with open(ply_file, "w") as f:
        f.write(ply_template % (8*voxel_count, 6*voxel_count))
        print('Writing to', ply_file)

        for idx in range(voxel_count):
            loc = unique[0][idx], unique[1][idx], unique[2][idx]

            if occupancy_semantic[loc] != 0:
                if args.type == 'occupancy':
                    color = color_tsdf[0] # borrow color_tsdf's colors
                else:
                    color = color_suncg[occupancy_semantic[loc]] # use class id as index for color array
            write_vertex(f, loc, color, args.preprocessor)
        write_face(f, voxel_count)


elif args.type == 'weights':
    if weights.shape != (60,36,60):
        shape = np.array(weights.shape)
        axis = int(np.where(shape == 1)[0])
        weights = np.squeeze(weights, axis=axis)

    ply_file, unique, voxel_count = get_scene_properties(args.input, args.output, args.type, weights)
    with open(ply_file, "w") as f:
        f.write(ply_template % (8*voxel_count, 6*voxel_count))
        print('Writing to', ply_file)

        for idx in range(voxel_count):
            loc = unique[0][idx], unique[1][idx], unique[2][idx]

            if weights[loc] == 1:
                color = color_tsdf[1]
            write_vertex(f, loc, color, args.preprocessor)
        write_face(f, voxel_count)

elif args.type == 'mapping':
    if args._preprocessor == 'edgenet':
        print("Mapping not implemented for edgenet!") # should use 'raise' here i think..
        pass
    else:
        scene = np.zeros((240*144*240))
        scene[mapping] = 1
        scene = np.reshape(scene, (240,144,240))

        ply_file, unique, voxel_count = get_scene_properties(args.input, args.output, args.type, scene)
        with open(ply_file, "w") as f:
            f.write(ply_template % (8*voxel_count, 6*voxel_count))
            print('Writing to', ply_file)

            for idx in range(voxel_count):
                loc = unique[0][idx], unique[1][idx], unique[2][idx]

                if scene[loc] == 1:
                    color = color_tsdf[1]
                write_vertex(f, loc, color, args.preprocessor)
            write_face(f, voxel_count)


print("Done processing.")
