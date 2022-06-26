"""
Compare tsdf from SATNET and Edgenet - see if channel-wise gives problem
now edgenet
"""
import numpy as np
import os
import argparse
from scene_templates import *
from scene_functions import *
import zipfile

parser = argparse.ArgumentParser()
parser.add_argument('--input', help='Provide input filepath (either single file or folder)', type=str, required=True)
parser.add_argument('--output', '-o', help='Output folder to store output .ply', type=str, required=False, default='same')
parser.add_argument('--flip_lr', help='Set to True if resulting voxel is flipped when compared to image', type=str, choices=['False', 'True'], required=False)
args = parser.parse_args()

# Parameters
_3D_shapelist_hr = [(240,144,240),
                    (240,144,240,1),
                    (1,240,144,240)]
_3D_shapelist_lr = [(60,36,60),
                    (60,36,60,1),
                    (1,60,36,60)]
_2D_shapelist    = [(3,480,640),
                    (480,640,3),
                    (480,640),
                    (1,480,640),
                    (480,640,1)]
_1D_shapelist    = [(240*144*240),
                    (1,240*144*240),
                    (240*144*240,1)]
class_labels = [0,1,2,3,4,5,6,7,8,9,10,11,255]
vol_flags    = [-4.,-2.,-1.,-0.5, 0.5, 1.] # refer to _vol_d4.mat in eval

# Load paths
if os.path.isdir(args.input):
    filelist = [os.path.join(args.input, x) for x in os.listdir(args.input) if x.endswith('.npz')]
    print("Processing {} files".format(len(filelist)))
else:
    filelist = [args.input]
    print("Processing {}".format(args.input))

# Process file(s)
for file in filelist:
    try:
        voxel = np.load(file)
    except zipfile.BadZipFile:
        print("Cannot load {}".format(file))
        continue

    print('Found keys:', [key for key in voxel.keys()])

    vox_tsdf_count = 0
    for key in voxel.keys():
        # for debugging other volumes
        # if key in ['tsdf', 'edges']:
        #     continue
        array = voxel[key]

        # Identify either by key, shape, or any appropriate combination
        if array.shape in _3D_shapelist_hr: # tsdf, edge_tsdf, highres_labels, vox_binary
            # Standardize shape first
            array = array.reshape((240,144,240))

            # tsdf (should have 243 unique floating points)
            if key in ['tsdf','TSDF','vox_tsdf','edges','EDGES','edge_tsdf'] or len(np.unique(array)) > 240:
                print('processing {} | TSDF'.format(key))
                ply_file, unique, voxel_count = get_scene_properties(file, args.output, 'tsdf', array, iter=vox_tsdf_count)
                with open(ply_file, "w") as f:
                    f.write(ply_template % (8*voxel_count, 6*voxel_count))
                    print('Writing to', ply_file)

                    for idx in range(voxel_count):
                        loc = unique[0][idx], unique[1][idx], unique[2][idx]
                        if array[loc] > 0.8 and array[loc] < 1:
                            color = color_tsdf[0]
                        elif array[loc] < 0:
                            color = color_tsdf[1]
                        elif array[loc] == 1:
                            color = color_tsdf[2]
                        write_vertex(f, loc, color, args.flip_lr)
                    write_face(f, voxel_count)
                vox_tsdf_count += 1

            # occupancy grid should have only 2 unique values (0.,1.), (0,1)
            elif key in ['vox_binary','grid','occupancy_grid'] or len(np.unique(array)) == 2:
                print('processing {} | OCCUPANCY GRID'.format(key))
                ply_file, unique, voxel_count = get_scene_properties(file, args.output, 'occupancy', array)
                with open(ply_file, "w") as f:
                    f.write(ply_template % (8*voxel_count, 6*voxel_count))
                    print('Writing to', ply_file)

                    for idx in range(voxel_count):
                        loc = unique[0][idx], unique[1][idx], unique[2][idx]
                        color = color_tsdf[2]
                        write_vertex(f, loc, color, args.flip_lr)
                    write_face(f, voxel_count)

            # check if voxel values are found in class_labels list
            elif key in ['vox_label','hr_label','hr_labels','gt'] or set(np.unique(array)).issubset(set(class_labels)):
                # seg_class_map = [0, 1, 2, 3, 4, 11, 5, 6, 7, 8, 8, 10, 10, 10, 11, 11, 9, 8, 11, 11, 11, 11, 11, 11, 11, 11, 11, 10,
                #                 10, 11, 8, 10, 11, 9, 11, 11, 11, 12] # 37 elements
                # if 255 in np.unique(occupancy_semantic):
                #     occupancy_semantic[occupancy_semantic == 255] = 0
                # occupancy_semantic = np.reshape(occupancy_semantic, (-1))
                # print(occupancy_semantic.shape)
                # _occupancy_semantic = np.take(seg_class_map, occupancy_semantic)
                # occupancy_semantic = np.reshape(_occupancy_semantic, (240,144,240))

                # process highres_labels
                print('processing {} | HIGH RES VOXEL LABELS'.format(key))
                ply_file, unique, voxel_count = get_scene_properties(file, args.output, 'semantic', array, hr_or_lr='hr')
                with open(ply_file[:-4], "w") as f:
                    f.write(ply_template % (8*voxel_count, 6*voxel_count))
                    print('Writing to', ply_file)

                    for idx in range(voxel_count):
                        loc = unique[0][idx], unique[1][idx], unique[2][idx]
                        if array[loc] != 0:
                            color = color_suncg[array[loc]] # use class id as index for color array
                        write_vertex(f, loc, color, args.flip_lr)
                    write_face(f, voxel_count)

        elif array.shape in _3D_shapelist_lr: # lowres_labels, weights, vol
            # Standardize shape first
            array = array.reshape((60,36,60))

            if key in ['vol'] or set(np.unique(array)).issubset(set(vol_flags)):
                print('Not processing {} | VOL'.format(key))
                continue

            # weights should have only 2 unique values (0.,1.), (0,1)
            elif key in ['weight','weights'] or len(np.unique(array)) == 2:
                ply_file, unique, voxel_count = get_scene_properties(file, args.output, 'weights', array)

                #  Create colormap for different weights, different shades to indicate weights
                weight_count = np.unique(array)
                linspace = np.linspace(0, 1, len(weight_count), endpoint=True)
                linspace = np.array(linspace * 255, dtype=np.uint8)
                color_weight = np.zeros((len(linspace), 3), dtype=np.uint8)
                color_weight[:,2] = linspace

                with open(ply_file, "w") as f:
                    f.write(ply_template % (8*voxel_count, 6*voxel_count))
                    print('Writing to', ply_file)

                    for idx in range(voxel_count):
                        loc = unique[0][idx], unique[1][idx], unique[2][idx]
                        idx = np.where(array[loc] == weight_count)[0][0]
                        color = color_weight[idx] # use class id as index for color array
                        write_vertex(f, loc, color, args.flip_lr)
                    write_face(f, voxel_count)

            # but edgenet weights can have other floating points to signify different 'voxel importance' during training
            # so we check if every unique weight value is found inside the list of class labels
            elif not set(np.unique(array)).issubset(set(class_labels)):
                ply_file, unique, voxel_count = get_scene_properties(file, args.output, 'weights', array)

                # Create colormap for different weights, different shades to indicate weights
                weight_count = np.unique(array)
                linspace = np.linspace(0, 1, len(weight_count), endpoint=True)
                linspace = np.array(linspace * 255, dtype=np.uint8)
                color_weight = np.zeros((len(linspace), 3), dtype=np.uint8)
                color_weight[:,2] = linspace

                with open(ply_file, "w") as f:
                    f.write(ply_template % (8*voxel_count, 6*voxel_count))
                    print('Writing to', ply_file)

                    for idx in range(voxel_count):
                        loc = unique[0][idx], unique[1][idx], unique[2][idx]
                        idx = np.where(array[loc] == weight_count)[0][0]
                        color = color_weight[idx] # use class id as index for color array
                        write_vertex(f, loc, color, args.flip_lr)
                    write_face(f, voxel_count)

            elif key in ['vox_label','lr_label','lr_labels','lbl'] or set(np.unique(array)).issubset(set(class_labels)):
                print('Processing {} | LOW RES VOXEL LABELS'.format(key))
                ply_file, unique, voxel_count = get_scene_properties(file, args.output, 'semantic', array, hr_or_lr='lr')
                with open(ply_file, "w") as f:
                    f.write(ply_template % (8*voxel_count, 6*voxel_count))
                    print('Writing to', ply_file)

                    for idx in range(voxel_count):
                        loc = unique[0][idx], unique[1][idx], unique[2][idx]
                        color = color_suncg[array[loc]] # use class id as index for color array
                        write_vertex(f, loc, color, args.flip_lr)
                    write_face(f, voxel_count)

        elif array.shape in _2D_shapelist or array.shape in _1D_shapelist:
            if array.shape in _2D_shapelist[:2]: # 3 channel
                print('Not processing {} | RGB/HSV'.format(key))
                continue
            else:
                print('Processing {} | Mapping'.format(key))
                if array.shape in _2D_shapelist[2:]:
                    array = array.reshape(480*640)
                    array = array[array != -1]
                    scene = np.zeros((240*144*240))
                    scene[array] = 1
                    scene = np.reshape(scene, (240,144,240))

                    ply_file, unique, voxel_count = get_scene_properties(file, args.output, 'mapping', scene)
                    with open(ply_file, "w") as f:
                        f.write(ply_template % (8*voxel_count, 6*voxel_count))
                        print('Writing to', ply_file)

                        for idx in range(voxel_count):
                            loc = unique[0][idx], unique[1][idx], unique[2][idx]

                            if scene[loc] == 1:
                                color = color_tsdf[1]
                            write_vertex(f, loc, color, args.flip_lr)
                        write_face(f, voxel_count)

        else:
            print('Not able to identify keys')

print("Done processing.")
