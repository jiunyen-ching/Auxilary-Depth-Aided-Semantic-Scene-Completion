import cv2
import os
import numpy as np

# val ground truth
label_path = "C:/Users/Jy/Downloads/SATNet/Semantic_Scene_Completion/SUNCG_RGBD/labels/"
labels = os.listdir(label_path)

label_list = []
for label in labels:
    label = os.path.join(label_path, label)
    label = cv2.imread(label, -1).reshape((60,36,60))
    label_list.append(label)
        
label_list = np.array(label_list)

# Visualize the gt
ply_cube_template = """ply
format ascii 1.0
element vertex %d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
element face %d
property list uchar int vertex_index
end_header
"""

item = 0
for gt in label_list:
    invalid = gt == 255
#     print(np.unique(valid, return_counts=True))
    
    tmp_gt = gt.copy()
    tmp_gt[invalid] = 0
    valid = tmp_gt != 0
#     print(np.unique(valid, return_counts=True))
    
    _, count = np.unique(tmp_gt != 0, return_counts=True)
    count = count[1] # For True only
#     print(count)
    
    size = np.shape(gt)

    ply_file = "C:/Users/Jy/Desktop/satnet gt/%03d" % item + '.ply'
    print(ply_file)


    with open(ply_file, "w") as f:
        f.write(ply_cube_template % (8*count, 6*count))
        print('Writing to', ply_file)

        shift = 0.5
        for z in range(size[2]):
            for y in range(size[1]):
                for x in range(size[0]):
                    if gt[z][y][x] != 0 and gt[z][y][x] != 255:
                        color = colormap[gt[z][y][x]]

                        color_z, color_y, color_x = int(color[0]), int(color[1]), int(color[2])

                        _x, _y, _z = z, y, x # check here; set like this for same angle/view as rgb image, not sure about other gt's
                        # Follow VVNet's ordering
                        f.write('%f %f %f %d %d %d' % (_x - shift, _y - shift, _z - shift, color_z, color_y, color_x))
                        f.write('\n%f %f %f %d %d %d' % (_x - shift, _y + shift, _z - shift, color_z, color_y, color_x))
                        f.write('\n%f %f %f %d %d %d' % (_x + shift, _y + shift, _z - shift, color_z, color_y, color_x))
                        f.write('\n%f %f %f %d %d %d' % (_x + shift, _y - shift, _z - shift, color_z, color_y, color_x))
                        f.write('\n%f %f %f %d %d %d' % (_x - shift, _y - shift, _z + shift, color_z, color_y, color_x))
                        f.write('\n%f %f %f %d %d %d' % (_x - shift, _y + shift, _z + shift, color_z, color_y, color_x))
                        f.write('\n%f %f %f %d %d %d' % (_x + shift, _y + shift, _z + shift, color_z, color_y, color_x))
                        f.write('\n%f %f %f %d %d %d' % (_x + shift, _y - shift, _z + shift, color_z, color_y, color_x))
                        f.write('\n')

        counter = 0
        for i in range(count):
            f.write("4 %d %d %d %d\n" % (counter, counter + 1, counter + 2, counter + 3))
            f.write("4 %d %d %d %d\n" % (counter, counter + 4, counter + 5, counter + 1))
            f.write("4 %d %d %d %d\n" % (counter, counter + 3, counter + 7, counter + 4))
            f.write("4 %d %d %d %d\n" % (counter + 6, counter + 5, counter + 4, counter + 7))
            f.write("4 %d %d %d %d\n" % (counter + 6, counter + 7, counter + 3, counter + 2))
            f.write("4 %d %d %d %d\n" % (counter + 6, counter + 2, counter + 1, counter + 5))
            counter += 8

    print("Done processing.")

    item += 1
