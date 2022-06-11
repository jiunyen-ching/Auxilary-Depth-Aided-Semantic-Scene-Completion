"""
Contains template for writing .ply file as well as color templates used in NYUv2 and SUNCG
"""

import numpy as np

ply_template = """ply
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

color_tsdf = np.array([[0, 0, 255],
                       [255, 0, 0],
                       [0,255,0],
                       [124,124,124]], dtype=np.uint8)

color_nyu = np.array([[0,0,0.6667],
                      [0,0,1.0000],
                      [0,0.3333,1.0000],
                      [0,0.6667,1.0000],
                      [0,1.0000,1.0000],
                      [0.3333,1.0000,0.6667],
                      [0.6667,1.0000,0.3333],
                      [1.0000,1.0000,0],
                      [1.0000,0.6667,0],
                      [1.0000,0.3333,0],
                      [1.0000,0,0],
                      [0.6667,0,0]],dtype=float)
color_nyu = np.array(color_nyu*255, dtype=np.uint8)

color_suncg = np.array([[ 22,191,206],
                        [214, 38, 40],
                        [ 43,160, 43],
                        [158,216,229],
                        [114,158,206],
                        [204,204, 91],
                        [255,186,119],
                        [147,102,188],
                        [ 30,119,181],
                        [188,188, 33],
                        [255,127, 12],
                        [196,175,214],
                        [153,153,153]])
