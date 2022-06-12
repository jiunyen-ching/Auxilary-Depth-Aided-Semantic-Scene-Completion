# ADASSC-Net

## Download
### 3D Ground truth, depth images
- SUNCG (Test), NYU (Train/Test), NYUCAD (Train/Test) [Download](http://sscnet.cs.princeton.edu/sscnet_release/data/depthbin_eval.zip)
- ~SUNCG (Train) [Download](http://sscnet.cs.princeton.edu/sscnet_release/data/SUNCGtrain.zip)~
### 2D Labels, RGB
- NYU [Download](http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat)
- NYUCAD (No RGB, Labels will be provided soon)

## Data details
- Data format
  1. Depth map : 16-bit png with bitshifting (for visualization). Refer to ```_read_bitshift()``` in ```helper_functions.py``` for more details.
  2. 3D volume : 
     1. 3 floating points store the origin of the 3D volume in world coordinates. 
     2. 16 floating points store the camera pose in world coordinates. 
     3. 3D volume encoded by run-length encoding. Refer to ```_read_bin()``` in ```helper_functions.py``` for more details.

- Data availability

| Dataset   | 3D GT   | Depth   | RGB               | Amodal depth | 2D Instance Label | 2D Semantic Label |
| :-------- | :-----: | :-----: | :-----:           | :----------: | :---------------: | :---------------: |
| NYU       | &#9745; | &#9745; | &#9745;           | &#9744;      | &#9745;           | &#9745;           |
| NYUCAD    | &#9745; | &#9745; | Use NYU color img | &#9745;      | &#9745;           | &#9745;           |
| ~SUNCG~   | &#9745; | &#9745; | &#9744;           | &#9744;      | &#9744;           | &#9744;           |
| SUNCG-RGB | &#9745; | &#9745; | &#9745;           | &#9744;      | &#9744;           | &#9745;           |

## Data organisation
```
- ADASSC
  - data
    - temp
      - depthbin_eval.zip
      - nyu_depth_v2_labeled.mat
    - depthbin_eval
      - depthbin
        - NYUtest
        - NYUtrain
        - NYUCADtest
        - NYUCADtrain
        - SUNCGtest_49700_49884
        - ~SUNCGtrain~
      - eval
        - NYUtest
        - NYUCADtest
        - SUNCGtest_49700_49884
  - demo
    - sample_data
    - demo.ipynb
  - scripts
```

## Data preparation
### Under ```./scripts```...
1. Extract RGB for NYU
   - Run ```python extract_from_mat.py -t color```.
   - Specify ```-t label``` for instance and semantic labels.
2. Depth-to-normal
   - Run ```python depth2normal.py -d NYU```. Refer to ```_gen_normal()``` in ```helper_functions.py``` for more details.
3. Depth-to-HHA
   - Run ```python depth2hha.py -d NYU```.
4. Depth-to-TSDF (CPU ver.)
   - Run ```python depth2tsdf.py -d NYU -f ../data/depthbin_eval/depthbin/NYUtrain/NYU0003_0000.bin```.

## 3D Data visualization
- Requires Meshlab [Download](https://www.meshlab.net/#download) / Blender [Download](https://www.blender.org/download/)
- Visualize TSDF
  - Under ```./scripts/vis_utils```, run ```python scene_viewer.py -i path/to/file.npz -p edgenet```
  - e.g. Visualizing sample data: ```python scene_viewer.py -i ../../demo/sample_data/NYU0003_0000.npz -p edgenet```. By default, the .ply file will be saved in the same directory as the .npz file

## TODO
- &#9744; Add SUNCG-RGB download link
- &#9744; Add NYUCAD labels download link
- &#9744; Add SUNCG-RGB into data organisation tree
