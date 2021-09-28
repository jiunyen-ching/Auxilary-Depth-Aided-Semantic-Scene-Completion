depth = '/content/NYU0003_0000.png'
mapping = '/content/NYU0003_0000_mapping.npz'

import numpy as np
import cv2
import matplotlib.pyplot as plt

def read_bitshift(depth_path, return_flat=False, return_float=True):
  depth = cv2.imread(depth_path, -1)
  lower_depth = depth >> 3
  higher_depth = (depth % 8) << 13
  real_depth = (lower_depth | higher_depth)

  if return_float:
      real_depth = real_depth.astype(np.float32)/1000

  return real_depth

bs, ch, w, h, d = 1, 1, 240, 144, 240

# prepare depth
depth = read_bitshift(depth)
depth = np.expand_dims(depth, axis=0)
depth = np.expand_dims(depth, axis=0)
depth = np.reshape(depth, (1,1,-1)) # assume that this is a reshaped tensor with bs=1, ch=1
depth[depth != 0] = 1
depth = tf.cast(depth, tf.float32)

# prepare indices
mapping = np.load(mapping)['arr_0']
mapping = np.reshape(mapping, -1)
mapping = np.expand_dims(mapping, axis=1)
mapping = np.tile(mapping, (ch,1,1))

# print(mapping)
# print('mapping shape', mapping.shape)

rows = np.arange(ch)
rows = np.expand_dims(rows, axis=1)
# rows = np.repeat(rows, repeats=mapping.shape[1], axis=1)
# print(rows)

mapping = np.insert(mapping, 0, rows, axis=2)

mapping = np.tile(mapping, (ch,1,1,1))
# print(mapping)
# print(mapping.shape)

batch = np.arange(bs)
batch = np.expand_dims(batch, axis=-1)
batch = np.expand_dims(batch, axis=-1)

mapping = np.insert(mapping, 0, batch, axis=3)
mapping = tf.cast(mapping, tf.int32)
print(mapping.shape)

tensor = tf.zeros((bs,ch,w*h*d), dtype=tf.float32)
updated = tf.tensor_scatter_nd_max(tensor, mapping, depth)
print(updated.shape)

updated = np.reshape(updated, (1,1,240,144,240))
updated = np.transpose(updated, (0,2,3,4,1))
print(updated.dtype)

filename = '/content/scattered_ch_last.npz'
np.savez_compressed(filename,
                    scatter = updated)
