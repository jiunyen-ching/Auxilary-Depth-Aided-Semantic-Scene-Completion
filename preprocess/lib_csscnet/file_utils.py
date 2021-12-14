from lib_csscnet.py_cuda import *
import numpy as np
from sklearn.utils import shuffle
from fnmatch import fnmatch
import os
from keras.utils import to_categorical
import threading


def get_file_prefixes_from_path(data_path, criteria="*.bin"):
    prefixes = []
    for path, subdirs, files in os.walk(data_path):
        for name in files:
            if fnmatch(name, criteria):
                prefixes.append(os.path.join(path, name)[:-4])
    prefixes.sort()

    return prefixes


class threadsafe_generator(object):
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, gen):
        self.gen = gen
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return self.gen.__next__()


def threadsafe(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return threadsafe_generator(f(*a, **kw))
    return g


@threadsafe
def ftsdf_generator(file_prefixes, batch_size=4, shuff=False, shape=(240, 144, 240), down_scale = 4):  # write the definition of your data generator

    while True:
        x_batch = np.zeros((batch_size, shape[0], shape[1], shape[2], 1))
        rgb_batch = np.zeros((batch_size, shape[0], shape[1], shape[2], 3))
        y_batch = np.zeros((batch_size, shape[0] // down_scale, shape[1] // down_scale, shape[2] // down_scale, 12))
        w_batch = np.zeros((batch_size, shape[0] // down_scale, shape[1] // down_scale, shape[2] // down_scale))
        batch_count = 0
        if shuff:
            file_prefixes_s = shuffle(file_prefixes)
        else:
            file_prefixes_s = file_prefixes

        for count, file_prefix in enumerate(file_prefixes_s):


            vox_tsdf, vox_rgb, segmentation_label, vox_weights = process(file_prefix, voxel_shape=(240, 144, 240), down_scale=4)

            x_batch[batch_count] = vox_tsdf.reshape((shape[0], shape[1], shape[2],1))
            rgb_batch[batch_count] = vox_rgb.reshape((shape[0], shape[1], shape[2],3))
            y_batch[batch_count] = to_categorical(segmentation_label.reshape((60, 36, 60,1)), num_classes=12)
            w_batch[batch_count] = vox_weights.reshape((60, 36, 60))
            batch_count += 1
            if batch_count == batch_size:
                yield [x_batch, rgb_batch], y_batch, w_batch
                x_batch = np.zeros((batch_size, shape[0], shape[1], shape[2], 1)) #channels last
                rgb_batch = np.zeros((batch_size, shape[0], shape[1], shape[2], 3)) #channels last
                y_batch = np.zeros((batch_size, shape[0] // down_scale, shape[1] // down_scale, shape[2] // down_scale, 12))
                w_batch = np.zeros((batch_size, shape[0]//down_scale, shape[1]//down_scale, shape[2]//down_scale))
                batch_count = 0

        if(batch_count > 0):
            yield [x_batch[:batch_count], rgb_batch[:batch_count]], y_batch[:batch_count], w_batch[:batch_count]


@threadsafe
def preproc_generator(file_prefixes, batch_size=4, shuff=False, aug=False, vol=False, shape=(240, 144, 240), down_scale = 4, type="rgb"):  # write the definition of your data generator

    down_shape = (shape[0] // down_scale,  shape[1] // down_scale, shape[2] // down_scale)

    while True:
        x_batch = np.zeros((batch_size, shape[0], shape[1], shape[2], 1))
        if type == "rgb":
            rgb_batch = np.zeros((batch_size, shape[0], shape[1], shape[2], 3))
        elif type == "edges":
            edges_batch = np.zeros((batch_size, shape[0], shape[1], shape[2], 1))
        y_batch = np.zeros((batch_size, down_shape[0], down_shape[1], down_shape[2], 12))
        if vol:
            vol_batch = np.zeros((batch_size, down_shape[0], down_shape[1], down_shape[2]))
        batch_count = 0
        if shuff:
            file_prefixes_s = shuffle(file_prefixes)
        else:
            file_prefixes_s = file_prefixes

        for count, file_prefix in enumerate(file_prefixes_s):

            npz_file = file_prefix + '.npz'
            loaded = np.load(npz_file)

            #print(file_prefix)

            vox_tsdf  = loaded['tsdf']
            if type == "rgb":
                vox_rgb = loaded['rgb']
            elif type == "edges":
                vox_edges = loaded['edges']
            vox_label  = loaded['lbl']
            vox_weights = loaded['weights']
            if vol:
                vox_vol = loaded['vol']

            x_batch[batch_count] = vox_tsdf
            if vol:
                vol_batch[batch_count] = vox_vol

            if aug:

                aug_v = np.random.normal(loc=1, scale=0.05, size=3)

            else:
                aug_v=np.array([1.,1.,1.])

            if type == "rgb":
                rgb_batch[batch_count] = np.clip(vox_rgb * aug_v, 0., 1.)
            elif type == "edges":
                edges_batch[batch_count] = vox_edges

            labels = to_categorical(vox_label, num_classes=12)
            weights =  np.repeat(vox_weights,12,axis=-1).reshape((down_shape[0], down_shape[1], down_shape[2], 12))
            y_batch[batch_count] = labels * (weights+1)
            batch_count += 1
            if batch_count == batch_size:
                if type == "rgb":
                    if vol:
                        yield [x_batch, rgb_batch], y_batch, vol_batch
                    else:
                        yield [x_batch, rgb_batch], y_batch
                elif type == "edges":
                    if vol:
                        yield [x_batch, edges_batch], y_batch, vol_batch
                    else:
                        yield [x_batch, edges_batch], y_batch
                elif type == "depth":
                    if vol:
                        yield x_batch, y_batch, vol_batch
                    else:
                        yield x_batch, y_batch

                x_batch = np.zeros((batch_size, shape[0], shape[1], shape[2], 1)) #channels last
                if type == "rgb":
                    rgb_batch = np.zeros((batch_size, shape[0], shape[1], shape[2], 3))
                elif type == "edges":
                    edges_batch = np.zeros((batch_size, shape[0], shape[1], shape[2], 1))
                y_batch = np.zeros((batch_size, down_shape[0], down_shape[1], down_shape[2], 12))
                if vol:
                    vol_batch = np.zeros((batch_size, down_shape[0], down_shape[1], down_shape[2]))
                batch_count = 0

        if batch_count > 0:
            if type == "rgb":
                if vol:
                    yield [x_batch[:batch_count], rgb_batch[:batch_count]], y_batch[:batch_count],vol_batch[:batch_count]
                else:
                    yield [x_batch[:batch_count], rgb_batch[:batch_count]], y_batch[:batch_count]
            elif type == "edges":
                if vol:
                    yield [x_batch[:batch_count], edges_batch[:batch_count]], y_batch[:batch_count],vol_batch[:batch_count]
                else:
                    yield [x_batch[:batch_count], edges_batch[:batch_count]], y_batch[:batch_count]
            elif type == "depth":
                if vol:
                    yield x_batch[:batch_count], y_batch[:batch_count],vol_batch[:batch_count]
                else:
                    yield x_batch[:batch_count], y_batch[:batch_count]


@threadsafe
def evaluate_generator(file_prefixes, batch_size=4, shuff=False, shape=(240, 144, 240), down_scale = 4):  # write the definition of your data generator

    while True:
        x_batch = np.zeros((batch_size, shape[0], shape[1], shape[2], 1))
        y_batch = np.zeros((batch_size, shape[0] // down_scale, shape[1] // down_scale, shape[2] // down_scale, 12))
        f_batch = np.zeros((batch_size, shape[0] // down_scale, shape[1] // down_scale, shape[2] // down_scale),dtype=np.int32)
        batch_count = 0
        if shuff:
            file_prefixes_s = shuffle(file_prefixes)
        else:
            file_prefixes_s = file_prefixes

        for count, file_prefix in enumerate(file_prefixes_s):


            vox_tsdf, segmentation_label, vox_flags = process_evaluate(file_prefix, voxel_shape=(240, 144, 240), down_scale=4)

            x_batch[batch_count] = vox_tsdf.reshape((shape[0], shape[1], shape[2],1))
            y_batch[batch_count] = to_categorical(segmentation_label.reshape((60, 36, 60,1)), num_classes=12)
            f_batch[batch_count] = vox_flags.reshape((60, 36, 60))
            batch_count += 1
            if batch_count == batch_size:
                yield x_batch, y_batch, f_batch
                x_batch = np.zeros((batch_size, shape[0], shape[1], shape[2], 1)) #channels last
                y_batch = np.zeros((batch_size, shape[0] // down_scale, shape[1] // down_scale, shape[2] // down_scale, 12))
                f_batch = np.zeros((batch_size, shape[0]//down_scale, shape[1]//down_scale, shape[2]//down_scale),dtype=np.int32)
                batch_count = 0

        if(batch_count > 0):
            yield x_batch[:batch_count], y_batch[:batch_count],f_batch[:batch_count]

def voxel_export(name, vox, shape, tsdf_range=None):

    from array import array
    import struct



    vox = vox.reshape(shape)

    coord_x=array('i')
    coord_y=array('i')
    coord_z=array('i')
    voxel_v=array('f')

    count = 0

    for x in range(shape[0]):
        for y in range (shape[1]):
             for z in range(shape[2]):
                 if ((tsdf_range is None) and (vox[x,y,z]!=0)) or \
                    ((not tsdf_range is None) and  ((vox[x,y,z]<=tsdf_range[0] or vox[x,y,z]>=tsdf_range[1]) and vox[x,y,z]!=1)):
                     coord_x.append(x)
                     coord_y.append(y)
                     coord_z.append(z)
                     voxel_v.append(vox[x,y,z])
                     count += 1
    print("saving...")
    f = open(name, 'wb')
    f.write(struct.pack("i", count))
    f.write(struct.pack(str(count)+"i", *coord_x))
    f.write(struct.pack(str(count)+"i", *coord_y))
    f.write(struct.pack(str(count)+"i", *coord_z))
    f.write(struct.pack(str(count)+"f", *voxel_v))
    f.close()

    print(count, "done...")

def prediction_export(name, vox, weights, shape, tsdf_range=None):

    from array import array
    import struct



    vox = vox.reshape(shape)

    coord_x=array('i')
    coord_y=array('i')
    coord_z=array('i')
    voxel_v=array('f')

    count = 0

    for x in range(shape[0]):
        for y in range (shape[1]):
             for z in range(shape[2]):
                 if ((vox[x,y,z]!=0) and (weights[x,y,z]!=0)):
                     coord_x.append(x)
                     coord_y.append(y)
                     coord_z.append(z)
                     voxel_v.append(vox[x,y,z])
                     count += 1
    print("saving...")
    f = open(name, 'wb')
    f.write(struct.pack("i", count))
    f.write(struct.pack(str(count)+"i", *coord_x))
    f.write(struct.pack(str(count)+"i", *coord_y))
    f.write(struct.pack(str(count)+"i", *coord_z))
    f.write(struct.pack(str(count)+"f", *voxel_v))
    f.close()

    print(count, "done...")
