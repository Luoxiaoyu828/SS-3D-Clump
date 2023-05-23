from tf_fits.image import image_decode_fits
import tensorflow as tf
import numpy as np
from scipy import ndimage
import os
import astropy.io.fits as fits


def fits_read(file_name):
    header = 0
    img = tf.io.read_file(file_name)
    img = image_decode_fits(img, header)
    img = tf.reshape(img, [1, 61, 61, 61, 1])

    return img


def scale_tf(img, scale):
    temp = img
    scaling_factor = np.array([scale / i for i in temp.shape]).min()
    # 体积归一化
    if scale < max(temp.shape):
        temp = ndimage.zoom(temp, (scaling_factor, scaling_factor, scaling_factor))

    [temp_x, temp_y, temp_z] = temp.shape
    pad_x, pad_y, pad_z = (scale - temp_x) // 2, (scale - temp_y) // 2, (scale - temp_z) // 2
    temp1 = np.pad(temp, ((pad_x, pad_x), (pad_y, pad_y), (pad_z, pad_z)), 'constant')
    [temp_x, temp_y, temp_z] = temp1.shape
    temp1 = np.pad(temp1, ((scale - temp_x, 0), (scale - temp_y, 0), (scale - temp_z, 0)), 'constant')

    data_bg = fits.getdata(r'/home/data/luoxy/work/Deep_Cluster_loc/bg.fits')
    temp1 = temp1 + data_bg
    temp1 = (temp1 - temp1.min()) / (temp1.max() - temp1.min())
    return temp1


def preprocess(x, y):

    # header = 0
    # img = tf.io.read_file(x[0])
    # img = image_decode_fits(img, header)
    # img = img.numpy()
    img = fits.getdata(x[0].numpy())
    scale = 30
    temp1 = scale_tf(img, scale)
    x = tf.cast(temp1, tf.float32)
    # y = tf.keras.utils.to_categorical(y, num_classes=2)
    # y = tf.squeeze(y)
    x = tf.expand_dims(x, -1)

    return x, y


def tf_serialize_example(x, y):
    x, y = tf.py_function(
    preprocess,
    (x, y),  # pass these args to the above function.
        (tf.float32, tf.int32))      # the return type is `tf.string`.# The result is a scalar
    return x, y


if __name__ == '__main__':

    # fits_file = r'../0130+015_L/MWISP012.760+01.427+27.403.fits'
    # header = 0
    # img = tf.io.read_file(fits_file)
    # img = image_decode_fits(img, header)

    path = r'D:\OneDrive_lxy\OneDrive - ctgu.edu.cn\Deep_Cluster_loc\data\test'
    file_list1 = [os.path.join(path, item) for item in os.listdir(path)]
    file_label1 = [1 for _ in file_list1]

    # for item in dataset_value.take(2):
    #     print(item.shape)
    #     print(type(item))
    #     print(item.numpy())
    #     break

    # features_dataset = tf.data.Dataset.from_tensor_slices((file_list1, file_label1))
    # features_dataset_value = features_dataset.map(preprocess)
    features = tf.constant(file_list1, tf.string, shape=(len(file_list1), 1))  # ==> 3x2 tensor
    labels = tf.constant(file_label1, shape=(len(file_list1), 1))  # ==> 3x1 tensor

    features_dataset = tf.data.Dataset.from_tensor_slices(features)
    labels_dataset = tf.data.Dataset.from_tensor_slices(labels)
    dataset = tf.data.Dataset.zip((features_dataset, labels_dataset))
    # Use `take(1)` to only pull one example from the x_dataset.
    for f0, f1 in dataset.take(1):
        print(f0)
        print(f1)
    tf_serialize_example(f0, f1)
    dataset = dataset.map(tf_serialize_example)

