import astropy.io.fits as fits
import numpy as np
import tensorflow as tf
from scipy import ndimage
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from deep_clustering_odc_new_0406 import ResNet_dc
from model_ulti.get_tfrecord import tf_serialize_example
import tqdm


def scale_tf(img, scale):
    scaling_factor = np.array([scale / i for i in img.shape]).min()
    # 体积归一化
    if scale < max(img.shape):
        img = ndimage.zoom(img, (scaling_factor, scaling_factor, scaling_factor))

    [temp_x, temp_y, temp_z] = img.shape
    pad_x, pad_y, pad_z = (scale - temp_x) // 2, (scale - temp_y) // 2, (scale - temp_z) // 2
    temp1 = np.pad(img, ((pad_x, pad_x), (pad_y, pad_y), (pad_z, pad_z)), 'constant')
    [temp_x, temp_y, temp_z] = temp1.shape
    temp1 = np.pad(temp1, ((scale - temp_x, 0), (scale - temp_y, 0), (scale - temp_z, 0)), 'constant')

    data_bg = fits.getdata(r'example_data/bg.fits')
#     data_bg = np.random.normal(0, 0.23, size=[30,30,30])
    temp1 = temp1 + data_bg
    temp1 = (temp1 - temp1.min()) / (temp1.max() - temp1.min())
    return temp1


def get_trained_model(model_path):
    model = ResNet_dc([1], feature_num=256, num_classes=2)
    model.build(input_shape=(None, 30, 30, 30, 1))
    model.load_weights(model_path)
    print('ok: load model from %s' % model_path)
    return model


def get_pic_png(data_cube, title, savepath=None):
    fig = plt.figure()
    for i in range(3):
        ax = fig.add_subplot(1, 3, i + 1)
        ax.imshow(data_cube.sum(i))
        if i == 1:
            ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    if savepath is None:
        plt.show()
    else:
        fig.savefig(savepath, dpi=200, format='png')
        plt.close(fig)


def get_label_by_trained_model_png(data_path, model, savepath):
    scale = 30
    temp = fits.getdata(data_path)
    temp_max = temp.max()
    temp_sum = temp.sum()
    temp_v = np.where(temp > 0)[0].shape[0]
    info = 'max_value: %.2f, sum: %.2f, volumn: %d\n' %(temp_max, temp_sum, temp_v)
    data_cube = scale_tf(temp, scale)
    data = tf.cast(data_cube, tf.float32)
    data = tf.expand_dims(data, -1)
    data = tf.expand_dims(data, 0)
    y_test_pred, _ = model.predict(data)
    get_pic_png(data_cube, info + '\n %.3f_%.3f' % (y_test_pred[0, 0], y_test_pred[0, 1]), savepath)


if __name__ == '__main__':

    model_path = 'SS-3D-Clump_model/ss-3d-clump_model_all/deep_clustering_All.h5'
    ss_3d_clump_model = get_trained_model(model_path)
    data_path = 'example_data/clump_fits/1_MWISP010.037-00.412+33.022.fits'
    get_label_by_trained_model_png(data_path, ss_3d_clump_model)

