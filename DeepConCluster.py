import matplotlib.pyplot as plt
import tensorflow as tf
from tf_fits.image import image_decode_fits
from tensorflow.keras import layers, Sequential
import numpy as np
import pandas as pd
import os
import tqdm
from deep_clustering import BasicBlock
from model_ulti.get_tfrecord import scale_tf, tf_serialize_example
from deep_clustering import load_dataset
from scipy.spatial.distance import pdist, squareform, cdist
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
from ConstrainedKMeans.src.ConstrainedKMeans import ConstrainedKMeans as CKM
# import tensorflow as tf
np.random.seed(1)
# tf.random.set_seed(1)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpus = tf.config.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
# 1.TF_CPP_MIN_LOG_LEVEL = 1 //默认设置，为显示所有信息
# 2.TF_CPP_MIN_LOG_LEVEL = 2 //只显示error和warining信息
# 3.TF_CPP_MIN_LOG_LEVEL = 3 //只显示error信息
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
# 定义ResNet
class ResNet_dc(tf.keras.Model):

    def build_resblock(self, filter_num, blocks, stride=1):
        res_blocks = Sequential()
        res_blocks.add(BasicBlock(filter_num, stride))
        for _ in range(1, blocks):
            res_blocks.add(BasicBlock(filter_num, stride=1))
        return res_blocks

    def __init__(self, layer_dims, feature_num=64, num_classes=2):  # mnist有10类,此时2类
        super(ResNet_dc, self).__init__()
        self.stem = Sequential([layers.Input(shape=(30, 30, 30, 1)),
                                layers.BatchNormalization(axis=1),
                                layers.Conv3D(16, (3, 3, 3), strides=(2, 2, 2), padding='same'),  # 15，15，15
                                layers.Activation('relu'),
                                layers.MaxPool3D(pool_size=(2, 2, 2), strides=(1, 1, 1), padding='same')
                                ])
        self.layer1 = self.build_resblock(32, layer_dims[0], stride=2)  # 8，8，8
        self.layer2 = Sequential([layers.Dropout(0.5),
                                  layers.Conv3D(64, (3, 3, 3), strides=(2, 2, 2), padding='same')])  # 4，4，4

        self.layer3 = Sequential([layers.Flatten(),
                                  layers.Dense(512, activation='relu'),
                                  layers.Dense(feature_num, activation='relu')
                                  ], name='Dense_1')

        self.layer4 = Sequential([layers.Dense(16, activation='relu'),
                                  layers.Dense(num_classes, activation='softmax')], name='Dense_2')

    def call(self, inputs, training=None):
        x = self.stem(inputs, training=training)
        x = self.layer1(x, training=training)
        x = self.layer2(x, training=training)
        # x = self.layer3(x,training=training)
        extrect_feature = self.layer3(x, training=training)
        x = self.layer4(extrect_feature, training=training)

        return x, extrect_feature


def loss(model, x, y, weight=False):
    # training=training is needed only if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    y_pred, feature = model(x, training=True)

    if weight:
        sample_weight_ = np.zeros(y.shape[0])
        idx_1 = np.where(y[:, 1] == 1)[0]
        idx_0 = np.where(y[:, 0] == 1)[0]
        sample_weight_[idx_1] = 1 / idx_1.shape[0] * 0.5 * sample_weight_.shape[0]
        sample_weight_[idx_0] = 1 / idx_0.shape[0] * 0.5 * sample_weight_.shape[0]
        loss0 = loss_object(y_true=y, y_pred=y_pred, sample_weight=sample_weight_)
    else:
        loss0 = loss_object(y_true=y, y_pred=y_pred)

    return loss0


def grad(model, x, y, weight=False):

    with tf.GradientTape() as tape:
        loss_value = loss(model, x, y, weight=weight)
    grads = tape.gradient(loss_value, model.trainable_variables)
    return loss_value, grads


def load_dataset1(data_set_path):
    file_list = os.listdir(data_set_path)
    file_list1 = [os.path.join(data_set_path, item) for item in file_list]
    file_label1 = np.array([item.split('_')[0] for item in file_list], np.int32)

    labels = tf.constant(file_label1, tf.int32, shape=(len(file_list1)))  # ==> 3x1 tensor
    labels_dataset = tf.data.Dataset.from_tensor_slices(labels)
    features = tf.constant(file_list1, tf.string, shape=(len(file_list1)))  # ==> 3x2 tensor
    features_dataset = tf.data.Dataset.from_tensor_slices(features)

    dataset = tf.data.Dataset.zip((features_dataset, labels_dataset))
    dataset = dataset.shuffle(buffer_size=len(file_list1))
    dataset = dataset.map(tf_serialize_example)

    return dataset, labels_dataset, file_label1


def preprocess_x(x):

    header = 0
    img = tf.io.read_file(x[0])
    img = image_decode_fits(img, header)
    img = img.numpy()

    scale = 30
    temp1 = scale_tf(img, scale)
    x = tf.cast(temp1, tf.float32)
    x = tf.expand_dims(x, -1)

    return x


def tf_serialize_example_x(x):
    x = tf.py_function(preprocess_x, [x], tf.float32)      # the return type is `tf.string`.# The result is a scalar
    return x


def decisionGraph(feature_all, show=False):
    if show:
        feature_all_n = (feature_all - feature_all.min(0)) / (feature_all.max(0) - feature_all.min(0) + 0.0001)
        dist_array = pdist(feature_all_n, metric='euclidean')
        dist_m = squareform(dist_array, 'tomatrix')
        NE = dist_m.shape[0]
        sort_distrow = np.sort(dist_array)
        dc = sort_distrow[int(NE * (NE - 1) / 2 * 0.02)]
        rho = np.sum(np.exp(-1 * (dist_m / dc) ** 2), axis=0) - 1

        delta = np.zeros([NE], np.float32)
        INN = np.inf * np.ones_like(delta, np.int32)
        rhoRho = np.argsort(-1 * rho)   # 降序的索引
        for rho_i in tqdm.tqdm(range(1, NE)):
            delta[rhoRho[rho_i]] = max(dist_m[rhoRho[rho_i], ...])
            for rho_j in range(rho_i):
                if delta[rhoRho[rho_i]] > dist_m[rho_i, rho_j]:
                    delta[rhoRho[rho_i]] = dist_m[rho_i, rho_j]
                    INN[rhoRho[rho_i]] = rhoRho[rho_j]
        delta[rhoRho[0]] = max(delta)
        INN[rhoRho[0]] = 0

        plt.figure()
        plt.scatter(rho, delta+0.01)
        # plt.xscale('log')
        # plt.yscale('log')
        plt.show()


def re_sign_label(cen_new, cen_old, label_new, label_old):
    # update cluster --> y_pseudo
    d_0 = np.linalg.norm(cen_new - cen_old[0, ...], ord=2, axis=1)
    aa = 'change'
    if d_0[0] < d_0[1]:
        label = label_new
        aa = 'not ' + aa
    else:
        label = 1 - label_new
        cen_temp = cen_new
        cen_temp[0, ...] = cen_new[1, ...]
        cen_temp[1, ...] = cen_new[0, ...]
        cen_new = cen_temp

    NMI_0 = normalized_mutual_info_score(label_old, label)
    NMI_1 = normalized_mutual_info_score(label_old, label_new)
    print('%s: %.1f, %.1f' % (aa, NMI_0, NMI_1))
    return cen_new, label


def plot_train_result(Result, savepath=None):
    # colnames = ['acc_kmeans', 'recall_train', 'precision_train', 'f1_train', 'loss_value_train', 'accuracy_train',
    #             'recall_test', 'precision_test', 'f1_test', 'loss_value_test', 'accuracy_test']
    acc_kmeans = Result['acc_kmeans'].values

    recall_train = Result['recall_train'].values
    precision_train = Result['precision_train'].values
    f1_train = Result['f1_train'].values
    loss_value_train = Result['loss_value_train'].values
    accuracy_train = Result['accuracy_train'].values

    recall_test = Result['recall_test'].values
    precision_test = Result['precision_test'].values
    f1_test = Result['f1_test'].values
    loss_value_test = Result['loss_value_test'].values
    accuracy_test = Result['accuracy_test'].values

    # 画损失函数图
    fig = plt.figure(figsize=(10, 6))
    # 坐标轴的刻度设置向内(in)或向外(out)
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'

    ax1 = fig.add_subplot(221)
    ax1.tick_params(top=True, right=True)
    ax1.plot(loss_value_train, label='Train loss')
    ax1.plot(loss_value_test, label='Test loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()

    # subplot acc_train
    ax3 = fig.add_subplot(222)
    ax3.tick_params(top=True, right=True)
    ax3.plot(accuracy_train, label='Train accuracy')
    ax3.plot(accuracy_test, label='Test accuracy')
    ax3.plot(acc_kmeans, label='K-Means accuracy')
    ax3.set_xlabel('Epochs')
    ax3.set_ylabel('Accuracy')
    ax3.legend()

    # subplot acc_train
    ax2 = fig.add_subplot(223)
    ax2.tick_params(top=True, right=True)
    ax2.plot(recall_train, label='Train recall')
    ax2.plot(recall_test, label='Test recall')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Recall')
    ax2.legend()

    # subplot acc_train
    ax4 = fig.add_subplot(224)
    ax4.tick_params(top=True, right=True)
    ax4.plot(precision_train, label='Train precision')
    ax4.plot(precision_test, label='Test precision')
    ax4.plot(f1_train, label='Train f1')
    ax4.plot(f1_test, label='Test f1')
    ax4.set_xlabel('Epochs')
    ax4.set_ylabel('Precision & f1')
    ax4.legend()

    plt.tight_layout()
    if savepath is None:
        plt.show()
    else:
        fig.savefig(savepath, dpi=600, format='png')
        plt.close(fig)


def plot_test_acc_nmi(Result, savepath=None):
    # colnames = ['acc_kmeans', 'recall_train', 'precision_train', 'f1_train', 'loss_value_train', 'accuracy_train',
    #             'recall_test', 'precision_test', 'f1_test', 'loss_value_test', 'accuracy_test', 'NMI_t_t1']
    recall_test = Result['recall_test'].values
    precision_test = Result['precision_test'].values
    f1_test = Result['f1_test'].values
    loss_value_test = Result['loss_value_test'].values
    accuracy_test = Result['accuracy_test'].values
    NMI_t_label = Result['NMI_t_label'].values
    NMI_t_t1 = Result['NMI_t_t1'].values
    resign_num_ratio = Result['cluster_resign_num'].values / Result['cluster_resign_num'].values.max()

    # 画损失函数图
    fig = plt.figure(figsize=(10, 3))
    # 坐标轴的刻度设置向内(in)或向外(out)
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'

    ax1 = fig.add_subplot(131)
    ax1.tick_params(top=True, right=True)
    ax1.plot(NMI_t_label, label='MNI t / labels')
    ax1.plot(NMI_t_t1, label='NMI t-1 / t')
    ax1.plot(resign_num_ratio, label='Reassignment ratio')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('MNI Test Dataset')
    ax1.legend()

    # subplot acc_train
    ax2 = fig.add_subplot(132)
    ax2.tick_params(top=True, right=True)
    ax2.plot(recall_test, label='Test recall')
    ax2.plot(precision_test, label='Test precision')

    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Evaluation Indicators')
    ax2.legend()

    ax3 = fig.add_subplot(133)
    ax3.tick_params(top=True, right=True)
    ax3.plot(f1_test, label='Test F1 score')
    ax3.plot(accuracy_test, label='Test accuracy')

    ax3.set_xlabel('Epochs')
    ax3.set_ylabel('Evaluation Indicators')
    ax3.legend()

    plt.tight_layout()
    if savepath is None:
        plt.show()
    else:
        fig.savefig(savepath, dpi=600, format='png')
        plt.close(fig)


def plot_test_acc_reassign(Result, savepath=None):
    # colnames = ['acc_kmeans', 'recall_train', 'precision_train', 'f1_train', 'loss_value_train', 'accuracy_train',
    #             'recall_test', 'precision_test', 'f1_test', 'loss_value_test', 'accuracy_test', 'NMI_t_t1']

    f1_test = Result['f1_test'].values
    NMI_t_label = Result['NMI_t_label'].values
    NMI_t_t1 = Result['NMI_t_t1'].values
    # 画损失函数图
    fig = plt.figure(figsize=(10, 3))
    # 坐标轴的刻度设置向内(in)或向外(out)
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'

    ax1 = fig.add_subplot(131)
    ax1.tick_params(top=True, right=True)
    ax1.plot(NMI_t_label)
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('NMI t / labels')

    # subplot acc_train
    ax2 = fig.add_subplot(132)
    ax2.tick_params(top=True, right=True)

    ax2.plot(NMI_t_t1)
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('NMI t-1 / t')

    ax3 = fig.add_subplot(133)
    ax3.tick_params(top=True, right=True)
    ax3.plot(f1_test)
    ax3.set_ylim([0.5, 1])
    # ax3.plot(accuracy_test, label='Test accuracy')

    ax3.set_xlabel('Epochs')
    ax3.set_ylabel('$F_1$')

    plt.tight_layout()
    if savepath is None:
        plt.show()
    else:
        fig.savefig(savepath, dpi=600, format='png')
        plt.close(fig)


def get_r_p_f1_loss(y_true_all, y_pred_all):

    recall = recall_score(y_true_all, y_pred_all.argmax(1))
    precision = precision_score(y_true_all, y_pred_all.argmax(1))
    f1 = f1_score(y_true_all, y_pred_all.argmax(1))
    confusion = confusion_matrix(y_true_all, y_pred_all.argmax(1))
    loss_value = loss_object(y_true=y_true_all, y_pred=y_pred_all.argmax(1).astype(np.float32))
    accuracy = accuracy_score(y_true_all, y_pred_all.argmax(1))  # 模型预测的准确率
    return recall, precision, f1, confusion, loss_value, accuracy


def save_paraset(result_path_para, para_dict):
    with open(result_path_para, 'w') as ww:
        for key, values in zip(para_dict.keys(), para_dict.values()):
            print('{}: {}'.format(key, values), file=ww)


def get_stable_samples_idx(feature_all, pseudo_label, ratio, cluster_id=1):
    """
    kmeans聚类完成后，计算聚类中心，将最靠近中心的那部分（ratio）点取出来，进行训练；
    :param feature_all:
    :param pseudo_label:
    :param ratio:
    :param cluster_id:
    :return:
    """
    idx = np.where(pseudo_label == cluster_id)[0]
    x_cen = feature_all[idx, ...].mean(axis=0)
    x_cen = x_cen.reshape([1, x_cen.shape[0]])
    cdistance = cdist(feature_all[idx, ...], x_cen, metric='euclidean')

    cdistance_ = cdistance.copy().reshape(cdistance.shape[0])
    cdistance_.sort()  # 升序
    idx_threshold = int(idx.shape[0] * ratio)
    cd_1_th = cdistance_[idx_threshold]

    idx_1_ = np.where(cdistance <= cd_1_th)[0]

    return idx[idx_1_]


def get_stable_pseudo_label(feature_all, pseudo_label, can_change, ratio=0.5):
    """
    在kmeans聚类完成后，计算聚类中心，将最靠近中心的那部分（ratio）点取出来，进行训练；
    同时利用can_change，将种子样本一起取出进行训练。

    :param feature_all: N*256
    :param pseudo_label: N*1
    :param can_change:
    :param ratio:
    :return:

    code:
    idx_1 = np.where(pseudo_label == 1)[0]
    x_1_cen = feature_all[idx_1, ...].mean(axis=0)
    x_1_cen = x_1_cen.reshape([1, x_1_cen.shape[0]])
    cd_1 = cdist(feature_all[idx_1, ...], x_1_cen, metric='euclidean')
    # sample_weight_[idx_1] = 1 / idx_1.shape[0] * 0.5
    idx_0 = np.where(pseudo_label == 0)[0]
    x_0_cen = feature_all[idx_0, ...].mean(axis=0)
    x_0_cen = x_0_cen.reshape([1, x_0_cen.shape[0]])
    cd_0 = cdist(feature_all[idx_0, ...], x_0_cen, metric='euclidean')

    cd_0_ = cd_0.copy().reshape(cd_0.shape[0])
    cd_0_.sort()   # 升序
    idx_0_th = int(idx_0.shape[0] * ratio)
    cd_0_th = cd_0[idx_0_th][0]

    cd_1_ = cd_1.copy().reshape(cd_1.shape[0])
    cd_1_.sort()  # 升序
    idx_1_th = int(idx_1.shape[0] * (1 - ratio))
    cd_1_th = cd_1[idx_1_th][0]

    idx_0_ = np.where(cd_0 <= cd_0_th)[0]
    idx_1_ = np.where(cd_1 <= cd_1_th)[0]
    idx = np.union1d(idx_0[idx_0_], idx_1[idx_1_])
    """

    idx_1 = get_stable_samples_idx(feature_all, pseudo_label, ratio, cluster_id=1)
    idx_0 = get_stable_samples_idx(feature_all, pseudo_label, ratio, cluster_id=0)
    idx = np.union1d(idx_0, idx_1)
    idx_seed = np.where(can_change == 0)[0]   # 将种子样本的下标取出放入，进行训练
    idx = np.union1d(idx, idx_seed)

    return idx


def get_dataset(pseudo_label, x_dataset, index_dataset, batch_size=1000):
    pseudo_label_hot = tf.keras.utils.to_categorical(pseudo_label, num_classes=2)
    pseudo_labels_dataset = tf.data.Dataset.from_tensor_slices(pseudo_label_hot)
    dataset_new = tf.data.Dataset.zip((x_dataset, pseudo_labels_dataset, index_dataset))
    dataset_new = dataset_new.batch(batch_size=batch_size)

    return dataset_new


def get_p_distribution(y_true_label, y_test_pred, savepath):
    # save_path = 'result/R2_feature_256_block_1_test_seed_0507_ratio_04_01'
    # savepath = 'result/R2_feature_256_block_1_test_seed_0507_ratio_04_01/matrix.png'
    # os.makedirs(save_path, exist_ok=True)
    TP = []
    FN = []
    FP = []
    TN = []
    for true_label, y_pred in zip(y_true_label, y_test_pred):
        pred_label = np.argmax(y_pred)

        if true_label == 1 and pred_label == 1:
            TP.append(y_pred)
        if true_label == 1 and pred_label == 0:
            FN.append(y_pred)
        if true_label == 0 and pred_label == 1:
            FP.append(y_pred)
        if true_label == 0 and pred_label == 0:
            TN.append(y_pred)

    TP = np.array(TP)
    TN = np.array(TN)
    FP = np.array(FP)
    FN = np.array(FN)

    fig = plt.figure(figsize=[10, 6])
    ide_style = ['TP', 'FP', 'FN', 'TN']

    for i, item in enumerate([TP, FP, FN, TN]):
        ax1 = fig.add_subplot(2, 2, i + 1)
        ax1.hist(y_test_pred[:, 1], 38)
        ax1.tick_params(top=True, right=True)
        if item.shape[0] > 0:
            if ide_style[i] == 'TP':
                data = item[:, 1]
            if ide_style[i] == 'FP':
                data = item[:, 1]
            if ide_style[i] == 'FN':
                data = item[:, 1]
            if ide_style[i] == 'TN':
                data = item[:, 1]
            ax1.hist(data, 20, label='%s\nnum=%d' % (ide_style[i], data.shape[0]))
        else:
            ax1.hist(y_test_pred[:, 1], 38, label='%s\nnum=%d' % (ide_style[i], 0))
        ax1.legend()
    fig.savefig(savepath, dpi=600, format='png')
    plt.close(fig)


if __name__ == '__main__':
    for ratio in [0.4]:
        feat_num = 256
        load_model_wight = True
        # model_path = r'model/G0100_feature_256_block_1_test_seed_0507_ratio_020_01/epoch_model/model_117.h5'
        model_path = r'model/R2_feature_256_block_1_test_seed_0507_ratio_040_05/epoch_model/model_025.h5'
        # model_path = r'model/R16_feature_256_block_1_test_seed_0507_ratio_050_01/epoch_model/model_025.h5'
        para_dict = {'model_path': model_path, 'n_clusters': 2, 'num_epochs': 5, 'batch_size': 1000,
                     'feature_num': feat_num, 'block_num': 1, 'learning rate': 0.005, 'region': 'R16',
                     'savepath_log': 'fine_tuning_%03d_01' % (100*ratio), 'train_num': 5,
                     'model': 'R2'}

        if para_dict['region'] == 'M16':
            data_set_path = r'/home/data/clumps_share/real_data_cube_set/M16_clumps/train'
            data_test_path = r'/home/data/clumps_share/real_data_cube_set/M16_clumps/test'
            seed_samples_path = ''
        elif para_dict['region'] == 'R16':
            data_set_path = r'/home/data/clumps_share/real_data_cube_set/R16_clumps_masked/R16_150/train/'
            data_test_path = r'/home/data/clumps_share/real_data_cube_set/R16_clumps_masked/R16_150/test'
            seed_samples_path = r'/home/data/clumps_share/real_data_cube_set/G0100+00_masked/seed_samples/'
        elif para_dict['region'] == 'G0100':
            data_set_path = '/home/data/clumps_share/real_data_cube_set/G0100+00_masked/train_data2/'
            data_test_path = r'/home/data/clumps_share/real_data_cube_set/G0100+00_masked/test_data2/'
            seed_samples_path = r'/home/data/clumps_share/real_data_cube_set/G0100+00_masked/seed_samples/'
        else:
            data_set_path = r'/home/data/clumps_share/real_data_cube_set/R2_clumps_masked/train1'
            seed_samples_path = r'/home/data/clumps_share/real_data_cube_set/R2_clumps_masked/seed_samples'
            data_test_path = r'/home/data/clumps_share/real_data_cube_set/R2_clumps_masked/test1_revise_again'

        log_dir = os.path.join('model', 'model_%s_to_%s_test_%s' % (para_dict['model'],
            para_dict['region'], para_dict['savepath_log']))
        os.makedirs(log_dir, exist_ok=True)
        epoch_model_path = os.path.join(log_dir, 'epoch_model')
        os.makedirs(epoch_model_path, exist_ok=True)
        epoch_result_path = os.path.join(log_dir, 'epoch_result')
        os.makedirs(epoch_result_path, exist_ok=True)

        model = ResNet_dc([para_dict['block_num']], feature_num=para_dict['feature_num'], num_classes=2)
        model.build(input_shape=(None, 30, 30, 30, 1))
        if load_model_wight:
            model.load_weights(para_dict['model_path'])
            epoch_od = int(para_dict['model_path'].split('/')[-1].split('.')[0].split('_')[-1])
            print('ok: load model from %s' % para_dict['model_path'])
        else:
            epoch_od = 0
            print('model initial random')

        png_path = os.path.join(log_dir, 'loss_accuracy_%03d.png' % epoch_od)
        model_path = os.path.join(log_dir, 'deep_clustering_%03d.h5' % epoch_od)
        dot_img_file = os.path.join(log_dir, 'model_struct_%03d.png' % epoch_od)
        result_path = os.path.join(log_dir, 'r_p_f1_loss_%03d.csv' % epoch_od)
        result_path_para = os.path.join(log_dir, 'para_set_%03d.txt' % epoch_od)

        dataset = load_dataset(data_set_path, batch_size=para_dict['batch_size'])
        dataset_seed = load_dataset(seed_samples_path, batch_size=para_dict['batch_size'])
        seed_samples_num = len(os.listdir(seed_samples_path))
        sample_num = len(os.listdir(data_set_path)) + seed_samples_num
        print('x_dataset over')

        feature_all = np.zeros((sample_num, para_dict['feature_num']), np.float32)   # 模型提取的特征
        y_true_all = np.zeros(sample_num, np.float32)                   # 样本标签

        y_pred_all = np.zeros((sample_num, para_dict['n_clusters']), np.float32)       # 模型预测标签
        x_all = np.zeros((sample_num, 30, 30, 30, 1), np.float32)       # 样本数据

        dataset_test = load_dataset(data_test_path, batch_size=para_dict['batch_size'])
        sample_test_num = len(os.listdir(data_test_path))

        y_test_true = np.zeros([sample_test_num], np.float32)
        y_test_pred_all = np.zeros([sample_test_num, para_dict['n_clusters']], np.float32)

        f_i = 0
        for X_, Y in tqdm.tqdm(dataset):
            _, feature_all_ = model.predict(X_)
            st = f_i * para_dict['batch_size']
            ed = min((f_i + 1) * para_dict['batch_size'], sample_num - seed_samples_num)
            feature_all[st: ed, ...] = feature_all_
            y_true_all[st: ed] = Y.numpy()
            x_all[st: ed, ...] = X_.numpy()
            f_i += 1

        for X_, Y in tqdm.tqdm(dataset_seed):
            _, feature_all_ = model.predict(X_)
            st = sample_num - seed_samples_num
            ed = sample_num
            feature_all[st: ed, ...] = feature_all_
            y_true_all[st: ed] = Y.numpy()
            x_all[st: ed, ...] = X_.numpy()

        # 绘制决策图
        decisionGraph(feature_all, show=False)

        x_dataset = tf.data.Dataset.from_tensor_slices(x_all)  # 云核数据
        index_dataset = tf.data.Dataset.range(x_all.shape[0])
        # 对应的索引构建的数据集

        optimizer = tf.keras.optimizers.SGD(learning_rate=para_dict['learning rate'])

        can_change = np.ones(sample_num, np.int32)
        can_change[sample_num - seed_samples_num:sample_num] = 0          # 0 代表该数据点的标签不能随着聚类迭代而改变

        ckm = CKM(n_clusters=para_dict['n_clusters'])
        ckm.fit(feature_all, can_change, y_true_all)
        pseudo_label = ckm.get_labels()

        Result = []           # 保存每一轮训练过程中的损失、准确率 [epoch, loss_train, acc_train, loss_test, acc_test]
        for epoch in range(para_dict['num_epochs']):
            model_path_epoch = os.path.join(epoch_model_path, 'model_%03d.h5' % (epoch + epoch_od))
            save_p_path = os.path.join(epoch_result_path, 'matrix_%03d.png' % (epoch + epoch_od))

            acc_labeled = accuracy_score(y_true_all[sample_num - seed_samples_num:sample_num],
                                         pseudo_label[sample_num - seed_samples_num:sample_num])
            acc_kmeans = accuracy_score(y_true_all, pseudo_label)
            print(' ' * 40 + '\n' + '*' * 40)

            idx = get_stable_pseudo_label(feature_all, pseudo_label, can_change, ratio=0.4)
            pseudo_label_ = pseudo_label[idx]
            acc_kmeans_ = accuracy_score(y_true_all[idx], pseudo_label_)
            print('Accuracy for kmeans is %.1f, loc_acc is %.1f, labeled data acc_train = %.1f' % (
            100 * acc_kmeans, 100 * acc_kmeans_, acc_labeled))

            pseudo_label_hot_ = tf.keras.utils.to_categorical(pseudo_label_, num_classes=2)
            pseudo_labels_dataset_ = tf.data.Dataset.from_tensor_slices(pseudo_label_hot_)       # constrain-seed-kmeans 产生的伪标签

            x_dataset_ = tf.data.Dataset.from_tensor_slices(x_all[idx, ...])  # 云核数据
            index_dataset_ = tf.data.Dataset.from_tensor_slices(idx)

            # 构建kmeans聚类后比较可靠的样本数据集，进行模型训练
            dataset_new_stable = get_dataset(pseudo_label_, x_dataset_, index_dataset_, batch_size=para_dict['batch_size'])
            # Training loop - using batches of batch_size = 1000

            # 每一次再跑5遍
            for ii in range(para_dict['train_num']):
                # 数据，伪标签，索引，真实标签
                for x_, y_, inx_ in tqdm.tqdm(dataset_new_stable):
                    # Optimize the model
                    loss_value, grads = grad(model, x_, y_, weight=True)
                    optimizer.apply_gradients(zip(grads, model.trainable_variables))

            # 构建所有样本的数据集，重新计算feature 用于下一轮模型kmeans分类
            dataset_new = get_dataset(pseudo_label, x_dataset, index_dataset, batch_size=para_dict['batch_size'])
            for x_, _, inx_ in tqdm.tqdm(dataset_new):
                # update feature
                y_pred, feature = model(x_)
                feature_new = feature.numpy()
                feature_all[inx_, ...] = feature_new
                y_pred_all[inx_, ...] = y_pred.numpy()

            ckm.fit(feature_all, can_change, y_true_all)
            pseudo_label_new = ckm.get_labels()
            NMI_t_t1 = normalized_mutual_info_score(pseudo_label, pseudo_label_new)
            cluster_resign_num = sum(abs(pseudo_label_new - pseudo_label))
            pseudo_label = pseudo_label_new
            recall_train, precision_train, f1_train, confusion_train, loss_value_train, accuracy_train = get_r_p_f1_loss(
                y_true_all, y_pred_all)

            f_i = 0
            for X_1, y_true_label in tqdm.tqdm(dataset_test):
                y_test_pred, _ = model.predict(X_1)
                st = f_i * para_dict['batch_size']
                ed = min((f_i + 1) * para_dict['batch_size'], sample_num)
                y_test_pred_all[st: ed, ...] = y_test_pred
                y_test_true[st: ed] = y_true_label.numpy()
                f_i += 1
            NMI_t_label = normalized_mutual_info_score(y_test_true, y_test_pred_all.argmax(1))
            recall_test, precision_test, f1_test, confusion_test, loss_value_test, accuracy_test = get_r_p_f1_loss(
                y_test_true, y_test_pred_all)
            print(
                "Epoch {:03d}: Training loss: {:.3f}, Training accuracy: {:.3%}, Test loss: {:.3f}, Test accuracy: {:.3%}".format(
                    epoch, loss_value_train, accuracy_train, loss_value_test, accuracy_test))

            get_p_distribution(y_test_true, y_test_pred_all, save_p_path)

            Result.append(
                [acc_kmeans, recall_train, precision_train, f1_train, loss_value_train, accuracy_train,
                 recall_test, precision_test, f1_test, loss_value_test, accuracy_test, NMI_t_t1, NMI_t_label,
                 cluster_resign_num])
            model.save_weights(model_path_epoch)

        Result = np.vstack(Result)
        colnames = ['acc_kmeans', 'recall_train', 'precision_train', 'f1_train', 'loss_value_train', 'accuracy_train',
                    'recall_test', 'precision_test', 'f1_test', 'loss_value_test', 'accuracy_test', 'NMI_t_t1',
                    'NMI_t_label', 'cluster_resign_num']
        Result_df = pd.DataFrame(Result, columns=colnames)

        save_paraset(result_path_para, para_dict)
        Result_df.to_csv(result_path.replace('.txt', '_1.txt'), sep='\t', index=False)
        model.save_weights(model_path)
        plot_train_result(Result_df, savepath=png_path)
        plot_test_acc_reassign(Result_df, os.path.join(log_dir, 'model_trainig_reassign.png'))
        plot_test_acc_nmi(Result_df, os.path.join(log_dir, 'model_trainig.png'))

    # plot_model(model, to_file=dot_img_file, show_shapes=True)