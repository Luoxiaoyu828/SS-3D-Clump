import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, Sequential
import numpy as np
import pandas as pd
import os
import tqdm
from scipy.spatial.distance import pdist, squareform, cdist
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
from ConstrainedKMeans.src.ConstrainedKMeans import ConstrainedKMeans as CKM

from model_ulti.make_pic import get_p_distribution, plot_train_result, plot_test_acc_nmi, \
    plot_test_acc_reassign, decisionGraph
from deep_clustering import load_dataset, BasicBlock, load_dataset_all


np.random.seed(1)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpus = tf.config.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
    # 1.TF_CPP_MIN_LOG_LEVEL = 1 //默认设置，为显示所有信息
    # 2.TF_CPP_MIN_LOG_LEVEL = 2 //只显示error和warining信息
    # 3.TF_CPP_MIN_LOG_LEVEL = 3 //只显示error信息
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

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


if __name__ == '__main__':
    for ratio in [0.4]:
        feat_num = 256
        load_model_wight = True
        model_path = r'model/model_R2_to_R16_test_model_all_040/epoch_model/model_044.h5'

        para_dict = {'model_path': model_path, 'n_clusters': 2, 'num_epochs': 10, 'batch_size': 1000,
                     'feature_num': feat_num, 'block_num': 1, 'learning rate': 0.005, 'region': 'All',
                     'savepath_log': 'model_all_%03d_fellwalker_1' % (100*ratio), 'train_num': 5, 'model': 'R2'}

        data_set_path_R16 = r'/home/data/clumps_share/real_data_cube_set/background/R16_background/clump_dataset/'
        data_test_path_R16 = r'/home/data/clumps_share/real_data_cube_set/background/R16_background/clump_dataset/'
        seed_samples_path_R16 = r'/home/data/clumps_share/real_data_cube_set/background/R16_background/seed_samples/'

        data_set_path_G0100 = '/home/data/clumps_share/real_data_cube_set/background/G100+00_background/clump_dataset/'
        data_test_path_G0100 = r'/home/data/clumps_share/real_data_cube_set/background/G100+00_background/clump_dataset/'
        seed_samples_path_G0100 = r'/home/data/clumps_share/real_data_cube_set/background/G100+00_background/seed_samples/'

        data_set_path_R2 = r'/home/data/clumps_share/real_data_cube_set/background/R2_background/clump_dataset/'
        data_test_path_R2 = r'/home/data/clumps_share/real_data_cube_set/background/R2_background/clump_dataset/'
        seed_samples_path_R2 = r'/home/data/clumps_share/real_data_cube_set/background/R2_background/seed_samples/'

        data_set_path_list = [data_set_path_R16, data_set_path_G0100, data_set_path_R2]
        seed_samples_path_list = [seed_samples_path_R16, seed_samples_path_G0100, seed_samples_path_R2]
        data_test_path_list = [data_test_path_R16, data_test_path_G0100, data_test_path_R2]

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

        dataset, sample_train_num = load_dataset_all(data_set_path_list, batch_size=para_dict['batch_size'])
        dataset_seed, seed_samples_num = load_dataset_all(seed_samples_path_list, batch_size=para_dict['batch_size'])
        dataset_test, sample_test_num = load_dataset_all(data_test_path_list, batch_size=para_dict['batch_size'])
        sample_num = sample_train_num + seed_samples_num
        print('x_dataset over')

        feature_all = np.zeros((sample_num, para_dict['feature_num']), np.float32)   # 模型提取的特征
        y_true_all = np.zeros(sample_num, np.float32)                   # 样本标签

        y_pred_all = np.zeros((sample_num, para_dict['n_clusters']), np.float32)       # 模型预测标签
        x_all = np.zeros((sample_num, 30, 30, 30, 1), np.float32)       # 样本数据

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