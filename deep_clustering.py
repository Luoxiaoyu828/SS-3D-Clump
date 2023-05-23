import tensorflow as tf
from model_ulti.get_tfrecord import fits_read, preprocess, tf_serialize_example
from tensorflow.keras import layers, Sequential
import numpy as np
import os
import tqdm
from resnet_3 import ResNet as resnet
import pre_data_1 as pre_data
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
from sklearn.metrics.cluster import entropy, normalized_mutual_info_score

# tf.config.experimental_run_functions_eagerly(True)
# np.random.seed(1)

# os.environ["CUDA_VISIBLE_DEVICES"] = '0'
# gpus = tf.config.list_physical_devices(device_type='GPU')
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)


# 定义Basic Block
class BasicBlock(layers.Layer):
    def __init__(self, filter_num, stride=1):
        super(BasicBlock, self).__init__()
        # 第一小块
        self.conv1 = layers.Conv3D(filter_num, (3, 3, 3), strides=stride, padding='same')
        self.bn1 = layers.BatchNormalization()
        self.relu = layers.Activation('relu')
        # 第二小块
        self.conv2 = layers.Conv3D(filter_num, (3, 3, 3), strides=1, padding='same')
        self.bn2 = layers.BatchNormalization()
        if stride != 1:
            self.downsample = Sequential()
            self.downsample.add(layers.Conv3D(filter_num, (1, 1, 1), strides=stride, padding='same'))
        else:
            self.downsample = lambda x: x

    def call(self, inputs, training=None):
        identity = self.downsample(inputs)
        out = self.conv1(inputs)
        out = self.bn1(out, training=training)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out, training=training)
        out = layers.add([out, identity])
        out = tf.nn.relu(out)
        return out


# 定义ResNet
class ResNet_dc(tf.keras.Model):

    def build_resblock(self, filter_num, blocks, stride=1):
        res_blocks = Sequential()
        res_blocks.add(BasicBlock(filter_num, stride))
        for _ in range(1, blocks):
            res_blocks.add(BasicBlock(filter_num, stride=1))
        return res_blocks

    def __init__(self, layer_dims, num_classes=2, feature=True):  # mnist有10类,此时2类
        super(ResNet_dc, self).__init__()
        self.feature = feature
        self.stem = Sequential([layers.Input(shape=(30, 30, 30, 1)),
                                layers.Conv3D(16, (3, 3, 3), strides=(2, 2, 2), padding='same'),  # 15，15，15
                                layers.BatchNormalization(),
                                layers.Activation('relu'),
                                layers.MaxPool3D(pool_size=(2, 2, 2), strides=(1, 1, 1), padding='same')
                                ])
        self.layer1 = self.build_resblock(32, layer_dims[0], stride=2)  # 8，8，8
        self.layer2 = Sequential([layers.Dropout(0.5),
                                  layers.Conv3D(64, (3, 3, 3), strides=(2, 2, 2), padding='same')])  # 4，4，4

        self.layer3 = Sequential([layers.Flatten(),
                                  layers.Dense(512, activation='relu'),
                                  layers.Dense(64, activation='relu')
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
        if self.feature:
            return x, extrect_feature

        return x

    def get_loss(self, inputs):
        x = self.stem(inputs)
        x = self.layer1(x)
        x = self.layer2(x)
        extrect_feature = self.layer3(x)
        x = self.layer4(extrect_feature)
        extrect_feature = tf.convert_to_tensor(extrect_feature)

        # 计算loss
        kmeans = KMeans(n_clusters=2, random_state=0).fit(extrect_feature)
        Y_test_pred_hot = tf.keras.utils.to_categorical(1 - kmeans.labels_, num_classes=2)
        ans = tf.keras.losses.binary_crossentropy(Y_test_pred_hot, x)
        return ans


def load_model_weights(model, model_h5='model/model.h5'):
    model_old = resnet([1])
    model_old.build(input_shape=(None, 30, 30, 30, 1))
    model_old.load_weights(model_h5)
    for i in range(3):
        model.layers[i].set_weights(model_old.layers[i].get_weights())

    for i in range(3):
        model.layer3.layers[i].set_weights(model_old.layer3.layers[i].get_weights())

    model.layer4.set_weights(model_old.layer3.layers[3].get_weights())
    return model


class WeightedSDRLoss(tf.keras.losses.Loss):

    def __init__(self, name='WeightedSDRLoss'):
        super().__init__(name=name)

    def call(self, y_true, y_pred):

        # print('*****')

        # print('88888')
        # print(y_pred[0].shape)
        # print(y_pred[1].shape)
        # print(y_pred)
        # print('777777')
        if y_pred.shape[1] <= 2:
            return 0
        # y_true_all = y_true_all.numpy()
        # print(y_true_all)
        kmeans = KMeans(n_clusters=2, random_state=0).fit(y_pred)
        Y_test_pred_hot = tf.keras.utils.to_categorical(kmeans.labels_, num_classes=2)
        ans = tf.keras.losses.binary_crossentropy(Y_test_pred_hot, y_pred)
        return ans


# loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def loss(feature, y_pred):
    # training=training is needed only if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    # y_pred, feature = model(x, training=training)
    kmeans = KMeans(n_clusters=2, random_state=0).fit(feature)
    #
    y_pseudo = tf.keras.utils.to_categorical(kmeans.labels_, num_classes=2)   # seudo-labels
    loss0 = loss_object(y_true=y_pseudo, y_pred=y_pred)
    # loss1 = tf.keras.losses.binary_crossentropy(y_true_all=y_true_all, y_pred=y_pred)
    # epoch_loss_avg = tf.keras.metrics.Mean()
    # epoch_loss_avg.update_state(loss0)
    # loss0 = epoch_loss_avg.result()
    return loss0


def grad(model, feature_all, y_pred_all):

    with tf.GradientTape() as tape:
        loss_value = loss(feature_all, y_pred_all)
        print(1)
    grads = tape.gradient(loss_value, model.trainable_variables)
    return loss_value, grads


def loss1(model, x, targets, training):
    # training=training is needed only if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    y_, feature = model(x, training=training)
    kmeans = KMeans(n_clusters=2, random_state=0).fit(feature)
    Y_test_pred_hot = tf.keras.utils.to_categorical(kmeans.labels_, num_classes=2)
    return loss_object(y_true=Y_test_pred_hot, y_pred=y_)


def grad1(model, x, targets):
    with tf.GradientTape() as tape:
        loss_value = loss1(model, x, targets, training=False)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)


def load_dataset(data_set_path, batch_size=3000):
    temp = os.listdir(data_set_path)
    file_list1 = [os.path.join(data_set_path, item) for item in temp]
    file_label1 = np.array([item.split('_')[0] for item in temp], np.int32)
    features = tf.constant(file_list1, tf.string, shape=(len(file_list1), 1))  # ==> 3x2 tensor
    labels = tf.constant(file_label1, shape=(len(file_list1)))  # ==> 3x1 tensor

    features_dataset = tf.data.Dataset.from_tensor_slices(features)
    labels_dataset = tf.data.Dataset.from_tensor_slices(labels)
    dataset = tf.data.Dataset.zip((features_dataset, labels_dataset))
    dataset = dataset.shuffle(buffer_size=len(file_list1))
    dataset = dataset.map(tf_serialize_example)
    dataset = dataset.batch(batch_size=batch_size)
    return dataset


def load_dataset_M16_500():
    data_set_path = '/home/data/clumps_share/real_data_cube_set/M16_clumps/train/'
    temp = [item for item in os.listdir(data_set_path) if not item.split('_')[-2].isalpha()]
    sample_num = len(temp)

    file_list1 = [os.path.join(data_set_path, item) for item in temp]
    file_label1 = np.array([item.split('_')[0] for item in temp], np.int32)
    features = tf.constant(file_list1, tf.string, shape=(len(file_list1), 1))  # ==> 3x2 tensor
    labels = tf.constant(file_label1, shape=(len(file_list1)))  # ==> 3x1 tensor

    features_dataset = tf.data.Dataset.from_tensor_slices(features)
    labels_dataset = tf.data.Dataset.from_tensor_slices(labels)
    dataset = tf.data.Dataset.zip((features_dataset, labels_dataset))
    dataset = dataset.shuffle(buffer_size=len(file_list1))
    dataset = dataset.map(tf_serialize_example)
    dataset = dataset.batch(batch_size=sample_num)
    return dataset, sample_num


def main():
    num_epochs = 500
    batch_size = 3000
    data_set_path = r'/home/data/clumps_share/real_data_cube_set/R2_R16_clumps'
    train_pseudo_label = True
    load_trained_model = True

    model = ResNet_dc([1])
    model.build(input_shape=(None, 30, 30, 30, 1))
    if load_trained_model:
        model_h5 = 'model/deep_clustering_838.h5'
        # model.load_weights(model_h5)
        load_model_weights(model, model_h5=model_h5)
    print('model over')
    dataset = load_dataset(data_set_path, batch_size=batch_size)
    print('x_dataset over')
    learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.01,
                                                                   decay_rate=0.96,
                                                                   decay_steps=10000)
    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)

    train_loss_results = []
    train_accuracy_results = []

    epoch_loss_avg = tf.keras.metrics.Mean()
    epoch_accuracy = tf.keras.metrics.BinaryAccuracy()
    for epoch in range(num_epochs):
        # Training loop - using batches of 32
        with tf.GradientTape() as tape:
            train_x_, y_true_label = next(iter(dataset))
            y_pred_all, feature_all = model(train_x_, training=True)
            print('*')
            for train_x_, train_y_ in tqdm.tqdm(dataset):
                y_pred, feature = model(train_x_, training=True)
                feature_all = tf.concat([feature_all, feature], axis=0)
                y_pred_all = tf.concat([y_pred_all, y_pred], axis=0)
                y_true_label = tf.concat([y_true_label, train_y_], axis=0)

            if train_pseudo_label:
                kmeans = KMeans(n_clusters=2).fit(feature_all)
                kmeans_labels = kmeans.labels_
                y_pseudo = tf.keras.utils.to_categorical(kmeans_labels, num_classes=2)
                y_label = y_pseudo
                loss_value = tf.keras.losses.BinaryCrossentropy(from_logits=True)(y_true=y_pseudo, y_pred=y_pred_all)
            else:
                y_label = y_true_label
                loss_value = tf.keras.losses.BinaryCrossentropy(from_logits=True)(y_true=y_label, y_pred=y_pred_all)
        y_true_label = y_true_label.numpy().squeeze()
        NMI = normalized_mutual_info_score(np.argmax(y_true_label, 1), np.argmax(y_pred_all, 1))
        grads1 = tape.gradient(loss_value, model.trainable_variables)
        optimizer.apply_gradients(zip(grads1, model.trainable_variables))
        epoch_loss_avg.update_state(loss_value)  # Add current batch loss
        epoch_accuracy.update_state(y_true_label, y_pred_all)

        # End epoch
        train_loss_results.append(epoch_loss_avg.result())
        train_accuracy_results.append(epoch_accuracy.result())

        print("Epoch {:03d}: Loss: {:.3f}; Accuracy: {:.3f}; NMI_t_t1={}; cluster_n={} grads={}".format(epoch,
                                                                                                   epoch_loss_avg.result(),
                                                                                                   epoch_accuracy.result(),
                                                                                                   NMI,
                                                                                                   kmeans_labels.sum(),
                                                                                                   grads1[0][:1, 0, 0,
                                                                                                   0, 0].numpy()[0]))

    test_x, test_y = pre_data.per_fits_data(path='data/test/')
    test_y_ = tf.keras.utils.to_categorical(test_y, num_classes=2)
    test_x = tf.expand_dims(test_x, -1)
    test_x = tf.cast(test_x, dtype=tf.float32)
    pre_y = model(test_x, training=False)
    pre_y_ = np.argmax(pre_y[0], axis=1)
    print(accuracy_score(test_y, pre_y_))

    feature = pre_y[1].numpy()
    kmeans = KMeans(n_clusters=2, random_state=0).fit(feature)
    print(accuracy_score(test_y, kmeans.labels_))


if __name__ == '__main__':
   pass