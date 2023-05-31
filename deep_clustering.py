import tensorflow as tf
from tensorflow.keras import layers, Sequential
import numpy as np
import os
from sklearn.cluster import KMeans
from model_ulti.get_tfrecord import tf_serialize_example
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


def load_dataset_all(data_set_path_list, batch_size=3000):
    file_list_all = []
    file_label_all = []
    for data_set_path in data_set_path_list:
        temp = os.listdir(data_set_path)
        file_list1 = [os.path.join(data_set_path, item) for item in temp]
        file_label1 = [item.split('_')[0] for item in temp]

        file_list_all = file_list_all + file_list1
        file_label_all = file_label_all + file_label1

    samples_num = len(file_list_all)
    file_label_all = np.array(file_label_all, np.int32)
    features = tf.constant(file_list_all, tf.string, shape=(samples_num, 1))  # ==> 3x2 tensor
    labels = tf.constant(file_label_all, shape=samples_num)  # ==> 3x1 tensor

    features_dataset = tf.data.Dataset.from_tensor_slices(features)
    labels_dataset = tf.data.Dataset.from_tensor_slices(labels)
    dataset = tf.data.Dataset.zip((features_dataset, labels_dataset))
    dataset = dataset.shuffle(buffer_size=samples_num)
    dataset = dataset.map(tf_serialize_example)
    dataset = dataset.batch(batch_size=batch_size)

    return dataset, samples_num


if __name__ == '__main__':
   pass