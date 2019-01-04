# _*_ coding:utf-8 _*_
# @Time     :12/24/18 11:45 AM
# @Author   :zer01ike
# @FileName : LiveCHIQADataset.py
# @gitHub   : https://github.com/zer01ike

import scipy.io as scio
import numpy as np
from PIL import Image
import tensorflow as tf

class LiveCHIQADataset(object):
    def __init__(self, batch_size, shuffle=True, crop_size=50, num_epochs=10, crop_shape=[224,224,3]):
        self.path_to_im_mat = '/home/wangkai/Paper_MultiFeature_Data/ChallengeDB_release/ChallengeDB_release/Data/AllImages_release.mat'
        self.path_to_mos_mat = '/home/wangkai/Paper_MultiFeature_Data/ChallengeDB_release/ChallengeDB_release/Data/AllMOS_release.mat'
        self.path_to_mosstddev_mat = '/home/wangkai/Paper_MultiFeature_Data/ChallengeDB_release/ChallengeDB_release/Data/AllStdDev_release.mat'
        self.path_to_image = '/home/wangkai/Paper_MultiFeature_Data/ChallengeDB_release/ChallengeDB_release/Images/'
        self.path_to_train_db = '/home/wangkai/Paper_MultiFeature_Data/ChallengeDB_release/ChallengeDB_release/train_normalized.tfrecord'
        self.path_to_test_db = '/home/wangkai/Paper_MultiFeature_Data/ChallengeDB_release/ChallengeDB_release/test_normalized.tfrecord'

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.crop_size = crop_size
        self.crop_shape = crop_shape
        self.num_epochs = num_epochs
        self.means = [103.94, 116.78, 123.68]

    def preprocessing(self, features):
        dmos = tf.cast(features['dmos'], tf.float32)
        dmos = tf.reshape(dmos, [1])
        hight = tf.cast(features['height'], tf.int64)
        width = tf.cast(features['width'], tf.int64)
        channel = tf.cast(features['channel'], tf.int64)
        img = tf.decode_raw(features['image_raw'], tf.uint8)
        img = tf.reshape(img, [hight, width, channel])

        # pass
        img = tf.random_crop(img, self.crop_shape)
        img = tf.to_float(img)

        img = self._mean_image_subtraction(img, self.means, 3)

        return dmos, img

    def _mean_image_subtraction(self, image, means, channel):

        image_channels = tf.split(axis=2, num_or_size_splits=channel, value=image)
        for i in range(channel):
            image_channels[i] -= means[i]
        return tf.concat(axis=2, values=image_channels)

    def get_train_dataset(self):
        dataset = tf.data.TFRecordDataset([self.path_to_train_db])
        dataset = dataset.map(self.decode_tfrecord)
        dataset = dataset.map(self.preprocessing)
        dataset = dataset.batch(self.batch_size).repeat(self.num_epochs)

        return dataset
    def get_test_dataset(self):
        dataset = tf.data.TFRecordDataset([self.path_to_train_db])
        dataset = dataset.map(self.decode_tfrecord)
        dataset = dataset.map(self.preprocessing)
        dataset = dataset.batch(self.batch_size)

        return dataset


    def decode_tfrecord(self, value):
        features = tf.parse_single_example(value,
                                           features={
                                               'height': tf.FixedLenFeature([], tf.int64),
                                               'width': tf.FixedLenFeature([], tf.int64),
                                               'channel': tf.FixedLenFeature([], tf.int64),
                                               'dmos': tf.FixedLenFeature([], tf.float32),
                                               'image_raw': tf.FixedLenFeature([], tf.string)
                                           })
        return features

    def _prase_file(self):
        # prase_file to get the info of the file

        image_name = []
        image_dmos = []
        imageList =[]

        image_mat = scio.loadmat(self.path_to_im_mat)
        image_mat = image_mat['AllImages_release']

        for i in image_mat:
            #print(i[0][0])
            image_name.append(i[0][0])

        mos_mat = scio.loadmat(self.path_to_mos_mat)
        mos_mat = mos_mat['AllMOS_release'][0]
        mos_max = np.max(mos_mat)
        mos_min = np.min(mos_mat)

        for j in mos_mat:
            #image_dmos.append((j-mos_min)/(mos_max-mos_min))
            image_dmos.append(j/100)
        #print(image_dmos)

        for i,j in zip(image_name,image_dmos):
            imageList.append([i,j])


        # std_dev = scio.loadmat(self.path_to_mosstddev_mat)
        # for m in std_dev['AllStdDev_release'][0]:
        #     print(m)

        return imageList

    def generateTrainTestDataSet(self, percentage=[8, 2]):
        imageList = self._prase_file()

        train_end = int(percentage[0] * (len(imageList)-7) * 0.1)

        train_list = imageList[7:train_end]
        test_list = imageList[train_end:]

        self.save(mode='tfRecord', name=self.path_to_train_db, nameList=train_list)
        self.save(mode='tfRecord', name=self.path_to_test_db, nameList=test_list)


    def save(self, mode='tfRecord', name=None, nameList=[]):
        def _bytes_feature(value):
            """Returns a bytes_list from a string / byte."""
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

        def _float_feature(value):
            """Returns a float_list from a float / double."""
            return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

        def _int64_feature(value):
            """Returns an int64_list from a bool / enum / int / uint."""
            return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

        writer = tf.python_io.TFRecordWriter(name)
        count = 0
        for im_dir, demos in nameList:
            image = Image.open(self.path_to_image+im_dir)
            width, height = image.size
            image = image.convert('RGB')

            count += 1
            image = np.array(image)
            image_raw = Image.fromarray(image).tobytes()

            feature = {
                'height': _int64_feature(height),
                'width': _int64_feature(width),
                'channel': _int64_feature(3),
                'dmos': _float_feature(float(demos)),
                'image_raw': _bytes_feature(image_raw)
            }

            tf_example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(tf_example.SerializeToString())
            print("write:".ljust(10) + str(count) + "Down!".rjust(4))
        writer.close()



if __name__ == '__main__':
    dataset = LiveCHIQADataset(25)
    dataset.generateTrainTestDataSet()
