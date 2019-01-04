# _*_ coding:utf-8 _*_
# @Time     :12/27/18 7:12 PM
# @Author   :zer01ike
# @FileName : TID2013Dataset.py
# @gitHub   : https://github.com/zer01ike

import scipy.io as scio
import numpy as np
from PIL import Image
import tensorflow as tf
import random

class TID2013Dataset(object):
    def __init__(self,batch_size, shuffle=True, crop_size=50, num_epochs=10, crop_shape=[224,224,3]):
        self.path_to_mosandnames = '/home/wangkai/Paper_MultiFeature_Data/tid2013/mos_with_names.txt'
        self.path_to_images = '/home/wangkai/Paper_MultiFeature_Data/tid2013/distorted_images/'
        self.path_to_train_db = '/home/wangkai/Paper_MultiFeature_Data/tid2013/train_normalized.tfrecord'
        self.path_to_test_db = '/home/wangkai/Paper_MultiFeature_Data/tid2013/test_normalized.tfrecord'

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.crop_size = crop_size
        self.crop_shape = crop_shape
        self.num_epochs = num_epochs
        self.means = [103.94, 116.78, 123.68]

    def _prase_file(self):
        imageList = []

        with open(self.path_to_mosandnames) as pathfile:
            lines_pathfile = pathfile.readlines()

            for eachline in lines_pathfile:
                eachline = eachline.strip("\n")
                demos,image = eachline.split(" ")
                distort_type = image.split("_")[1]
                imageList.append([image,demos,distort_type])
        #print(imageList)

        return imageList

    def save(self,name=None,nameList=[]):
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

        for im_dir,demos,distort_type in nameList:
            image = Image.open(self.path_to_images+im_dir)
            width ,height = image.size
            count+=1
            image = np.array(image)

            image_raw = Image.fromarray(image).tobytes()

            feature = {
                'height': _int64_feature(height),
                'width': _int64_feature(width),
                'channel': _int64_feature(3),
                'dmos': _float_feature(float(demos)/10),
                'type': _int64_feature(int(distort_type)),
                'image_raw': _bytes_feature(image_raw)
            }

            tf_example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(tf_example.SerializeToString())
            print("write:".ljust(10) + str(count) + "Down!".rjust(4))
        writer.close()

    def generateTrainTestDataSet(self,percentage=[8,2]):
        imageList = self._prase_file()
        imagetype = [i for i in range(1, 26)]
        # print(imagetype)
        random.shuffle(imagetype)

        # print(imagetype)

        train_end = int(percentage[0]*(len(imagetype))*0.1)

        for i in range(0,len(imagetype)):
            if imagetype[i] < 10:
                imagetype[i] = 'i0'+str(imagetype[i])
            else:
                imagetype[i] = 'i' + str(imagetype[i])
        print(imagetype)

        train_ref_list = imagetype[:train_end]
        test__ref_list = imagetype[train_end:]

        train_list = []
        test_list = []

        for i in range(0,len(imageList)):
            image_type = imageList[i][0].split("_")[0]
            # print(image_type)

            if image_type in train_ref_list:
                train_list.append(imageList[i])
            else:
                test_list.append(imageList[i])

            # print(train_list)

        self.save(name=self.path_to_train_db,nameList=train_list)
        self.save(name = self.path_to_test_db,nameList=test_list)

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
                                               'type': tf.FixedLenFeature([],tf.int64),
                                               'image_raw': tf.FixedLenFeature([], tf.string)
                                           })
        return features










if __name__ == '__main__':
    dataset = TID2013Dataset(25)
    dataset.generateTrainTestDataSet()

