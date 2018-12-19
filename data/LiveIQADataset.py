from __future__ import print_function, unicode_literals
import os

import tensorflow as tf
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


class LiveIQADataset(object):
    def __init__(self, mode, batch_size=1, shuffle=True, crop_size=50, num_epochs=10, crop_shape=[224, 224, 3]):
        # # check read mode
        # if mode == 'training':
        #     self.path_to_db = '/home/wangkai/Paper_MultiFeature_Data/databaserelease2/train.tfrecord'
        #     self.train = True
        # elif mode == 'test':
        #     self.path_to_db = '/home/wangkai/Paper_MultiFeature_Data/databaserelease2/test.tfrecord'
        #     self.train = False
        # else:
        #     assert 0, "Unknown dataset mode."

        self.path_to_train_db = '/home/wangkai/Paper_MultiFeature_Data/databaserelease2/train.tfrecord'
        self.path_to_test_db = '/home/wangkai/Paper_MultiFeature_Data/databaserelease2/test.tfrecord'

        self.image_dir = ' '
        self.iqa_dir = '/home/wangkai/Paper_MultiFeature_Data/databaserelease2/'

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.crop_size = crop_size
        self.crop_shape = crop_shape
        self.num_epochs = num_epochs
        self.means = [103.94, 116.78, 123.68]

    def example(self):
        dataset = tf.data.TFRecordDataset([self.path_to_db])
        reader = tf.TFRecordReader()
        _, value = reader.read(tf.train.string_input_producer([self.path_to_db], num_epochs=self.num_epochs,
                                                              shuffle=self.shuffle))
        # _, value = reader.read(tf.train.string_input_producer([self.path_to_db]))
        features = self.decode_tfrecord(value=value)
        # return features
        dmos, img = self.preprocessing(features)

        #
        return dmos, img

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

    def get_test_dataset(self):
        dataset = tf.data.TFRecordDataset([self.path_to_test_db])
        dataset = dataset.map(self.decode_tfrecord)
        dataset = dataset.map(self.preprocessing)
        dataset = dataset.batch(self.batch_size)
        return dataset

    def get_train_dataset(self):
        dataset = tf.data.TFRecordDataset([self.path_to_train_db])
        dataset = dataset.map(self.decode_tfrecord)
        dataset = dataset.map(self.preprocessing)
        dataset = dataset.batch(self.batch_size).repeat(self.num_epochs)
        # dataset = dataset.repeat(self.num_epochs)

        return dataset

    def get(self):
        # my_example = self.example()
        # TODO:
        demos, img = self.example()

        min_after_dequeue = 10000
        capacity = min_after_dequeue + 3 * self.batch_size
        demos, img = tf.train.shuffle_batch(
            [demos, img], batch_size=self.batch_size, capacity=capacity,
            min_after_dequeue=min_after_dequeue
        )
        return demos, img

    def _mean_image_subtraction(self, image, means, channel):

        image_channels = tf.split(axis=2, num_or_size_splits=channel, value=image)
        for i in range(channel):
            image_channels[i] -= means[i]
        return tf.concat(axis=2, values=image_channels)

    def _prase_file(self):
        self.imagesinfo = []
        parentdir = self.iqa_dir
        IQA_name = "LIVE_IQA.txt"
        with open(parentdir + IQA_name) as pathfile:
            lines_pathfile = pathfile.readlines()

            # this is the pattern of the file in IQA_name.txt
            for eachline in lines_pathfile:
                single_imagepath = eachline.split(" ")[3]
                single_ref = eachline.split(" ")[2].split("/")[1].split('.')[0]
                single_distoration = single_imagepath.split("/")[0]
                single_name = single_imagepath.split("/")[1].split('.')[0]
                single_dmos_normalize = eachline.split(" ")[4]

                single_image = [parentdir + single_imagepath, single_ref, single_distoration, single_name,
                                single_dmos_normalize]
                self.imagesinfo.append(single_image)

    def generateTrainTestDataSet(self, percentage=[8, 2]):
        self._prase_file()
        if sum(percentage) != 10: assert 0, 'error percentage for generate Train and test'

        train_name_list = []
        test_name_list = []

        ref_list = [v[1] for v in self.imagesinfo]
        ref_set = set(ref_list)
        ref_list = list(ref_set)

        train_ref_list = ref_list[0:int(percentage[0] * len(ref_list) * 0.1)]
        test_ref_list = ref_list[int(percentage[0] * len(ref_list) * 0.1):]

        for image in self.imagesinfo:
            if image[1] in test_ref_list:
                test_name_list.append([image[0], image[4]])
            else:
                train_name_list.append([image[0], image[4]])

        self.save(mode='tfRecord', name=self.iqa_dir + 'train.tfrecord', nameList=train_name_list)
        self.save(mode='tfRecord', name=self.iqa_dir + 'test.tfrecord', nameList=test_name_list)

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
            image = Image.open(im_dir)
            width, height = image.size
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


if __name__ == '__main__':
    data = LiveIQADataset('training', batch_size=2, shuffle=False, crop_size=50, num_epochs=10)
    data.test_tfdata()
    # # feature = data.example()
    # demos,img = data.get()
    # #image_batch = data.get()
    # init_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
    # with tf.Session() as sess:
    #     #sess.as_default()
    #     sess.run(init_op)
    #     coord = tf.train.Coordinator()
    #     threads = tf.train.start_queue_runners(coord=coord)
    #     print('Threads:%s' % threads)
    #     try:
    #         for i in range(1):
    #             if not coord.should_stop():
    #                 demos_v, img_v = sess.run([demos, img])
    #                 print(demos_v)
    #                 print(img_v.shape)
    #                 # plt.figure()
    #                 # plt.imshow(Image.fromarray(img_v,'RGB'))
    #                 # plt.show()
    #     except tf.errors.OutOfRangeError:
    #         print('Catch OutRangeError')
    #     finally:
    #         coord.request_stop()
    #         print('Finish reading')
    #
    #
    #     #feature_v = sess.run([feature])
    #
    #     coord.join(threads)
