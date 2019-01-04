# _*_ coding:utf-8 _*_
# @Time     :12/24/18 11:45 AM
# @Author   :zer01ike
# @FileName : LiveCHIQADataset.py
# @gitHub   : https://github.com/zer01ike

import scipy.io as scio
from typing import List
import numpy as np
from PIL import Image
import tensorflow as tf

class SynIQADataset(object):
    def __init__(self, batch_size=1, shuffle=True, crop_size=50, num_epochs=10, crop_shape=[224,224,3]):
        self.path_to_mos = '/home/wangkai/Paper_MultiFeature_Data/syn_data/synthesized_image_score1.txt'
        self.path_to_image = '/home/wangkai/Paper_MultiFeature_Data/syn_data/images/'
        self.path_to_train_db = '/home/wangkai/Paper_MultiFeature_Data/syn_data/train_normalized.tfrecord'
        self.path_to_test_db = '/home/wangkai/Paper_MultiFeature_Data/syn_data/test_normalized.tfrecord'
        self.path_to_single_db = '/home/wangkai/Paper_MultiFeature_Data/syn_data/single_normalized.tfrecord'

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

        img = self._mean_image_subtraction(img,self.means)

        return dmos, img
    def preprocessing_mean_sub(self,features):
        dmos = tf.cast(features['dmos'], tf.float32)
        dmos = tf.reshape(dmos, [1])
        # hight = tf.cast(features['height'], tf.int64)
        # width = tf.cast(features['width'], tf.int64)
        # channel = tf.cast(features['channel'], tf.int64)
        img = tf.decode_raw(features['image_raw'], tf.uint8)
        img = tf.reshape(img, self.crop_shape)

        img = tf.to_float(img)

        img = self._mean_image_subtraction(img, self.means)
        return dmos, img

    def _mean_image_subtraction(self, img, means):

        image_channels = tf.split(axis=2, num_or_size_splits=3, value=img)
        for i in range(3):
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

    def get_single_dataset(self, image_name, demos):
        self.generateSingleDataset(image_name, demos)
        dataset = tf.data.TFRecordDataset([self.path_to_single_db])
        dataset = dataset.map(self.decode_tfrecord)
        dataset = dataset.map(self.preprocessing_mean_sub)
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

        imageList =[]

        with open(self.path_to_mos) as pathfile:
            lines_pathfile = pathfile.readlines()

            for eachline in lines_pathfile:
                eachline = eachline.strip("\n")
                image_name = eachline.split(",")[0]
                image_demos = eachline.split(",")[1]

                imageList.append([image_name, image_demos])

        return imageList

    def generateTrainTestDataSet(self, percentage=[8, 2]):
        imageList = self._prase_file()

        train_end = int(percentage[0] * (len(imageList)) * 0.1)

        train_list = imageList[0:train_end]
        test_list = imageList[train_end:]

        self.save(mode='tfRecord', name=self.path_to_train_db, nameList=train_list)
        self.save(mode='tfRecord', name=self.path_to_test_db, nameList=test_list)

    def generateSingleDataset(self, file_name, demos):
        # read image
        image = Image.open(self.path_to_image + file_name)
        width, height = image.size


        # crop image with stride
        crop_images = self._crop_pil_stride(image, 64, 224, 224)

        self.save_single(self.path_to_single_db,crop_images,demos)
        # save_dir = '/home/wangkai/Paper_MultiFeature_Data/syn_data/test_image_crop/'
        # count = 0
        # for i in crop_images:
        #     i.save(save_dir+str(count)+'.bmp')
        #     count += 1


    def _crop_pil_stride(self,image,stride,crop_height,crop_width):
        width, height = image.size
        images = []
        for offset_width in range(0,width,stride):
            for offset_height in range(0,height,stride):
                images_crop = self._crop_pil(image,offset_height,offset_width,crop_height,crop_width)
                images.append(images_crop)

        return images

    def _crop_pil(self,image,offset_height,offset_width,crop_height,crop_width):
        width,height = image.size

        if offset_height > height or offset_width > width:
            assert "Error with offset"

        if offset_width+crop_width > width :
            offset_width = width - crop_width
        if offset_height+crop_height > height:
            offset_height = height - crop_height

        image_croped = image.crop((offset_width,offset_height,offset_width+crop_width,offset_height+crop_height))

        return image_croped



    def _crop_tf(self, image, offset_height, offset_width, crop_height, crop_width):
        """Crops the given image using the provided offsets and sizes.
        Note that the method doesn't assume we know the input image size but it does
        assume we know the input image rank.
        Args:
          image: an image of shape [height, width, channels].
          offset_height: a scalar tensor indicating the height offset.
          offset_width: a scalar tensor indicating the width offset.
          crop_height: the height of the cropped image.
          crop_width: the width of the cropped image.
        Returns:
          the cropped (and resized) image.
        Raises:
          InvalidArgumentError: if the rank is not 3 or if the image dimensions are
            less than the crop size.
        """
        original_shape = tf.shape(image)

        rank_assertion = tf.Assert(
            tf.equal(tf.rank(image), 3),
            ['Rank of image must be equal to 3.'])
        with tf.control_dependencies([rank_assertion]):
            cropped_shape = tf.stack([crop_height, crop_width, original_shape[2]])

        size_assertion = tf.Assert(
            tf.logical_and(
                tf.greater_equal(original_shape[0], crop_height),
                tf.greater_equal(original_shape[1], crop_width)),
            ['Crop size greater than the image size.'])

        offsets = tf.to_int32(tf.stack([offset_height, offset_width, 0]))

        # Use tf.slice instead of crop_to_bounding box as it accepts tensors to
        # define the crop size.
        with tf.control_dependencies([size_assertion]):
            image = tf.slice(image, offsets, cropped_shape)
        return tf.reshape(image, cropped_shape)


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
                'dmos': _float_feature(float(demos)/5),
                'image_raw': _bytes_feature(image_raw)
            }

            tf_example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(tf_example.SerializeToString())
            print("write:".ljust(10) + str(count) + "Down!".rjust(4))
        writer.close()

    def save_single(self,name, Images, demos):
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
        print("Writing TFrecord to File %s."%self.path_to_single_db)
        for image in Images:
            width, height = image.size
            image = image.convert('RGB')
            count += 1
            image = np.array(image)
            image_raw = Image.fromarray(image).tobytes()

            feature = {
                'height': _int64_feature(height),
                'width': _int64_feature(width),
                'channel': _int64_feature(3),
                'dmos': _float_feature(float(demos) / 5),
                'image_raw': _bytes_feature(image_raw)
            }

            tf_example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(tf_example.SerializeToString())
            #print("write:".ljust(10) + str(count) + "Down!".rjust(4))
        writer.close()



if __name__ == '__main__':
    dataset = SynIQADataset(25)
    #dataset.generateTrainTestDataSet()
    file_name ='01_72_04_Book_arrival_A6_8_to_9_70.bmp'
    demos = 3.666666667
    dataset.generateSingleDataset(file_name,demos)
