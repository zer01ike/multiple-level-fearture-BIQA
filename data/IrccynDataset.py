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
import pandas as pd

class IrccynDataset(object):
    def __init__(self,batch_size, shuffle=True, crop_size=50, num_epochs=10, crop_shape=[224,224,3]):
        self.path_to_mos = '/home/wangkai/Paper_MultiFeature_Data/IRCCyN_IVC_DIBR_dataset/IRCCyN_IVC_DIBR_Images_Database_ACR_Score.xlsx'
        self.path_to_images = '/home/wangkai/Paper_MultiFeature_Data/IRCCyN_IVC_DIBR_dataset/Images/'
        self.path_to_train_db = '/home/wangkai/Paper_MultiFeature_Data/IRCCyN_IVC_DIBR_dataset/train_normalized.tfrecord'
        self.path_to_test_db = '/home/wangkai/Paper_MultiFeature_Data/IRCCyN_IVC_DIBR_dataset/test_normalized.tfrecord'
        self.path_to_single_db = '/home/wangkai/Paper_MultiFeature_Data/IRCCyN_IVC_DIBR_dataset/single_normalized.tfrecord'

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.crop_size = crop_size
        self.crop_shape = crop_shape
        self.num_epochs = num_epochs
        self.means = [103.94, 116.78, 123.68]

    def _prase_file(self):
        # imageList = []
        #
        # with open(self.path_to_mosandnames) as pathfile:
        #     lines_pathfile = pathfile.readlines()
        #
        #     for eachline in lines_pathfile:
        #         eachline = eachline.strip("\n")
        #         demos,image = eachline.split(" ")
        #         distort_type = image.split("_")[1]
        #         imageList.append([image,demos,distort_type])
        #print(imageList)
        dmos_file = pd.read_excel(self.path_to_mos)
        dmos_file = dmos_file.dropna(axis=0,how='all')
        imageList = []
        #print(dmos_file)
        index =0
        for image, dmos in zip(dmos_file['image name'].values, dmos_file['Unnamed: 45'].values):
            if index == 0 :
                index +=1
                continue
            else:
                imageList.append([image,dmos])
                #index += 1

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

    def get_test_list(self):
        imageList = self._prase_file()

        train_end = int(8 * (len(imageList)) * 0.1)
        test_list = imageList[train_end:]

        return test_list

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
        #print(imagetype)

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

    def generateSingleDataset(self,image_name,demos):
        image = Image.open(self.path_to_images + image_name)
        width, height = image.size

        # crop image with stride
        crop_images = self._crop_pil_stride(image, 64, 224, 224)

        self.save_single(self.path_to_single_db, crop_images, demos)


    def preprocessing(self, features):
        dmos = tf.cast(features['dmos'], tf.float32)
        dmos = tf.reshape(dmos, [1])
        hight = tf.cast(features['height'], tf.int32)
        width = tf.cast(features['width'], tf.int32)
        channel = tf.cast(features['channel'], tf.int32)
        img = tf.decode_raw(features['image_raw'], tf.uint8)
        img = tf.reshape(img, [hight,width,channel])

        # pass
        img = tf.random_crop(img, self.crop_shape)
        img = tf.to_float(img)

        img = self._mean_image_subtraction(img, self.means, 3)

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

        img = self._mean_image_subtraction(img, self.means,3)
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

    def get_single_dataset(self,image_name,demos,type):
        self.generateSingleDataset(image_name,demos)

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









if __name__ == '__main__':
    dataset = IrccynDataset(25)
    dataset._prase_file()
    #dataset.generateTrainTestDataSet()

