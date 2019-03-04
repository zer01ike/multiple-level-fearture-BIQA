import numpy as np
from PIL import Image
import tensorflow as tf
import pandas as pd


class IETRDataset(object):
    def __init__(self, batch_size=1, shuffle=True, crop_size=50, num_epochs=10, crop_shape=[224,224,3]):
        self.path_to_mos = '/home/wangkai/Paper_MultiFeature_Data/IETR_DIBR_Database/DMOS_variance.xlsx'
        self.path_to_image = '/home/wangkai/Paper_MultiFeature_Data/IETR_DIBR_Database/IETR_DIBR_database_PNG/'
        self.path_to_train_db = '/home/wangkai/Paper_MultiFeature_Data/IETR_DIBR_Database/train_normalized.tfrecord'
        self.path_to_test_db = '/home/wangkai/Paper_MultiFeature_Data/IETR_DIBR_Database/test_normalized.tfrecord'
        self.path_to_single_db = '/home/wangkai/Paper_MultiFeature_Data/IETR_DIBR_Database/single_normalized.tfrecord'

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.crop_size = crop_size
        self.crop_shape = crop_shape
        self.num_epochs = num_epochs
        self.means = [103.94, 116.78, 123.68]

    def _prase_file(self):
        # prase_file to get the info of the file

        dmos_file = pd.read_excel(self.path_to_mos)
        imageList = []
        #print(dmos_file)
        for image,dmos in zip(dmos_file['Image'].values,dmos_file['DMOS'].values):
            type = image.split('_')[-1].split('.')[0]
            imageList.append([image,dmos,type])

        #print(imageList)
        return imageList

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
                'dmos': _float_feature(float(demos)),
                'image_raw': _bytes_feature(image_raw)
            }

            tf_example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(tf_example.SerializeToString())
            #print("write:".ljust(10) + str(count) + "Down!".rjust(4))
        writer.close()

    def get_single_dataset(self, image_name, demos,type):
        self.generateSingleDataset(image_name, demos)
        dataset = tf.data.TFRecordDataset([self.path_to_single_db])
        dataset = dataset.map(self.decode_tfrecord)
        dataset = dataset.map(self.preprocessing_mean_sub)
        dataset = dataset.batch(self.batch_size)

        return dataset

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
    dataset = IETRDataset(234)
    dataset._prase_file()