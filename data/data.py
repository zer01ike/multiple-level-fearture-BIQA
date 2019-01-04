# _*_ coding:utf-8 _*_
# @Time     :12/27/18 4:49 PM
# @Author   :zer01ike
# @FileName : data.py.py
# @gitHub   : https://github.com/zer01ike

import tensorflow as tf

class data (object):
    def __init__(self) -> None:
        super().__init__()

        pass

    def get_test_dataset(self):
        pass

    def get_train_dataset(self):
        pass

    def __mean_image_subtraction(self,image,means,channel):
        image_channels = tf.split(axis=2, num_or_size_splits=channel, value=image)
        for i in range(channel):
            image_channels[i] -= means[i]
        return tf.concat(axis=2, values=image_channels)

    def tenerateTrainTestDataSet(self,percentage):
        pass

    def save(self,nameList=[]):
        pass



