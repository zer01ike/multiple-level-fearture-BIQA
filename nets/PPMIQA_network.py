# _*_ coding:utf-8 _*_
# @Time     :12/19/18 10:31 AM
# @Author   :zer01ike
# @FileName : PPMIQA_network.py
# @gitHub   : https://github.com/zer01ike

from __future__ import print_function, unicode_literals

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.nets import resnet_v1
from tensorflow.contrib.slim.nets import vgg
from tensorflow.contrib import layers as layers_lib
from tensorflow.contrib.layers.python.layers import layers
from tensorflow.python.ops import math_ops
import numpy as np

class PPMIQA(object):
    def __init__(self,sess):
        self.features_tensor_name = {
            "feature_block_1":"resnet_v1_50/block1/unit_3/bottleneck_v1/Relu:0",
            "feature_block_2":"resnet_v1_50/block2/unit_4/bottleneck_v1/Relu:0",
            "feature_block_3":"resnet_v1_50/block3/unit_6/bottleneck_v1/Relu:0",
            "feature_block_4":"resnet_v1_50/block4/unit_3/bottleneck_v1/Relu:0"
        }
        self.sess = sess
    
    def init(self):
        pass

    def inference(self,image):
        net ,end_points = self.resnet_reload(image)
        # 原网络
        result = self.spatial_pooling()
        # 3.1
        # result = self.spatial_pooling_without_upsample()
        # 3.2
        # result = self.spatial_pooling_with_downsampling()
        # 3.3
        # result = self.spatial_pooling_without_upsmaple_and_block1()
        # 3.4
        # result = self.spatial_pooling_without_upsample_and_block4()
        return result

    def inference_with_VGG19(self,image):
        with slim.arg_scope(vgg.vgg_arg_scope()):
            net ,end_point = vgg.vgg_19(image,is_training=False)

        with tf.variable_scope("multiple_concat"):
            conv3_4 = tf.get_default_graph().get_tensor_by_name("vgg_19/conv3/conv3_4/Relu:0")
            conv5_4 = tf.get_default_graph().get_tensor_by_name("vgg_19/conv5/conv5_4/Relu:0")

            conv3_4_feature = layers_lib.conv2d(conv3_4, 256, [1, 1], stride=1, padding='SAME', scope="conv1",
                                               activation_fn=tf.nn.relu, normalizer_fn=layers.batch_norm,
                                               trainable=True)

            conv5_4_feature = layers_lib.conv2d(conv5_4, 256, [1, 1], stride=1, padding='SAME', scope="conv4",
                                               activation_fn=tf.nn.relu, normalizer_fn=layers.batch_norm,
                                               trainable=True)

            conv5_4_feature_up = layers_lib.conv2d_transpose(conv5_4_feature, 256, [4, 4], stride=4, padding='VALID',
                                                            scope='upsample',  # 28 * 28 * 256
                                                            activation_fn=tf.nn.relu, normalizer_fn=layers.batch_norm,
                                                            trainable=True)

            concat_upsample = tf.concat([conv3_4_feature, conv5_4_feature_up], -1, name='concat_upsample')

            gap_up = math_ops.reduce_mean(concat_upsample,[1, 2], name='gap_up', keepdims=False)

            gap_conv5_4 = math_ops.reduce_mean(conv5_4_feature, [1, 2], name='gap_down', keepdims=False)

            concat = tf.concat([gap_up, gap_conv5_4], -1, name='concat_all')
            fc = layers_lib.fully_connected(concat, 1, activation_fn=tf.nn.sigmoid, scope="multi_FC")

        return fc

    def resnet_reload(self, image):
        with slim.arg_scope(resnet_v1.resnet_arg_scope()):
            net, end_points = resnet_v1.resnet_v1_50(image,is_training=False)
        return net, end_points

    def get_block_feature(self):

        block1_feature = tf.get_default_graph().get_tensor_by_name(
            self.features_tensor_name['feature_block_1'])  # 28 * 28 * 256
        block4_feature = tf.get_default_graph().get_tensor_by_name(
            self.features_tensor_name['feature_block_4'])  # 7 * 7 * 2048
        block4_up_feature = tf.get_default_graph().get_tensor_by_name("multiple_concat/upsample/Relu:0")

        return block1_feature,block4_up_feature
    
    def spatial_pooling(self):
        with tf.variable_scope("multiple_concat"):
            block1_feature = tf.get_default_graph().get_tensor_by_name(self.features_tensor_name['feature_block_1']) # 28 * 28 * 256
            block4_feature = tf.get_default_graph().get_tensor_by_name(self.features_tensor_name['feature_block_4']) # 7 * 7 * 2048

            # block1_feature_numpy = block1_feature.eval(session=self.sess)
            # block4_feature_numpy = block4_feature.eval(session=self.sess)
            # np.save('/home/wangkai/logs_save/feautre1.npy',block1_feature_numpy)
            # np.save('/home/wangkai/logs_save/feautre4.npy',block4_feature_numpy)


            # insert save feature_map function here

            #1*1*256 for every feature map
            block1_feature = layers_lib.conv2d(block1_feature, 256, [1, 1], stride=1, padding='SAME', scope="conv1",
                              activation_fn=tf.nn.relu, normalizer_fn=layers.batch_norm,
                              trainable=True)

            block4_feature = layers_lib.conv2d(block4_feature, 256, [1, 1], stride=1, padding='SAME', scope="conv4",
                                               activation_fn=tf.nn.relu, normalizer_fn=layers.batch_norm,
                                               trainable=True)

            # block1_feature_down = layers_lib.conv2d(block1_feature, 256, [4, 4],stride=4,padding='VALID',scope='downsample', # 7 * 7 * 2048
            #                                         activation_fn=tf.nn.relu,normalizer_fn=layers.batch_norm,
            #                                         trainable=True)
            block4_feature_up = layers_lib.conv2d_transpose(block4_feature, 256, [4, 4], stride=4, padding='VALID', scope='upsample', # 28 * 28 * 256
                                                            activation_fn=tf.nn.relu, normalizer_fn=layers.batch_norm,
                                                            trainable=True)

            # block1 concate with the block4_up
            concat_upsample = tf.concat([block1_feature, block4_feature_up], -1, name='concat_upsample')

            # block4 concate with the block1_down
            # concat_downsample = tf.concat([block4_feature,block1_feature_down],-1,name='concat_downsample')

            #Gap for every concat result

            gap_up = math_ops.reduce_mean(concat_upsample,[1, 2], name='gap_up', keepdims=False)
            #gap_down = math_ops.reduce_mean(concat_downsample, [1, 2], name='gap_down',keepdims=False)
            gap_block4 = math_ops.reduce_mean(block4_feature, [1, 2], name='gap_down', keepdims=False)


            # concat the two features
            # concat = tf.concat([gap_up,gap_down],-1,name='concat_all')
            concat = tf.concat([gap_up, gap_block4],-1,name='concat_all')

            # full connected
            # fc = layers_lib.fully_connected(concat_upsample, 1, activation_fn=tf.nn.sigmoid, scope="multi_FC")
            fc = layers_lib.fully_connected(concat, 1, activation_fn=tf.nn.sigmoid, scope="multi_FC")

        return fc

    def spatial_pooling_without_upsample(self):
        with tf.variable_scope("multiple_concate_without_upsample"):
            block1_feature = tf.get_default_graph().get_tensor_by_name(
                self.features_tensor_name['feature_block_1'])  # 28 * 28 * 256
            block4_feature = tf.get_default_graph().get_tensor_by_name(
                self.features_tensor_name['feature_block_4'])  # 7 * 7 * 2048

            # 1*1*256 for every feature map
            block1_feature = layers_lib.conv2d(block1_feature, 256, [1, 1], stride=1, padding='SAME', scope="conv1",
                                               activation_fn=tf.nn.relu, normalizer_fn=layers.batch_norm,
                                               trainable=True)

            block4_feature = layers_lib.conv2d(block4_feature, 256, [1, 1], stride=1, padding='SAME', scope="conv4",
                                               activation_fn=tf.nn.relu, normalizer_fn=layers.batch_norm,
                                               trainable=True)

            # global average pooling method

            gap_block1 = math_ops.reduce_mean(block1_feature, [1, 2], name='gap_block1', keepdims=False)
            gap_block4 = math_ops.reduce_mean(block4_feature, [1, 2], name='gap_block4', keepdims=False)

            # concat
            concat = tf.concat([gap_block1,gap_block4],-1,name='concat_all')

            # full connected
            fc = layers_lib.fully_connected(concat,1,activation_fn=tf.nn.sigmoid,scope='FC_with_no_upsampling')

        return fc

    def spatial_pooling_with_downsampling(self):
        with tf.variable_scope("multiple_concat"):
            block1_feature = tf.get_default_graph().get_tensor_by_name(self.features_tensor_name['feature_block_1']) # 28 * 28 * 256
            block4_feature = tf.get_default_graph().get_tensor_by_name(self.features_tensor_name['feature_block_4']) # 7 * 7 * 2048

            #1*1*256 for every feature map
            block1_feature = layers_lib.conv2d(block1_feature, 256, [1, 1], stride=1, padding='SAME', scope="conv1",
                              activation_fn=tf.nn.relu, normalizer_fn=layers.batch_norm,
                              trainable=True)

            block4_feature = layers_lib.conv2d(block4_feature, 256, [1, 1], stride=1, padding='SAME', scope="conv4",
                                               activation_fn=tf.nn.relu, normalizer_fn=layers.batch_norm,
                                               trainable=True)

            block1_feature_down = layers_lib.conv2d(block1_feature, 256, [4, 4],stride=4,padding='VALID',scope='downsample', # 7 * 7 * 2048
                                                    activation_fn=tf.nn.relu,normalizer_fn=layers.batch_norm,
                                                    trainable=True)
            block4_feature_up = layers_lib.conv2d_transpose(block4_feature, 256, [4, 4], stride=4, padding='VALID', scope='upsample', # 28 * 28 * 256
                                                            activation_fn=tf.nn.relu, normalizer_fn=layers.batch_norm,
                                                            trainable=True)

            concat_upsample = tf.concat([block1_feature, block4_feature_up], -1, name='concat_upsample')

            # block4 concate with the block1_down
            concat_downsample = tf.concat([block4_feature,block1_feature_down],-1,name='concat_downsample')
            gap_up = math_ops.reduce_mean(concat_upsample, [1, 2], name='gap_up', keepdims=False)
            gap_down = math_ops.reduce_mean(concat_downsample, [1, 2], name='gap_down',keepdims=False)
            concat = tf.concat([gap_up,gap_down],-1,name='concat_all')
            # full connected
            fc = layers_lib.fully_connected(concat, 1, activation_fn=tf.nn.sigmoid, scope='FC_with_no_upsampling')

        return fc
    def spatial_pooling_without_upsmaple_and_block1(self):
        with tf.variable_scope("multiple_concat"):
            block4_feature = tf.get_default_graph().get_tensor_by_name(
                self.features_tensor_name['feature_block_4'])  # 7 * 7 * 2048
            gap_block4 = math_ops.reduce_mean(block4_feature,[1,2],name='only_feature4', keepdims=False)
            fc = layers_lib.fully_connected(gap_block4,1,activation_fn=tf.nn.sigmoid,scope='FC_with_feature4')

        return fc
    def spatial_pooling_without_upsample_and_block4(self):
        with tf.variable_scope("multiple_concat"):
            block1_feature =  tf.get_default_graph().get_tensor_by_name(
                self.features_tensor_name['feature_block_1'] # 28 * 28 * 256
            )
            gap_block1 = math_ops.reduce_mean(block1_feature,[1,2],name='only_feature1',keepdims=False)
            fc = layers_lib.fully_connected(gap_block1,1,activation_fn=tf.nn.sigmoid,scope='FC_with_feature1')

        return fc


    def get_resent50_var(self):
        result_var = []
        global_var = tf.global_variables()
        for var in global_var:
            # print(var.name)
            if 'resnet_v1_50' in var.name and 'Momentum' not in var.name:
                result_var.append(var)
        return result_var
    def get_vgg19_var(self):
        result_var = []
        global_var = tf.global_variables()
        for var in global_var:
            if 'vgg_19' in var.name and 'Momentum' not in var.name:
                result_var.append(var)
        return result_var

if __name__ == '__main__':
    # ppm = PPMIQA()
    #
    # feautre_block1 = tf.get_default_graph().get_tensor_by_name(ppm.features_tensor_name['feature_block_1'])
    #
    # print(feautre_block1.shape)

    pass

    
    