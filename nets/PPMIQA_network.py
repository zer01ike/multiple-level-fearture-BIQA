# _*_ coding:utf-8 _*_
# @Time     :12/19/18 10:31 AM
# @Author   :zer01ike
# @FileName : PPMIQA_network.py
# @gitHub   : https://github.com/zer01ike

from __future__ import print_function, unicode_literals

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.nets import resnet_v1
from tensorflow.contrib import layers as layers_lib
from tensorflow.contrib.layers.python.layers import layers
from tensorflow.python.ops import math_ops

class PPMIQA(object):
    def __init__(self):
        self.features_tensor_name = {
            "feature_block_1":"resnet_v1_50/block1/unit_3/bottleneck_v1/Relu:0",
            "feature_block_2":"resnet_v1_50/block2/unit_4/bottleneck_v1/Relu:0",
            "feature_block_3":"resnet_v1_50/block3/unit_6/bottleneck_v1/Relu:0",
            "feature_block_4":"resnet_v1_50/block4/unit_3/bottleneck_v1/Relu:0"
        }
    
    def init(self):
        pass

    def inference(self,image):
        net ,end_points = self.resnet_reload(image)
        result = self.spatial_pooling()
        return result
    
    def resnet_reload(self, image):
        with slim.arg_scope(resnet_v1.resnet_arg_scope()):
            net, end_points = resnet_v1.resnet_v1_50(image,is_training=False)
        return net, end_points
    
    def spatial_pooling(self):
        with tf.variable_scope("multiple_concat"):
            block1_feature = tf.get_default_graph().get_tensor_by_name(self.features_tensor_name['feature_block_1'])
            block4_feature = tf.get_default_graph().get_tensor_by_name(self.features_tensor_name['feature_block_4'])

            # 1*1*256 for every feature map
            block1_feature = layers_lib.conv2d(block1_feature, 256, [1, 1], stride=1, padding='SAME', scope="conv1",
                              activation_fn=tf.nn.relu, normalizer_fn=layers.batch_norm,
                              trainable=True)

            block4_feature = layers_lib.conv2d(block4_feature, 256, [1, 1], stride=1, padding='SAME', scope="conv4",
                                               activation_fn=tf.nn.relu, normalizer_fn=layers.batch_norm,
                                               trainable=True)

            block1_feature_down = layers_lib.conv2d(block1_feature,256,[4, 4],stride=4,padding='VALID',scope='downsample',
                                                    activation_fn=tf.nn.relu,normalizer_fn=layers.batch_norm,
                                                    trainable=True)
            block4_feature_up = layers_lib.conv2d_transpose(block4_feature,256,[4, 4],stride=4,padding='VALID',scope='upsample',
                                                            activation_fn=tf.nn.relu,normalizer_fn=layers.batch_norm,
                                                            trainable=True)

            # block1 concate with the block4_up
            concat_upsample = tf.concat([block1_feature, block4_feature_up], -1, name='concat_upsample')

            # block4 concate with the block1_down
            concat_downsample = tf.concat([block4_feature,block1_feature_down],-1,name='concat_downsample')

            #Gap for every concat result

            gap_up = math_ops.reduce_mean(concat_upsample, [1, 2], name='gap_up', keepdims=False)
            gap_down = math_ops.reduce_mean(concat_downsample,[1,2],name='gap_down',keep_dims=False)

            # concat the two features
            concat = tf.concat([gap_up,gap_down],-1,name='concat_all')

            # full connected
            fc = layers_lib.fully_connected(concat, 1, activation_fn=tf.nn.relu, scope="multi_FC")
        return fc

    def get_resent50_var(self):
        result_var = []
        global_var = tf.global_variables()
        for var in global_var:
            # print(var.name)
            if 'resnet_v1_50' in var.name and 'Momentum' not in var.name:
                result_var.append(var)
        return result_var

if __name__ == '__main__':
    # ppm = PPMIQA()
    #
    # feautre_block1 = tf.get_default_graph().get_tensor_by_name(ppm.features_tensor_name['feature_block_1'])
    #
    # print(feautre_block1.shape)

    pass

    
    