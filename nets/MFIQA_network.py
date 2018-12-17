from __future__ import print_function,unicode_literals

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.nets import resnet_v1
from tensorflow.contrib import layers as layers_lib
from tensorflow.contrib.layers.python.layers import layers
from tensorflow.python.ops import math_ops

class MFIQA_network(object):

    def __init__(self):
        self.encoder_paramater = {
        "encoder1":"resnet_v1_50/block1/unit_3/bottleneck_v1/Relu:0",
        "encoder2":"resnet_v1_50/block2/unit_4/bottleneck_v1/Relu:0",
        "encoder3":"resnet_v1_50/block3/unit_6/bottleneck_v1/Relu:0", 
        "encoder4":"resnet_v1_50/block4/unit_3/bottleneck_v1/Relu:0"}
        

    
    def init(self):
        current_epoch = tf.Variable(0,name="current_epoch")

        return current_epoch
        
    
    def inference(self,image):
        net,end_points = self.resnet_reload(image)
        tensor_out = self.finetune_layers()
        return tensor_out

    def finetune_layers(self,scope_name="encoder"):
        with tf.variable_scope(scope_name):
            encoder1 = self.encoder(self.encoder_paramater["encoder1"],"encoder1")
            encoder2 = self.encoder(self.encoder_paramater["encoder2"],"encoder2")
            encoder3 = self.encoder(self.encoder_paramater["encoder3"],"encoder3")
            encoder4 = self.encoder(self.encoder_paramater["encoder4"],"encoder4")

            concat = tf.concat([encoder1, encoder2, encoder3, encoder4], -1, name='concat')
            
            tensor_out = layers_lib.fully_connected(concat, 1, activation_fn=tf.nn.relu, scope="fintune_FC")
            return tensor_out

    def resnet_reload(self,image):
        with slim.arg_scope(resnet_v1.resnet_arg_scope()):
            net, end_points = resnet_v1.resnet_v1_50(image,is_training=False)
        return net,end_points

    def encoder(self,tensor_name,layer_name):
        with tf.variable_scope(layer_name):
            encoder_tensor = tf.get_default_graph().get_tensor_by_name(tensor_name)
            #tf.summary.histogram(layer_name+'resnet_out',encoder_tensor)
            encoder_tensor = layers_lib.conv2d(encoder_tensor, 256, [1, 1], stride=2, padding='SAME', scope="conv1",
                                               activation_fn=tf.nn.relu, normalizer_fn=layers.batch_norm,trainable=True)
            encoder_tensor = layers_lib.conv2d(encoder_tensor, 256, [3, 3], stride=2, padding='SAME', scope="conv3",
                                               activation_fn=tf.nn.relu, normalizer_fn=layers.batch_norm,trainable=True)
            out_tensor = math_ops.reduce_mean(encoder_tensor, [1, 2], name='gap', keepdims=False)

            #tf.summary.histogram(layer_name,out_tensor)

            #old style
            #out_tensor = tf.reduce_mean(encoder_tensor,axis=[1,2])
            return out_tensor
    
    def get_resent50_var(self):
        # target_tensor_list=[self.encoder_paramater["encoder1"],
        # self.encoder_paramater["encoder2"],
        # self.encoder_paramater["encoder3"],
        # self.encoder_paramater["encoder4"]]
        # all_list = []
        # all_var = []
        # result_var = []
        # # 遍历所有变量，node.name得到变量名称
        # # 不使用tf.trainable_variables()，因为batchnorm的moving_mean/variance不属于可训练变量
        # for var in tf.global_variables():
        #     #print(var.name)
        #     if var != []:
        #         if "/" not in var.name: continue
        #         all_list.append(var.name)
        #         all_var.append(var)
        #
        # all_list = list(map(lambda x: x.split("/")[1], all_list))
        #
        # for target_tensor in target_tensor_list:
        #     target = target_tensor.split("/")[1]
        #     try:
        #         # 查找对应变量作用域的索引
        #         ind = all_list[::-1].index(target)
        #         ind = len(all_list) - ind - 1
        #         #print(ind)
        #         #del all_list
        #         #return all_var[:ind + 1]
        #         result_var+=all_var[:ind+1]
        #     except:
        #         print("target_tensor is not exist!")
        # return list(set(result_var))
        result_var = []
        global_var = tf.global_variables()
        for var in global_var:
            #print(var.name)
            if 'resnet_v1_50' in var.name and 'Momentum' not in var.name:
                result_var.append(var)
        return result_var

    