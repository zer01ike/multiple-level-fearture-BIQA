from tensorflow.contrib.slim.nets import resnet_v1
from keras_preprocessing import image
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import nn_ops

batch_size = 1
height = 224
width = 224
channels = 3
path_to_ckpt = 'K:\\resnet\\resnet_v1_50.ckpt'

inputs  = tf.placeholder(tf.float32,shape=[batch_size,height,width,channels])
with slim.arg_scope(resnet_v1.resnet_arg_scope()):
    net,end_points = resnet_v1.resnet_v1_50(inputs,is_training=False)

saver = tf.train.Saver()
X_test = np.ones((1,224,224,3))
with tf.Session() as sess:
    saver.restore(sess,path_to_ckpt)
    block1_tensor = sess.graph.get_tensor_by_name('resnet_v1_50/block1/unit_3/bottleneck_v1/Relu:0')
    block2_tensor = sess.graph.get_tensor_by_name('resnet_v1_50/block2/unit_4/bottleneck_v1/Relu:0')
    block3_tensor = sess.graph.get_tensor_by_name('resnet_v1_50/block3/unit_6/bottleneck_v1/Relu:0')
    block4_tensor = sess.graph.get_tensor_by_name('resnet_v1_50/block4/unit_3/bottleneck_v1/Relu:0')
    #output = nn_ops.relu(b1_u1_shortcut_tensor + b1_u1_conv3_tensor)

    #img = image.load_img('',target_size=(224,224))
    features = sess.run([block1_tensor,block2_tensor,block3_tensor,block4_tensor],{'Placeholder:0':X_test})

    #variable_name = [v.name for v in tf.trainable_variables()]
    #print(variable_name)

    # file = open("op_info.txt","w")
    # op = sess.graph.get_operations()
    # for i in range(0,len(op)):
    #     file.write(str(op[i]))
    #
    # file.close()

    print(features.shape)