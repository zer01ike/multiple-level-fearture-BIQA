from tensorflow.contrib.slim.nets import resnet_v1
from keras_preprocessing import image
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

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
    spacific_tensor = sess.graph.get_tensor_by_name('resnet_v1_50/block2/unit_3/bottleneck_v1/conv2/BatchNorm/moving_mean:0')
    #img = image.load_img('',target_size=(224,224))
    features = sess.run(spacific_tensor,{'Placeholder:0':X_test})
    print(features.shape)