import numpy as np
import tensorflow as tf

sess = tf.Session()
saver = tf.train.import_meta_graph('K:\\resnet\\resnet_imagenet_v2_fp32_20181001\\model.ckpt-225207.meta')
saver.restore(sess,tf.train.latest_checkpoint('K:\\resnet\\resnet_imagenet_v2_fp32_20181001\\checkpoint'))
