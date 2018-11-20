import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import nets

import numpy as np
import os


def get_next_batch(batch_size=64, ...):
    """Get a batch set of training data.

    Args:
        batch_size: An integer representing the batch size.
        ...: Additional arguments.
    Returns:
        images: A 4-D numpy array with shape [batch_size, height, width,
            num_channels] representing a batch of images.
        labels: A 1-D numpy array with shape [batch_size] representing
            the groundtruth labels of the corresponding images.
    """
    ...  # Get images and the corresponding groundtruth labels.
    return images, labels

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] ="0"

    batch_size = 64
    num_classes = 5
    num_steps = 10000

    resnet_model_path = 'K:\\resnet\\resnet_v1_50.ckpt'
    model_save_path = 'K:\\resnet\\resnet_test\\model'

    inputs = tf.placeholder(tf.float32,shape=[None,224,224,3],name='inputs')
    labels = tf.placeholder(tf.int32,shape=[None],name='labels')
    is_training = tf.placeholder(tf.bool,name='is_training')

    with slim.arg_scope(nets.resnet_v2.resnet_arg_scope()):
        net,endpoints = nets.resnet_v2.resnet_v2_50(inputs,num_classes=None,is_training=is_training)

    with tf.variable_scope('Logits'):
        net = tf.squeeze(net,axis=[1,2])
        net = slim.dropout(net,keep_prob=0.5,scope='scope')
        logits = slim.fully_connected(net,num_outputs=num_classes,activation_fn=None,scope='fc')

    checkpoint_exclude_scopes = 'Logits'
    exclusions = None

    if checkpoint_exclude_scopes:
        exclusion = [scope.strip() for scope in checkpoint_exclude_scopes.split(',')]

    variables_to_restore = []

    for var in slim.get_model_variables():
        excluded =False
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                excluded = True
        if not excluded:
            variables_to_restore.append(var)

    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits)
    loss = tf.reduce_mean(losses)

    logits = tf.nn.softmax(logits)
    classes = tf.argmax(logits, axis=1, name='classes')
    accuracy = tf.reduce_mean(tf.cast(
        tf.equal(tf.cast(classes, dtype=tf.int32), labels), dtype=tf.float32))

    optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
    train_step = optimizer.minimize(loss)

    init = tf.global_variables_initializer()

    saver_restore = tf.train.Saver(var_list=variables_to_restore)
    saver = tf.train.Saver(tf.global_variables())

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.per_process_gpu_memory_fraction = 0.95
    with tf.Session(config=config) as sess:
        sess.run(init)

        # Load the pretrained checkpoint file xxx.ckpt
        saver_restore.restore(sess, resnet_model_path)

        for i in range(num_steps):
            images, groundtruth_lists = get_next_batch(batch_size, ...)
            train_dict = {inputs: images,
                          labels: groundtruth_lists,
                          is_training: True}

            sess.run(train_step, feed_dict=train_dict)

            loss_, acc_ = sess.run([loss, accuracy], feed_dict=train_dict)

            train_text = 'Step: {}, Loss: {:.4f}, Accuracy: {:.4f}'.format(
                i + 1, loss_, acc_)
            print(train_text)

            if (i + 1) % 1000 == 0:
                saver.save(sess, model_save_path, global_step=i + 1)
                print('save mode to {}'.format(model_save_path))




