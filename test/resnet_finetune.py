from tensorflow.contrib.slim.nets import resnet_v1
from keras_preprocessing import image
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

####

def leaky_relu( tensor, name='relu'):
    neg_slope_of_relu=0.01
    out_tensor = tf.maximum(tensor, neg_slope_of_relu*tensor, name=name)
    return out_tensor


def conv(in_tensor, layer_name, kernel_size, stride, out_chan, trainable=True):
    with tf.variable_scope(layer_name):
        in_size = in_tensor.get_shape().as_list()

        strides = [1, stride, stride, 1]
        kernel_shape = [kernel_size, kernel_size, in_size[3], out_chan]

        # conv
        # kernel = tf.get_variable('weights', kernel_shape, tf.float32,
        #                          tf.contrib.layers.xavier_initializer_conv2d(), trainable=trainable, collections=['wd', 'variables', 'filters'])
        kernel = tf.get_variable('weights', kernel_shape, tf.float32,
                                 tf.truncated_normal_initializer(), trainable=trainable)
        tmp_result = tf.nn.conv2d(in_tensor, kernel, strides, padding='SAME')

        # bias
        # biases = tf.get_variable('biases', [kernel_shape[3]], tf.float32,
        #                          tf.constant_initializer(0.0001), trainable=trainable, collections=['wd', 'variables', 'biases'])
        biases = tf.get_variable('biases', [kernel_shape[3]], tf.float32,
                                 tf.constant_initializer(0.0001), trainable=trainable)
        # out_tensor = tf.nn.bias_add(tmp_result, biases, name='out')
        out_tensor = tf.nn.bias_add(tmp_result, biases)


        return out_tensor


def conv_relu( in_tensor, layer_name, kernel_size, stride, out_chan, trainable=True):
    tensor = conv(in_tensor, layer_name, kernel_size, stride, out_chan, trainable)
    out_tensor = leaky_relu(tensor, name='out')
    return out_tensor


def max_pool( bottom, name='pool'):
    pooled = tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                            padding='VALID', name=name)
    return pooled

def fully_connected(in_tensor, layer_name, out_chan, trainable=True):
    with tf.variable_scope(layer_name):
        in_size = in_tensor.get_shape().as_list()
        assert len(in_size) == 2, 'Input to a fully connected layer must be a vector.'
        weights_shape = [in_size[1], out_chan]

        # weight matrix
        weights = tf.get_variable('weights', weights_shape, tf.float32,
                                  tf.truncated_normal_initializer(), trainable=trainable)
        #weights = tf.check_numerics(weights, 'weights: %s' % layer_name)

        # bias
        biases = tf.get_variable('biases', [out_chan], tf.float32,
                                 tf.constant_initializer(0.0001), trainable=trainable)
        #biases = tf.check_numerics(biases, 'biases: %s' % layer_name)

        out_tensor = tf.matmul(in_tensor, weights) + biases
        return out_tensor

####


tf.reset_default_graph()

batch_size = 1
height = 224
width = 224
channels = 3
path_to_ckpt = 'K:\\resnet\\resnet_v1_50.ckpt'

inputs = tf.placeholder(tf.float32,shape=[batch_size, height, width, channels])
with slim.arg_scope(resnet_v1.resnet_arg_scope()):
    net,end_points = resnet_v1.resnet_v1_50(inputs,is_training=False)

with tf.variable_scope("finetune_layers"):
    block1_tensor = tf.get_default_graph().get_tensor_by_name('resnet_v1_50/block1/unit_3/bottleneck_v1/Relu:0')
    # block1_encoder_conv1 = tf.layers.Conv2D(filters=256, kernel_size=1, name='block1_encoder_conv1')(block1_tensor)
    # block1_encoder_conv1 = tf.layers.conv2d(block1_tensor, 256, [1, 1], strides=1,name='block1_encoder_conv1')
    # block1_encoder_conv2 = tf.layers.Conv2D(filters=256,kernel_size=3,name='block1_encoder_conv3')(block1_encoder_conv1)

    block1_encoder_conv1 = conv(block1_tensor,"block1_encoder_conv1",1 ,1,256)
    block1_encoder_conv2 = conv(block1_encoder_conv1, "block1_encoder_conv2", 3, 1, 256)
    block1_encoder_gap = tf.reduce_mean(block1_encoder_conv2, axis=[1,2],name='block1_encoder_gap')

    block2_tensor = tf.get_default_graph().get_tensor_by_name('resnet_v1_50/block2/unit_4/bottleneck_v1/Relu:0')
    block2_encoder_conv1 = conv(block2_tensor, "block2_encoder_conv1", 1, 1, 256)
    block2_encoder_conv2 = conv(block2_encoder_conv1, "block2_encoder_conv2", 3, 1, 256)
    block2_encoder_gap = tf.reduce_mean(block2_encoder_conv2, axis=[1, 2], name='block2_encoder_gap')


    block3_tensor = tf.get_default_graph().get_tensor_by_name('resnet_v1_50/block3/unit_6/bottleneck_v1/Relu:0')
    # block3_encoder_conv1 = tf.layers.Conv2D(filters=256, kernel_size=1, name='block3_encoder_conv1')(block3_tensor)
    # block3_encoder_conv2 = tf.layers.Conv2D(filters=256, kernel_size=3, name='block3_encoder_conv3')(
    #     block3_encoder_conv1)
    # block3_encoder_gap = tf.reduce_mean(block3_encoder_conv2, axis=[1, 2],name='block3_encoder_gap')
    block3_encoder_conv1 = conv(block3_tensor, "block3_encoder_conv1", 1, 1, 256)
    block3_encoder_conv2 = conv(block3_encoder_conv1, "block3_encoder_conv2", 3, 1, 256)
    block3_encoder_gap = tf.reduce_mean(block3_encoder_conv2, axis=[1, 2], name='block3_encoder_gap')

    block4_tensor = tf.get_default_graph().get_tensor_by_name('resnet_v1_50/block4/unit_3/bottleneck_v1/Relu:0')
    # block4_encoder_conv1 = tf.layers.Conv2D(filters=256, kernel_size=1, name='block4_encoder_conv1')(block4_tensor)
    # block4_encoder_conv2 = tf.layers.Conv2D(filters=256, kernel_size=3, name='block4_encoder_conv3')(
    #     block4_encoder_conv1)
    # block4_encoder_gap = tf.reduce_mean(block4_encoder_conv2, axis=[1, 2],name='block4_encoder_gap')
    block4_encoder_conv1 = conv(block4_tensor, "block4_encoder_conv1", 1, 1, 256)
    block4_encoder_conv2 = conv(block4_encoder_conv1, "block4_encoder_conv2", 3, 1, 256)
    block4_encoder_gap = tf.reduce_mean(block4_encoder_conv2, axis=[1, 2], name='block4_encoder_gap')

    concat = tf.concat([block1_encoder_gap,block2_encoder_gap,block3_encoder_gap,block4_encoder_gap],1,name='concat')

    #add full connected
    predictions = fully_connected(concat,"full_connected",1)
    # weights = tf.Variable(tf.random_normal([1024, 1]), name="weights")
    # bias = tf.Variable(tf.random_normal([1]), name='bias')
    # predictions = tf.matmul(concat, weights)+bias


x_train = np.ones([141, 224, 224, 3])
y_train = np.ones([141,1])
#y_train = np.ones(size=(141, 1))
y_score = tf.placeholder(tf.float32, (1,1))

train_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='finetune_layers')

loss = tf.losses.mean_squared_error(y_score, predictions)

train_step = tf.train.GradientDescentOptimizer(0.0001).minimize(loss, var_list=train_var)

## 合并所有summary
# merge_all = tf.summary.merge_all()

def get_var_list(target_tensor=None):
    '''获取指定变量列表var_list的函数'''
    if target_tensor==None:
        target_tensor = r"MobilenetV2/expanded_conv_14/output:0"
    target = target_tensor.split("/")[1]
    all_list = []
    all_var = []
    # 遍历所有变量，node.name得到变量名称
    # 不使用tf.trainable_variables()，因为batchnorm的moving_mean/variance不属于可训练变量
    for var in tf.global_variables():
        if var != []:
            all_list.append(var.name)
            all_var.append(var)

    #print(all_list)
    try:
        all_list = list(map(lambda x:x.split("/")[1],all_list))
        # 查找对应变量作用域的索引
        ind = all_list[::-1].index(target)
        ind = len(all_list) -  ind - 1
        print(ind)
        del all_list
        return all_var[:ind+1]
    except:
        print("target_tensor is not exist!")
    #return all_var

def get_var_list_2(target_tensor_list = None):
    all_list = []
    all_var = []
    result_var = []
    # 遍历所有变量，node.name得到变量名称
    # 不使用tf.trainable_variables()，因为batchnorm的moving_mean/variance不属于可训练变量
    for var in tf.global_variables():
        if var != []:
            all_list.append(var.name)
            all_var.append(var)

    all_list = list(map(lambda x: x.split("/")[1], all_list))

    for target_tensor in target_tensor_list:
        target = target_tensor.split("/")[1]
        try:
            # 查找对应变量作用域的索引
            ind = all_list[::-1].index(target)
            ind = len(all_list) - ind - 1
            print(ind)
            #del all_list
            #return all_var[:ind + 1]
            result_var+=all_var[:ind+1]
        except:
            print("target_tensor is not exist!")
    return list(set(result_var))

target_tensor1 = 'resnet_v1_50/block1/unit_3/bottleneck_v1/Relu:0'
target_tensor2 = 'resnet_v1_50/block2/unit_4/bottleneck_v1/Relu:0'
target_tensor3 = 'resnet_v1_50/block3/unit_6/bottleneck_v1/Relu:0'
target_tensor4 = 'resnet_v1_50/block4/unit_3/bottleneck_v1/Relu:0'

var_list = get_var_list_2([target_tensor1,target_tensor2,target_tensor3,target_tensor4])


saver = tf.train.Saver(var_list=var_list)

# saver = tf.train.Saver(var_list=train_var)

with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    writer = tf.summary.FileWriter(r"./logs", sess.graph)
    sess.run(tf.variables_initializer(var_list=train_var))
    # sess.run(tf.global_variables_initializer())
    saver.restore(sess, path_to_ckpt)

    for i in range(2000):
        start = (i * batch_size) % x_train.shape[0]
        end = min(start + batch_size, x_train.shape[0])
        _, losses, score_v,predictions_v = sess.run([train_step, loss, y_score,predictions], feed_dict={inputs: x_train[start:end], y_score: y_train[start:end]})
        # if i % 100 == 0:
        #     writer.add_summary(merge, i)
        print(losses,score_v,predictions_v)
        #print(_v)



# def inference():
#     pass
#
#
# def loss():
#     pass
#
#
# def inputs():
#     pass
#
#
# def train(X, Y):
#
#
# def evaluate():
    #pass