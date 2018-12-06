from tensorflow.contrib.slim.nets import resnet_v1
from tensorflow.contrib import layers as layers_lib
from tensorflow.python.ops import math_ops
from DataTools.datatools import preporcessing
from DataTools.datatools import generatelist
from DataTools.datatools import readBatchSizeImage
from tensorflow.contrib.layers.python.layers import layers
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import cv2


tf.logging.set_verbosity(tf.logging.INFO)

path_to_ckpt = '/home/wangkai/Paper_MultiFeature_Data/resnet/resnet_v1_50.ckpt'


def datageneration():
    x_train=[]
    y_train=[]
    x_test =[]
    y_test =[]

    file_dir = "J:\\databaserelease2\\total_data\\"
    iqa_dir = "J:\\databaserelease2\\score.txt"

    file_num = 4910
    image_list =[]
    score_list =[]
    with open(iqa_dir) as f:
        line = f.readline()
        iqa_list = line.split(" ")
    print("reading:".ljust(15)+"IQA_TRUTH_SCORE")

    for i in range(0,file_num):
        score_list.append([iqa_list[i]])

    print("reading:".ljust(15)+"BATCH_IMAGE")
    for index in range(0,file_num):
        file = file_dir+str(i)+".png"
        image = cv2.imread(file)
        mean = np.mean(image,axis=(0,1))
        average_image = np.tile(mean,(image.shape[0],image.shape[1],1))

        image_list.append(image - average_image)

    x_train = np.array(image_list)
    y_train = np.array(score_list)

    return x_train,y_train

def datageneration_full():
    data_dir = "/home/wangkai/Paper_MultiFeature_Data/databaserelease2/Patched_data/"
    summary_file = "/home/wangkai/Paper_MultiFeature_Data/databaserelease2/croped_info.txt"
    mean_patch_file = "/home/wangkai/Paper_MultiFeature_Data/databaserelease2/average_mean.png"
    return preporcessing(data_dir,summary_file,mean_patch_file,0.8,0.2)
def datageneration_read_split():
    train_file = "/home/wangkai/Paper_MultiFeature_Data/databaserelease2/train.txt"
    data_dir = "/home/wangkai/Paper_MultiFeature_Data/databaserelease2/Patched_data/"
    mean_patch_file = "/home/wangkai/Paper_MultiFeature_Data/databaserelease2/average_mean.png"
    return generatelist(data_dir,train_file,mean_patch_file)

def datageneration_test():
    test_file = "/home/wangkai/Paper_MultiFeature_Data/databaserelease2/test.txt"
    data_dir = "/home/wangkai/Paper_MultiFeature_Data/databaserelease2/Patched_data/"
    mean_patch_file = "/home/wangkai/Paper_MultiFeature_Data/databaserelease2/average_mean.png"
    X_test, Y_test, meanBatch = generatelist(data_dir,test_file,mean_patch_file)
    X_test_np, Y_test_np = readBatchSizeImage(0,len(X_test),X_test,Y_test,meanBatch)

    return X_test,Y_test,X_test_np,Y_test_np
def test_accuracy(predictions,Y_test):
    count = 0
    sum_pre = 0
    sum_test = 0
    error = 0
    for i in range(0,len(Y_test)):
        count +=1
        if count == 50:
            error += abs(sum_pre-sum_test)/50
            sum_pre = 0
            sum_test = 0
        else:
            sum_pre += predictions[i]
            sum_test += float(Y_test[i])
    return error




def datageneration_slow(start,batch_size,X,Y,meanBatch):
    #TODO: need to add splited data generation
    X_train=[]
    Y_train=[]
    X_train_np = np.empty([0, 224, 224, 3])
    Y_train_np = np.empty([0, 1])
    if len(Y_train) <len(Y):
        x_train, y_train = readBatchSizeImage(start, batch_size, X, Y, meanBatch)
        X_train_np = np.append(X_train_np,x_train,axis=0)
        Y_train_np = np.append(Y_train_np,y_train,axis=0)
        return x_train,y_train
    else:
        return X_train_np[start:start+batch_size],Y_train_np[start:start+batch_size]


def conv(in_tensor, layer_name, kernel_size, stride, out_chan, trainable=True):
    with tf.name_scope(layer_name):
        in_size = in_tensor.get_shape().as_list()
        # print(in_size)

        strides = [1, stride, stride, 1]
        kernel_shape = [kernel_size, kernel_size, in_size[3], out_chan]

        # conv
        kernel = tf.get_variable(layer_name+'weights', kernel_shape, tf.float32,
                                 tf.contrib.layers.xavier_initializer_conv2d(), trainable=trainable, collections=['wd', 'variables', 'filters'])
        # kernel = tf.get_variable('weights', kernel_shape, tf.float32,
        #                          tf.truncated_normal_initializer(), trainable=trainable)


        # bias
        biases = tf.get_variable(layer_name+'biases', [kernel_shape[3]], tf.float32,
                                 tf.constant_initializer(0.0001), trainable=trainable, collections=['wd', 'variables', 'biases'])

        # out_tensor = tf.nn.bias_add(tmp_result, biases, name='out')

        # kernel = tf.Variable(tf.truncated_normal(shape=kernel_shape, stddev=0.1))
        # biases = tf.Variable(tf.constant(0.1, shape=[out_chan]))
        out_tensor = tf.nn.conv2d(in_tensor, kernel, strides, padding='VALID') + biases
        out_tensor = tf.nn.relu(out_tensor, name='out')

        return out_tensor

def fully_connected(in_tensor, layer_name, out_chan, trainable=True):
    with tf.variable_scope(layer_name):
        in_size = in_tensor.get_shape().as_list()
        assert len(in_size) == 2, 'Input to a fully connected layer must be a vector.'
        weights_shape = [in_size[1], out_chan]

        # weight matrix
        weights = tf.get_variable(layer_name+'weights', weights_shape, tf.float32,
                                  tf.truncated_normal_initializer(), trainable=trainable)
        #weights = tf.check_numerics(weights, 'weights: %s' % layer_name)

        # bias
        biases = tf.get_variable(layer_name+'biases', [out_chan], tf.float32,
                                 tf.constant_initializer(0.0001), trainable=trainable)
        #biases = tf.check_numerics(biases, 'biases: %s' % layer_name)

        out_tensor = tf.matmul(in_tensor, weights) + biases
        return out_tensor


def encoder(tensor_name,layer_name):
    with tf.variable_scope(layer_name):
        encoder_tensor = tf.get_default_graph().get_tensor_by_name(tensor_name)
        encoder_tensor = layers_lib.conv2d(encoder_tensor, 256, [1, 1], stride=1, padding='VALID', scope="conv1",
                                     activation_fn=tf.nn.relu, normalizer_fn=layers.batch_norm)
        encoder_tensor = layers_lib.conv2d(encoder_tensor, 256, [3, 3], stride=1, padding='VALID', scope="conv3",
                                     activation_fn=tf.nn.relu, normalizer_fn=layers.batch_norm)
        out_tensor = math_ops.reduce_mean(encoder_tensor, [1, 2], name='gap', keepdims=False)
        return out_tensor
####


tf.reset_default_graph()

batch_size = 8
epochs = 10
height = 224
width = 224
channels = 3


inputs = tf.placeholder(tf.float32,shape=[batch_size, height, width, channels])
with slim.arg_scope(resnet_v1.resnet_arg_scope()):
    net,end_points = resnet_v1.resnet_v1_50(inputs,is_training=False)

with tf.variable_scope("encoder"):
    encoder1 = encoder("resnet_v1_50/block1/unit_3/bottleneck_v1/Relu:0","encoder1")
    encoder2 = encoder("resnet_v1_50/block2/unit_4/bottleneck_v1/Relu:0","encoder2")
    encoder3 = encoder("resnet_v1_50/block3/unit_6/bottleneck_v1/Relu:0","encoder3")
    encoder4 = encoder("resnet_v1_50/block4/unit_3/bottleneck_v1/Relu:0","encoder4")

    #old samples
    # block1_tensor = tf.get_default_graph().get_tensor_by_name('resnet_v1_50/block1/unit_3/bottleneck_v1/Relu:0')
    # block1_encoder_conv1 = conv(block1_tensor, 'block1_encoder_conv1', 1, 1, 256)
    # block1_encoder_conv2 = conv(block1_encoder_conv1, 'block1_encoder_conv2', 3, 1, 256)
    # encoder1 = tf.reduce_mean(block1_encoder_conv2, axis=[1, 2], name='block1_encoder_gap')

with tf.variable_scope("concat"):
    concat = tf.concat([encoder1, encoder2, encoder3, encoder4], -1, name='concat')

with tf.variable_scope("fc"):
    predictions = layers_lib.fully_connected(concat,1,activation_fn=tf.nn.relu,scope="fintune_FC")
    current_epoch = tf.Variable(0,name="current_epoch")
    # fc_weights = tf.get_variable("fc_weights",[1024,1],trainable=True)
    # fc_bias = tf.get_variable("fc_bias",[1],trainable=True)
    # predictions = tf.sigmoid(tf.matmul(concat, fc_weights)+fc_bias)
    #predictions = tf.matmul(concat,fc_weights)+fc_bias


encoder_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="encoder")
concat_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="concat")
fc_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="fc")

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
        #print(ind)
        del all_list
        return all_var[:ind+1]
    except:
        print("target_tensor is not exist!")
    #return all_var

def get_all_weights():
    weight_list = []
    for var in tf.global_variables():
        if "weights" in var.name:
            weight_list.append(var)
        elif "gamma" in var.name:
            weight_list.append(var)
        elif "beta" in var.name:
            weight_list.append(var)
    return weight_list


def get_var_list_2(target_tensor_list = None):
    all_list = []
    all_var = []
    result_var = []
    # 遍历所有变量，node.name得到变量名称
    # 不使用tf.trainable_variables()，因为batchnorm的moving_mean/variance不属于可训练变量
    for var in tf.global_variables():
        #print(var.name)
        if var != []:
            if "/" not in var.name: continue
            all_list.append(var.name)
            all_var.append(var)

    all_list = list(map(lambda x: x.split("/")[1], all_list))

    for target_tensor in target_tensor_list:
        target = target_tensor.split("/")[1]
        try:
            # 查找对应变量作用域的索引
            ind = all_list[::-1].index(target)
            ind = len(all_list) - ind - 1
            #print(ind)
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


learing_rate = tf.train.exponential_decay(0.0001,
                                          current_epoch,
                                          decay_steps=epochs,
                                          decay_rate=0.1)


#tf.add_to_collection("encoder",current_epoch)
# train_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='encoder')
train_var = encoder_var+concat_var+fc_var

weight_list = get_all_weights()

# for i in tf.get_collection('weights'):
#    tf.add_to_collection(tf.GraphKeys.WEIGHTS,i)

regularizer = layers_lib.l2_regularizer(1e-4)
reg_term = layers_lib.apply_regularization(regularizer,weight_list)

X,Y,meanBatch = datageneration_read_split()
X_test,Y_test,X_test_np,Y_test_np=datageneration_test()
y_score = tf.placeholder(tf.float32, (batch_size,1))


loss = tf.losses.mean_squared_error(y_score, predictions)+reg_term
#print(train_var)
train_step = tf.train.GradientDescentOptimizer(learning_rate=learing_rate).minimize(loss, var_list=train_var,global_step=current_epoch)

## 合并所有summary
tf.summary.scalar("loss",loss)
#merge_all = tf.summary.merge_all()
merge_all = tf.summary.merge_all(key='summaries')

# x_train = np.ones([141, 224, 224, 3])
# y_train = np.ones([141, 1])
#y_train = np.ones(size=(141, 1))

#x_train,y_train = datageneration()
# x_train,y_train,x_test,y_test = datageneration_full()


# var_list = get_var_list_2([target_tensor1])



saver = tf.train.Saver(var_list=var_list)

# saver = tf.train.Saver(var_list=train_var)




with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    writer = tf.summary.FileWriter(r"./logs", sess.graph)
    sess.run(tf.variables_initializer(var_list=train_var))
    # sess.run(tf.global_variables_initializer())
    saver.restore(sess, path_to_ckpt)

    # for i in range(140000):
    #     start = (i * batch_size) % x_train.shape[0]
    #     end = min(start + batch_size, x_train.shape[0])
    #     _, losses, score_v,predictions_v,summary_v= sess.run([train_step, loss, y_score, predictions,merge_all], feed_dict={inputs: x_train[start:end], y_score: y_train[start:end]})
    #     if i % 100 == 0:
    #         writer.add_summary(summary_v, i)
    #         save_path = saver.save(sess, "save/model.ckpt")
    #         print("total_lossess = "+str(losses).ljust(15)+"  predictions = "+str(predictions_v[0][0]).ljust(15))
    #     #print(_v)

    for e in range(epochs):
        current_epoch = e
        for i in range(0,len(X),batch_size):
            start = i
            if start + batch_size > len(X):
                start = len(X) - batch_size
            x_train, y_train = readBatchSizeImage(start, batch_size, X, Y, meanBatch)
            _, losses, score_v, concat_v, predictions_v, summary_v = sess.run(
                [train_step, loss, y_score, concat, predictions, merge_all],
                feed_dict={inputs: x_train,
                           y_score: y_train})

            if i % 1000 == 0:
                writer.add_summary(summary_v, i)
                save_path = saver.save(sess, "save/model.ckpt")
                print("total_lossess = "+str(losses).ljust(15)+"  predictions = "+str(predictions_v[0][0]).ljust(15))


                predictions_sum = []
                for start in range(0, len(X_test), batch_size):
                    if start + batch_size > len(X_test):
                        start = len(X_test) - batch_size
                    x_inputs = X_test_np[start:start + batch_size]
                    predictions_test = sess.run(predictions, feed_dict={inputs: x_inputs})
                    for i in range(0,batch_size):
                        predictions_sum.append(predictions_test[i][0])


                error = test_accuracy(predictions_sum,Y_test)
                print("epochs = " + str(e+1).ljust(15) + " error = " + str(error).ljust(15)+"  predictions[0] = "+str(predictions_test[0][0]).ljust(15))
    # for i in range(0,len(X) * epochs, batch_size):
    #     start = i % len(X)
    #     if start + batch_size > len(X):
    #         start = len(X)-batch_size
    #     x_train,y_train = readBatchSizeImage(start,batch_size,X,Y,meanBatch)
    #     _, losses, score_v, concat_v, predictions_v, summary_v = sess.run([train_step, loss, y_score,concat, predictions, merge_all],
    #                                                             feed_dict={inputs: x_train,
    #                                                                        y_score: y_train})
    #     if i % 100 == 0:
    #         writer.add_summary(summary_v, i)
    #         save_path = saver.save(sess, "save/model.ckpt")
    #         #print("total_lossess = "+str(losses).ljust(15)+"  predictions = "+str(predictions_v[0][0]).ljust(15))
    #         print("total_lossess = "+str(losses).ljust(15)+"  concat = "+str(concat_v[0][4]).ljust(15)+"  predictions = "+str(predictions_v[0][0]).ljust(15))

