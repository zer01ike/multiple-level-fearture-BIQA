#-*-coding:utf-8-*

from tensorflow.contrib.slim.nets import resnet_v1
from tensorflow.contrib import layers as layers_lib
from tensorflow.python.ops import math_ops
from DataTools.datatools import preporcessing
from DataTools.datatools import generatelist
from DataTools.datatools import readBatchSizeImage
from DataTools.datatools import shuffleList
from tensorflow.contrib.layers.python.layers import layers
import tensorflow as tf
import tensorflow.contrib.slim as slim
import time


tf.logging.set_verbosity(tf.logging.INFO)


def test_accuracy(predictions,Y_test):
    count = 0
    sum_pre = 0
    sum_test = 0
    error = 0
    # for i in range(0, len(Y_test)):
    #     count += 1
    #     if count % 50 == 0:
    #         error += abs(sum_pre-sum_test)/50
    #
    #         sum_pre = 0
    #         sum_test = 0
    #     else:
    #         sum_pre += predictions[i]
    #         sum_test += float(Y_test[i])
    return sum(predictions)







def net_graph(inputs_X):
    def encoder(tensor_name, layer_name):
        with tf.variable_scope(layer_name):
            encoder_tensor = tf.get_default_graph().get_tensor_by_name(tensor_name)
            encoder_tensor = layers_lib.conv2d(encoder_tensor, 256, [1, 1], stride=1, padding='VALID', scope="conv1",
                                               activation_fn=tf.nn.relu, normalizer_fn=layers.batch_norm,weights_regularizer=layers_lib.l2_regularizer(1e-4))
            encoder_tensor = layers_lib.conv2d(encoder_tensor, 256, [3, 3], stride=1, padding='VALID', scope="conv3",
                                               activation_fn=tf.nn.relu, normalizer_fn=layers.batch_norm,weights_regularizer=layers_lib.l2_regularizer(1e-4))
            out_tensor = math_ops.reduce_mean(encoder_tensor, [1, 2], name='gap', keepdims=False)

            #old style
            #out_tensor = tf.reduce_mean(encoder_tensor,axis=[1,2])
            return out_tensor

    with slim.arg_scope(resnet_v1.resnet_arg_scope()):
        net, end_points = resnet_v1.resnet_v1_50(inputs_X, is_training=False)

    #orginal net
    with tf.variable_scope("encoder"):
        encoder1 = encoder("resnet_v1_50/block1/unit_3/bottleneck_v1/Relu:0", "encoder1")
        encoder2 = encoder("resnet_v1_50/block2/unit_4/bottleneck_v1/Relu:0", "encoder2")
        encoder3 = encoder("resnet_v1_50/block3/unit_6/bottleneck_v1/Relu:0", "encoder3")
        encoder4 = encoder("resnet_v1_50/block4/unit_3/bottleneck_v1/Relu:0", "encoder4")

        concat = tf.concat([encoder1, encoder2, encoder3, encoder4], -1, name='concat')
        predictions = layers_lib.fully_connected(concat, 1, name="fintune_FC",weights_regularizer=layers_lib.l2_regularizer(1e-4))

        tf.add_to_collection("predictions",predictions)
        current_epoch = tf.Variable(0, name="current_epoch")

    return predictions,current_epoch

def net_graph_debug(inputs_X):
    def encoder(tensor_name, layer_name):
        with tf.variable_scope(layer_name):
            encoder_tensor = tf.get_default_graph().get_tensor_by_name(tensor_name)
            encoder_tensor = layers_lib.conv2d(encoder_tensor, 256, [1, 1], stride=2, padding='SAME', scope="conv1",
                                               activation_fn=tf.nn.relu, normalizer_fn=layers.batch_norm)
            encoder_tensor = layers_lib.conv2d(encoder_tensor, 256, [3, 3], stride=2, padding='SAME', scope="conv3",
                                               activation_fn=tf.nn.relu, normalizer_fn=layers.batch_norm)
            out_tensor = math_ops.reduce_mean(encoder_tensor, [1, 2], name='gap', keepdims=False)

            #old style
            #out_tensor = tf.reduce_mean(encoder_tensor,axis=[1,2])
            return out_tensor
    with slim.arg_scope(resnet_v1.resnet_arg_scope()):
        net, end_points = resnet_v1.resnet_v1_50(inputs_X, is_training=True)
    with tf.variable_scope("encoder"):
        encoder1 = encoder("resnet_v1_50/block1/unit_3/bottleneck_v1/Relu:0", "encoder1")
        encoder2 = encoder("resnet_v1_50/block2/unit_4/bottleneck_v1/Relu:0", "encoder2")
        encoder3 = encoder("resnet_v1_50/block3/unit_6/bottleneck_v1/Relu:0", "encoder3")
        encoder4 = encoder("resnet_v1_50/block4/unit_3/bottleneck_v1/Relu:0", "encoder4")

        concat = tf.concat([encoder1, encoder2, encoder3, encoder4], -1, name='concat')
        predictions = layers_lib.fully_connected(concat, 1, activation_fn=tf.nn.relu, scope="fintune_FC")
        current_epoch = tf.Variable(0, name="current_epoch")

    return predictions,current_epoch

def inference(x):

    '''
    inference the output from the X
    :param x:
    :return:
    '''

    pass

def loss(x,y):
    '''
    define the losss from the input X with the real Y
    :param x:
    :param y:
    :return:
    '''

    loss_fn = tf.losses.mean_squared_error(x, y)
    tf.summary.scalar("loss", loss_fn)

    return loss_fn

def inputs_names_X_Y():
    '''
    get the data to train or test or evaluate
    :return:
    '''
    train_file = "/home/wangkai/Paper_MultiFeature_Data/databaserelease2/train.txt"
    data_dir = "/home/wangkai/Paper_MultiFeature_Data/databaserelease2/Patched_data/"
    mean_patch_file = "/home/wangkai/Paper_MultiFeature_Data/databaserelease2/average_mean.png"
    return generatelist(data_dir, train_file, mean_patch_file)



def inputs_data_full():
    data_dir = "/home/wangkai/Paper_MultiFeature_Data/databaserelease2/Patched_data/"
    summary_file = "/home/wangkai/Paper_MultiFeature_Data/databaserelease2/croped_info.txt"
    mean_patch_file = "/home/wangkai/Paper_MultiFeature_Data/databaserelease2/average_mean.png"
    return preporcessing(data_dir,summary_file,mean_patch_file,0.8,0.2)

def inputs_data_test():
    test_file = "/home/wangkai/Paper_MultiFeature_Data/databaserelease2/test.txt"
    data_dir = "/home/wangkai/Paper_MultiFeature_Data/databaserelease2/Patched_data/"
    mean_patch_file = "/home/wangkai/Paper_MultiFeature_Data/databaserelease2/average_mean.png"
    X_test, Y_test, meanBatch = generatelist(data_dir, test_file, mean_patch_file)
    X_test_np, Y_test_np = readBatchSizeImage(0, len(X_test), X_test, Y_test, meanBatch)

    return X_test, Y_test, X_test_np, Y_test_np


def train(total_loss,current_epoch,train_V):
    '''
    change the model by using the total_loss
    :param total_loss:
    :return:
    '''


    learing_rate = tf.train.exponential_decay(0.001,
                                              current_epoch,
                                              decay_steps=epochs_steps,
                                              decay_rate=0.1)

    momOp= tf.train.MomentumOptimizer(learning_rate=learing_rate,momentum=0.9)
    train_step = momOp.minimize(total_loss,var_list=train_V,global_step=current_epoch)
    # train_step = tf.train.GradientDescentOptimizer(learning_rate=learing_rate).minimize(total_loss, var_list=train_var,
    #                                                                                     global_step=current_epoch)
    tf.summary.scalar("learning_rate", learing_rate)
    return train_step

def evaluate(sess,x,y):
    '''

    :param sess:
    :param x:
    :param y:
    :return:
    '''

    pass


def get_resent50_var(target_tensor_list = None):
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


if __name__ == '__main__':

    batch_size = 256
    epochs = 2
    epochs_steps = 1
    height = 224
    width = 224
    channels = 3

    target_tensor1 = 'resnet_v1_50/block1/unit_3/bottleneck_v1/Relu:0'
    target_tensor2 = 'resnet_v1_50/block2/unit_4/bottleneck_v1/Relu:0'
    target_tensor3 = 'resnet_v1_50/block3/unit_6/bottleneck_v1/Relu:0'
    target_tensor4 = 'resnet_v1_50/block4/unit_3/bottleneck_v1/Relu:0'
    path_to_ckpt = '/home/wangkai/Paper_MultiFeature_Data/resnet/resnet_v1_50.ckpt'

    tf.reset_default_graph()

    # define the tensor for inputs
    inputs_X = tf.placeholder(tf.float32, shape=[batch_size, height, width, channels],name='x_input')
    #tf.add_to_collection("inputs",inputs_X)
    # define the tensor for values
    inputs_Y = tf.placeholder(tf.float32, (batch_size, 1))


    #predictions, current_epoch = net_graph(inputs_X)
    predictions,current_epoch = net_graph_debug(inputs_X)

    resnet50_list = get_resent50_var([target_tensor1,target_tensor2,target_tensor3,target_tensor4])

    #train_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="encoder")

    X, Y, meanBatch = inputs_names_X_Y()

    #X_test,Y_test,X_test_np,Y_test_np=inputs_data_test()

    total_loss = loss(inputs_Y, predictions)

    #tf.summary.scalar("groundTruth",inputs_Y)

    merge_all = tf.summary.merge_all()

    saver = tf.train.Saver(var_list=resnet50_list)


    train_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="encoder")
    for var_i in train_var:
        tf.add_to_collection("encoder_var",var_i)
    train_step = train(total_loss, current_epoch,train_var)

    for i in tf.global_variables():

        if 'Momentum' in i.name :
            tf.add_to_collection("encoder_var", i)

    train_var = tf.get_collection("encoder_var")
    
    strtime = str(time.localtime().tm_hour) +"-"+str(time.localtime().tm_min)
    session_file = open("session_file-"+strtime+".txt",'w')

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        writer = tf.summary.FileWriter(r"./logs/batch128epochs40", sess.graph)
        sess.run(tf.variables_initializer(var_list=train_var))
        saver.restore(sess, path_to_ckpt)
        total_step = 1

        for e in range(epochs):
            current_epoch = e
            X, Y = shuffleList(X, Y)
            for batch_index in range(0, len(X), batch_size):
                total_step += 1
                start = batch_index
                if start + batch_size > len(X):
                    start = len(X) - batch_size
                x_train, y_train = readBatchSizeImage(start, batch_size, X, Y, meanBatch)
                _, losses, score_v, predictions_v, summary_v = sess.run(
                    [train_step, total_loss, inputs_Y, predictions, merge_all],
                    feed_dict={inputs_X: x_train,
                               inputs_Y: y_train})
                # print(
                #     "total_lossess = " + str(losses).ljust(15) +  "  inputs[0] = " + str(score_v[0][0]).ljust(
                #         15)+"  predictions[0] = " + str(predictions_v[0][0]).ljust(
                #         15))
                pre_str = str(e)+":"+str(batch_index)+":"
                if batch_index % 100 == 0:
                    writer.add_summary(summary_v, total_step)
                    print("++Train::Eopchs = " + str(e).ljust(15) +
                          "Steps = " + str(batch_index).ljust(15) +
                          "Loss = " + str(losses).ljust(15) +
                          "predictions = " + str(predictions_v[0][0]).ljust(15))
                    for pre_index in range(batch_size):
                        pre_str +=" "+str(predictions_v[pre_index][0])+"-"
                    # session_file.write("++Train::Eopchs = " + str(e).ljust(15) +
                    #       "Steps = " + str(batch_index).ljust(15) +
                    #       "Loss = " + str(losses).ljust(15) +
                    #       "predictions = " + str(predictions_v[0][0]).ljust(15)+"\n")
                    session_file.write(pre_str+"\n")

            # test sequence
            # -------------------------------------------------------------------------------
            # predictions_sum = []
            # for start_index in range(0, len(X_test), batch_size):
            #     start = start_index
            #     if start + batch_size > len(X_test):
            #         start = len(X_test) - batch_size
            #     x_inputs = X_test_np[start:start_index + batch_size]
            #     predictions_test = sess.run(predictions, feed_dict={inputs_X: x_inputs})
            #     batch_str = ''
            #     for index in range(0,batch_size):
            #         batch_str += str(predictions_test[index][0])+" "
            #         predictions_sum.append(predictions_test[index][0])
            #     print("start:"+batch_str)


            #error = test_accuracy(predictions_sum,Y_test)
            # print("--TEST::epochs = " + str(e+1).ljust(15) +
            #       "error = " + str(error).ljust(15))
            # session_file.write("--TEST::epochs = " + str(e+1).ljust(15) +
            #       "error = " + str(error).ljust(15)+"\n")
            # save_path = saver.save(sess, "save/batch256epochs10/model"+str(e)+".ckpt")
            #---------------------------------------------------------------------------------
            #test sequence end

            #print("total_lossess = "+str(losses).ljust(15)+"  predictions[0] = "+str(predictions_v[0][0]).ljust(15))

            #save_path = saver.save(sess, "save/batch256epochs10/model.ckpt", global_step=e)
    session_file.close()

