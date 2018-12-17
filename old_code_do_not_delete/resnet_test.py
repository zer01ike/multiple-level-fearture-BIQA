#-*-coding:utf-8-*

import tensorflow as tf
from old_code_do_not_delete.DataTools.datatools import generatelist
from old_code_do_not_delete.DataTools.datatools import readBatchSizeImage

path_to_ckpt = "/home/wangkai/PycharmProjects/multiple-level-fearture-BIQA/save/batch128epochs40/model.ckpt-5"

def inputs_data_test():
    test_file = "/home/wangkai/Paper_MultiFeature_Data/databaserelease2/test.txt"
    data_dir = "/home/wangkai/Paper_MultiFeature_Data/databaserelease2/Patched_data/"
    mean_patch_file = "/home/wangkai/Paper_MultiFeature_Data/databaserelease2/average_mean.png"
    X_test, Y_test, meanBatch = generatelist(data_dir, test_file, mean_patch_file)
    X_test_np, Y_test_np = readBatchSizeImage(0, len(X_test), X_test, Y_test, meanBatch)

    return X_test, Y_test, X_test_np, Y_test_np

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

X_test,Y_test,X_test_np,Y_test_np=inputs_data_test()
batch_size = 256



with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    new_saver = tf.train.import_meta_graph(path_to_ckpt+".meta")
    new_saver.restore(sess,path_to_ckpt)

    predictions = tf.get_collection('predictions')

    grpah = tf.get_default_graph()
    inputs_X = grpah.get_operation_by_name('x_input').outputs[0]
    # test sequence
    # -------------------------------------------------------------------------------
    predictions_sum = []
    for start in range(0, len(X_test), batch_size):
        if start + batch_size > len(X_test):
            start = len(X_test) - batch_size
        x_inputs = X_test_np[start:start + batch_size]
        predictions_test = sess.run(predictions, feed_dict={inputs_X: x_inputs})
        for index in range(0,batch_size):
            predictions_sum.append(predictions_test[index][0])


    error = test_accuracy(predictions_sum,Y_test)
    #print(" error = " + str(error).ljust(15)+"  predictions[0] = "+str(predictions_test[0][0]).ljust(15))
    # ---------------------------------------------------------------------------------
    # test sequence end