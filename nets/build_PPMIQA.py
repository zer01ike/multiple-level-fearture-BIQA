# _*_ coding:utf-8 _*_
# @Time     :12/21/18 5:42 PM
# @Author   :zer01ike
# @FileName : build_PPMIQA.py
# @gitHub   : https://github.com/zer01ike

from __future__ import print_function,unicode_literals
from typing import List,Any,Sequence
import sys
sys.path.append('/home/wangkai/Pycharm_syn_from_my_macbook/')
# print(sys.path)

import tensorflow as tf
import numpy as np
import os
from data.LiveIQADataset import LiveIQADataset
from data.LiveCHIQADataset import LiveCHIQADataset
from data.TID2013Dataset import TID2013Dataset
from data.IETRDataset import IETRDataset
from data.SynIQADataset import SynIQADataset
from data.IrccynDataset import IrccynDataset
from nets.PPMIQA_network import PPMIQA
import time
from scipy.stats import spearmanr
from scipy.stats import pearsonr

class PPMIQAmodel(object):
    @classmethod
    def default_params(cls):
        return {
            'root_dir': "/home/wangkai/",
            'resnet_ckpt': "/home/wangkai/Paper_MultiFeature_Data/resnet/resnet_v1_50.ckpt",
            'vgg_ckpt':"/home/wangkai/Paper_MultiFeature_Data/vgg/vgg_19.ckpt",
            'output_log': "/home/wangkai/logs_save/ppmiqa_vgg_on_syn_data.txt",
            'summary_dir': "/home/wangkai/logs_save/logs/ppmiqa_vgg_on_syn_data/",
            'save_dir': "/home/wangkai/logs_save/save/ppmiqa_vgg_on_syn_data/",
            'train': False,
            'mode':'test_single',
            'dataset':'SynIQADataset',
            'restore_file': '/home/wangkai/logs_save/save/ppmiqa_vgg_on_syn_data/',
            'restore_name': 'saved_3ckpt',
            'orginal_learning_rate': 0.001,
            'decay_steps': 10,
            'decay_rate': 0.1,
            'momentum': 0.9,
            'epochs': 4,
            'corp_size': 10,
            'batch_size': 1,
            'height': 224,
            'width': 224,
            'channels': 3}

    def __init__(self,image_name,demos,type,dataset_in_use):
        self.image_name = image_name
        self.demos = demos
        self.type = type

        self.params = self.default_params()
        self.params.update({'dataset':dataset_in_use})

        self.graph = tf.Graph()

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)

        self.sess = tf.Session(graph=self.graph,config=tf.ConfigProto(gpu_options=gpu_options))

        with self.graph.as_default():
            self.ops = {}
            self.data = {}
            #self.get_DataSet()
            self.get_dataSet_with_name_mode()

            #self.build_PPMIQA_net(self.sess)

            self.build_PPMIQA_net_VGG(self.sess)

            self.make_train_step()

            if self.params['train'] is True:
                #self.initial_model()
                self.initial_model_VGG()
            else:
                self.restore_model(self.params['restore_file'])


    def createInstance(self,module_name,class_name,*args,**kwargs):
        module = __import__(module_name,globals(),locals(),[class_name])
        class_instance = getattr(module,class_name)
        obj = class_instance(*args,**kwargs)
        return obj

    def get_dataSet_with_name_mode(self):
        '''
        get the dataset with the name and
        :return:
        '''

        batch_size = self.params['batch_size']

        if self.params['train'] is not True:
            batch_size = 1

        dataset = self.createInstance('data.' + self.params['dataset'], self.params['dataset'],
                                      batch_size=batch_size, shuffle=True, crop_size=50,
                                      num_epochs=self.params['corp_size'],
                                      crop_shape=[self.params['height'], self.params['width'],
                                                  self.params['channels']])


        if 'test_single' == self.params['mode']:

            dataset = dataset.get_single_dataset(self.image_name,self.demos,self.type)
            iter = tf.data.Iterator.from_structure(dataset.output_types, dataset.output_shapes)

            self.ops['init_op'] = iter.make_initializer(dataset)

            self.data['demos'], self.data['image'] = iter.get_next()

        elif 'train_and_test' == self.params['mode']:
            train_dataset = dataset.get_train_dataset()
            test_dataset = dataset.get_test_dataset()

            iter = tf.data.Iterator.from_structure(train_dataset.output_types,train_dataset.output_shapes)

            self.ops['train_init_op'] = iter.make_initializer(train_dataset)
            self.ops['test_init_op'] = iter.make_initializer(test_dataset)

            self.data['demos'],self.data['image'] = iter.get_next()
        elif 'train' == self.params['mode']:
            train_dataset = dataset.get_train_dataset()
            iter = tf.data.Iterator.from_structure(train_dataset.output_types,train_dataset.output_shapes)
            self.ops['init_op'] = iter.make_initializer(train_dataset)

            self.data['demos'], self.data['image'] = iter.get_next()
        elif 'test' == self.params['mode']:
            test_dataset = dataset.get_test_dataset()

            iter = tf.data.Iterator.from_structure(test_dataset.output_types, test_dataset.output_shapes)

            self.ops['init_op'] = iter.make_initializer(test_dataset)

            self.data['demos'], self.data['image'] = iter.get_next()

        elif 'feature' == self.params['mode']:
            dataset = dataset.get_patch_dataset(self.image_name, self.demos, self.type)
            iter = tf.data.Iterator.from_structure(dataset.output_types, dataset.output_shapes)

            self.ops['init_op'] = iter.make_initializer(dataset)

            self.data['demos'], self.data['image'] = iter.get_next()

        else:
            assert("Error: Your should define the Mode first!")
            exit(code=25)



    def get_DataSet(self):
        '''
        this is the function to get the Dataset
        :return: None But set the dict for this class
        '''

        # get the dataset from the LiveIQA Dataset
        # dataset = LiveIQADataset(mode='training', batch_size=self.params['batch_size'], shuffle=True, crop_size=50,
        #                          num_epochs=self.params['corp_size'],
        #                          crop_shape=[self.params['height'], self.params['width'], self.params['channels']])

        # dataset = LiveCHIQADataset(batch_size=self.params['batch_size'], shuffle=True, crop_size=50,
        #                          num_epochs=self.params['corp_size'],
        #                          crop_shape=[self.params['height'], self.params['width'], self.params['channels']])

        dataset = TID2013Dataset(batch_size=self.params['batch_size'], shuffle=True, crop_size=50,
                                   num_epochs=self.params['corp_size'],
                                   crop_shape=[self.params['height'], self.params['width'], self.params['channels']])

        # if self.params['train'] is True:
        #     dataset = SynIQADataset(batch_size=self.params['batch_size'], shuffle=True, crop_size=50,
        #                          num_epochs=self.params['corp_size'],
        #                          crop_shape=[self.params['height'], self.params['width'], self.params['channels']])
        #     dataset = dataset.get_train_dataset()
        # else:
        #     dataset = SynIQADataset(batch_size=1, shuffle=True, crop_size=50,
        #                             num_epochs=self.params['corp_size'],
        #                             crop_shape=[self.params['height'], self.params['width'], self.params['channels']])
        #     #dataset = dataset.get_test_dataset()
        #     dataset = dataset.get_single_dataset(image_name='01_18_02_Book_arrival_A2_10_to_9_70.bmp', demos= 3.933333333)
        #
        # iter = tf.data.Iterator.from_structure(dataset.output_types, dataset.output_shapes)
        #
        # self.ops['init_op'] = iter.make_initializer(dataset)
        #
        # self.data['demos'], self.data['image'] = iter.get_next()

        # get the train dataset
        train_dataset = dataset.get_train_dataset()
        # get the test dataset
        test_dataset = dataset.get_test_dataset()

        # put it in the params dict
        self.params['train_dataset'] = train_dataset
        self.params['test_dataset'] = test_dataset

        # set the dict of both train and test
        iter = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)

        # define the init op of train step and test step
        self.ops['train_init_op'] = iter.make_initializer(train_dataset)
        self.ops['test_init_op'] = iter.make_initializer(test_dataset)

        # get the demo and image from the dataset
        self.data['demos'], self.data['image'] = iter.get_next()

    def build_PPMIQA_net(self,sess):
        self.net = PPMIQA(sess)

        self.current_epoch = tf.Variable(0, name="current_epoch")

        self.ops['predictions'] = self.net.inference(self.data['image'])

        #self.ops['loss'] = tf.losses.mean_squared_error(self.ops['predictions'],self.data['demos'])
        self.ops['loss'] = tf.reduce_sum(tf.square(self.ops['predictions']-self.data['demos']))

        # self.ops['feature_block1'],self.ops['feature_block4'] = self.net.get_block_feature()

        # set the writer for summary
        self.train_writer = tf.summary.FileWriter(self.params['summary_dir'], self.sess.graph)

        # add the loss to the summary
        tf.summary.scalar('loss_totoal', self.ops['loss'])

    def build_PPMIQA_net_VGG(self,sess):
        self.net = PPMIQA(sess)

        self.current_epoch = tf.Variable(0,name='current_epoch')
        self.ops['predictions'] = self.net.inference_with_VGG19(self.data['image'])

        self.ops['loss'] = tf.reduce_mean(tf.square(self.ops['predictions']-self.data['demos']))

        self.train_writer = tf.summary.FileWriter(self.params['summary_dir'], self.sess.graph)

        # add the loss to the summary
        tf.summary.scalar('loss_totoal', self.ops['loss'])


    def initial_model(self):

        PPMIQA_list = self.net.get_resent50_var()
        loader = tf.train.Saver(var_list=PPMIQA_list)
        self.saver = tf.train.Saver()
        init_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
        self.sess.run(init_op)

        loader.restore(self.sess,self.params['resnet_ckpt'])

        # add the ops with merged
        self.ops['merged'] = tf.summary.merge_all()

    def initial_model_VGG(self):
        PPMIQA_list = self.net.get_vgg19_var()

        loader = tf.train.Saver(var_list=PPMIQA_list)
        self.saver = tf.train.Saver()
        init_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
        self.sess.run(init_op)

        loader.restore(self.sess,self.params['vgg_ckpt'])

        self.ops['merged'] = tf.summary.merge_all()

    def restore_model(self, path):
        full_path = os.path.join(path, self.params['restore_name'])
        print("Restoring weights from file %s." % full_path)
        loader = tf.train.Saver()

        loader.restore(self.sess, full_path)

        self.ops['merged'] = tf.summary.merge_all()


    def make_train_step(self):
        trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

        # set the learning_rate
        learning_rate = tf.Variable(self.params['orginal_learning_rate'], trainable=False, dtype=tf.float32)

        # set the learning_rate_decay options
        self.ops['learning_rate_decay'] = learning_rate.assign(tf.multiply(learning_rate, self.params['decay_rate']))
        self.ops['learning_rate'] = learning_rate

        # add to the summary
        tf.summary.scalar('learning_rate', self.ops['learning_rate'])

        # add encoder block weights to the hsitogram
        for multiple_var in trainable_vars:
            if 'multiple_concat' in multiple_var.name:
                tf.summary.histogram(multiple_var.name, multiple_var)

        # define the optimizer
        momOp = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=self.params['momentum'])
        train_step = momOp.minimize(self.ops['loss'], var_list=trainable_vars, global_step=self.current_epoch)

        # add to ops
        self.ops['train_step'] = train_step

    def train(self):
        total_step = 0
        outputfile = open(self.params['output_log'],'w')
        with self.graph.as_default():
            for epochs in range(self.params['epochs']):
                self.current_epoch = epochs + 1
                for corp_time in range(5):
                    self.sess.run(self.ops['train_init_op'])
                    # print(self.sess.run([self.data['image']]))

                    while True:
                        try:
                            _, loss_v, predicitons_v, summary_v, demos_v, learningrate_v = self.sess.run(
                                [self.ops['train_step'], self.ops['loss'], self.ops['predictions'],self.ops['merged'],
                                 self.data['demos'], self.ops['learning_rate']])
                            total_step += 1

                            if total_step % 20 == 0:
                                print("Train:Eochs = " + str(epochs + 1).ljust(10) + "Step = " + str(total_step).ljust(
                                    10) + "leraning_rate = " + str(learningrate_v).ljust(15) + "Loss = " + str(
                                    loss_v).ljust(15) + "demos = " + str(
                                    np.sum(demos_v[:, 0]) / self.params['batch_size']).ljust(
                                    25) + "mean_prediction = " + str(
                                    np.sum(predicitons_v[:, 0]) / self.params['batch_size']).ljust(25))
                                outputfile.write("Train:Eochs = " + str(epochs + 1).ljust(10) + "Step = " + str(total_step).ljust(
                                    10) + "leraning_rate = " + str(learningrate_v).ljust(15) + "Loss = " + str(
                                    loss_v).ljust(15) + "demos = " + str(
                                    np.sum(demos_v[:, 0]) / self.params['batch_size']).ljust(
                                    25) + "mean_prediction = " + str(
                                    np.sum(predicitons_v[:, 0]) / self.params['batch_size']).ljust(25))
                                self.train_writer.add_summary(summary_v, total_step)
                        except tf.errors.OutOfRangeError:
                            break
                if (epochs + 1) % self.params['decay_steps'] == 0:
                    self.sess.run(self.ops['learning_rate_decay'])
                self.saver.save(self.sess, self.params['save_dir'] + 'saved_' + str(epochs) + 'ckpt')


        outputfile.close()

    def test(self):
        epochs=0
        # test sequence
        predictions_total = []
        demos_total = []
        loss_total = []
        for crop_time in range(50):
            self.sess.run(self.ops['init_op'])
            predictions_each_epochs = []
            demos_each_epochs = []
            loss_each_epochs = []
            while True:
                try:
                    predictions_v, demos_v, loss_v = self.sess.run([self.ops['predictions'],
                                                                    self.data['demos'],
                                                                    self.ops['loss']])
                    print("Testing loss: ", loss_v)
                    print(predictions_v.shape, demos_v.shape)
                    for i in range(predictions_v.shape[0]):
                        predictions_each_epochs.append(predictions_v[i][0])
                        demos_each_epochs.append(demos_v[i][0])
                    loss_each_epochs.append(loss_v)

                except tf.errors.OutOfRangeError:
                    break
            # get one epochs predictions and demos
            predictions_total.append(predictions_each_epochs)
            demos_total.append(demos_each_epochs)
            loss_total.append(loss_each_epochs)

        predictions_average = []
        demos_average = []
        loss_average = []

        for x in range(len(predictions_total[0])):
            sum_p = 0
            sum_d = 0
            for y in range(len(predictions_total)):
                sum_p += predictions_total[y][x]
                sum_d += demos_total[y][x]
            predictions_average.append(sum_p / 50)
            demos_average.append(sum_d / 50)
            srcc, p_s = spearmanr(predictions_average, demos_average)
            plcc, p_p = pearsonr(predictions_average, demos_average)

        for m in range(len(loss_total[0])):
            sum_l = 0
            for n in range(len(loss_total)):
                sum_l += loss_total[n][m]
            loss_average.append(sum_l / 50)

        print("Test :Eochs = " + str(epochs + 1).ljust(10) +
              "P_s =" + str(p_s).ljust(25) +
              "SRCC = " + str(srcc).ljust(25) +
              "P_P =" + str(p_p).ljust(25) +
              "PLCC = " + str(plcc).ljust(25) +
              "mean_loss = " + str(sum_l / 50).ljust(25))

    def test_single_image(self,type):
        self.sess.run(self.ops['init_op'])
        predictions_list =[]
        while True:
            try:
                predictions_v, demos_v, loss_v= self.sess.run([self.ops['predictions'],
                                                                self.data['demos'],
                                                                self.ops['loss']])
                predictions_list.append(predictions_v)
            except tf.errors.OutOfRangeError:
                break
        predictions_array = np.asarray(predictions_list)
        print("type=%2s. predictions=%10s. gt=%10s. loss=%10s." % (type,predictions_array.mean(),demos_v[0][0],loss_v))

        return type,predictions_array.mean(),demos_v[0][0],loss_v

    def get_feature_map(self):
        self.sess.run(self.ops['init_op'])
        while True:
            try:
                block1_feature_v,block4_feature_v = self.sess.run([self.ops['feature_block1'],self.ops['feature_block4']])
            except tf.errors.OutOfRangeError:
                break
        return block1_feature_v,block4_feature_v

    def train_and_test(self):
        total_step = 0
        with self.graph.as_default():
            for epochs in range(self.params['epochs']):
                self.current_epoch = epochs + 1
                for corp_time in range(5):
                    self.sess.run(self.ops['train_init_op'])
                    # print(self.sess.run([self.data['image']]))

                    while True:
                        try:
                            _, loss_v, predicitons_v, summary_v, demos_v, learningrate_v = self.sess.run(
                                [self.ops['train_step'], self.ops['loss'], self.ops['predictions'], self.ops['merged'],
                                 self.data['demos'], self.ops['learning_rate']])
                            total_step += 1

                            if total_step % 100 == 0:
                                print("Train:Eochs = " + str(epochs + 1).ljust(2) + "Step = " + str(total_step).ljust(
                                    5) + "leraning_rate = " + str(learningrate_v).ljust(15) + "Loss = " + str(
                                    loss_v).ljust(15) + "demos = " + str(
                                    np.sum(demos_v[:, 0]) / self.params['batch_size']).ljust(
                                    25) + "mean_prediction = " + str(
                                    np.sum(predicitons_v[:, 0]) / self.params['batch_size']).ljust(25))
                                self.train_writer.add_summary(summary_v, total_step)
                        except tf.errors.OutOfRangeError:
                            break
                if (epochs + 1) % self.params['decay_steps'] == 0:
                    self.sess.run(self.ops['learning_rate_decay'])
                self.saver.save(self.sess, self.params['save_dir'] + 'saved_' + str(epochs) + 'ckpt')

                predictions_total = []
                demos_total = []
                for crop_time in range(50):
                    self.sess.run(self.ops['test_init_op'])
                    predictions_each_epochs = []
                    demos_each_epochs = []
                    while True:
                        try:
                            predictions_v, demos_v = self.sess.run([self.ops['predictions'], self.data['demos']])
                            # print(predictions_v.shape,demos_v.shape)
                            for i in range(predictions_v.shape[0]):
                                predictions_each_epochs.append(predictions_v[i][0])
                                demos_each_epochs.append(demos_v[i][0])
                        except tf.errors.OutOfRangeError:
                            break
                    # get one epochs predictions and demos
                    predictions_total.append(predictions_each_epochs)
                    demos_total.append(demos_each_epochs)

                predictions_average = []
                demos_average = []

                for x in range(len(predictions_total[0])):
                    sum_p = 0
                    sum_d = 0
                    for y in range(len(predictions_total)):
                        sum_p += predictions_total[y][x]
                        sum_d += demos_total[y][x]
                    predictions_average.append(sum_p / 50)
                    demos_average.append(sum_d / 50)
                    srcc, p_s = spearmanr(predictions_average, demos_average)
                    plcc, p_p = pearsonr(predictions_average, demos_average)

                print("Test :Eochs = " + str(1 + 1).ljust(2) +
                      "P_s =" + str(p_s).ljust(25) +
                      "SRCC = " + str(srcc).ljust(25) +
                      "P_P =" + str(p_p).ljust(25) +
                      "PLCC = " + str(plcc).ljust(25))


# model = PPMIQAmodel(123,123,123)
# model.train_and_test()

# dataset = TID2013Dataset(1, shuffle=True, crop_size=50, num_epochs=10, crop_shape=[224,224,3])
#
# ImageList = dataset.get_test_list()
# with open('/home/wangkai/logs_save/ppmiqa_tid2013_on_tid2013_type.txt','w') as file:
#     for im_name,demos,distort_type in ImageList:
#         model = PPMIQAmodel(im_name,demos,distort_type)
#         type_save,prediction_save,demos_save,loss_save=model.test_single_image(distort_type)
#         file.write(str(type_save)+" "+str(prediction_save)+" "+str(demos_save)+" "+str(loss_save)+"\n")
        # time.sleep(3)

# dataset = SynIQADataset(1, shuffle=True, crop_size=50, num_epochs=10, crop_shape=[224,224,3])
#
# ImageList = dataset.get_test_list()
# with open('/home/wangkai/logs_save/ppmiqa_syn_on_syndata_type.txt','w') as file:
#     for im_name,demos in ImageList:
#         model = PPMIQAmodel(im_name,demos,0)
#         type_save,prediction_save,demos_save,loss_save=model.test_single_image(0)
#         file.write(str(type_save)+" "+str(prediction_save)+" "+str(demos_save)+" "+str(loss_save)+"\n")

# dataset = IrccynDataset(1, shuffle=True, crop_size=50, num_epochs=10, crop_shape=[224,224,3])
#
# ImageList = dataset.get_test_list()
# with open('/home/wangkai/logs_save/ppmiqa_IRccyn_on_syndata_type.txt','w') as file:
#     for im_name,demos in ImageList:
#         model = PPMIQAmodel(im_name,demos,0)
#         type_save,prediction_save,demos_save,loss_save=model.test_single_image(0)
#         file.write(str(type_save)+" "+str(prediction_save)+" "+str(demos_save)+" "+str(loss_save)+"\n")

# dataset = TID2013Dataset(1)
#
# ImageList = dataset.get_test_list()
#
# with open('/home/wangkai/logs_save/ppmiqa_tid2013_on_syndata_type.txt','w') as file:
#     for image,demos,type in ImageList:
#         model = PPMIQAmodel(image,demos,type,'TID2013Dataset')
#         type_save,prediction_save,demos_save,loss_save = model.test_single_image(type)
#         file.write(str(type_save) + " " + str(prediction_save) + " " + str(demos_save) + " " + str(loss_save) + " "+image+"\n")

# dataset  = IETRDataset(23)
# ImageList = dataset._prase_file()
# with open('/home/wangkai/logs_save/ppmiqa_ITEA_on_syndata_type.txt','w') as file:
#     for image,demos,type in ImageList:
#         model = PPMIQAmodel(image,demos,type)
#         type_save,prediction_save,demos_save,loss_save = model.test_single_image(type)
#         file.write(str(type_save) + " " + str(prediction_save) + " " + str(demos_save) + " " + str(loss_save) + "\n")


#测每张图的预测分数
# dataset = SynIQADataset(1, shuffle=True, crop_size=50, num_epochs=10, crop_shape=[224,224,3])
#
# ImageList = dataset._prase_file()
# with open('/home/wangkai/logs_save/ppmiqa_syn_on_syndata_every_picture.txt','w') as file:
#     for im_name,demos in ImageList:
#         model = PPMIQAmodel(im_name,demos,0,'SynIQADataset')
#         type_save,prediction_save,demos_save,loss_save=model.test_single_image(0)
#         file.write(im_name+" "+str(prediction_save)+" "+str(demos_save)+" "+str(loss_save)+"\n")

# dataset = IrccynDataset(1, shuffle=True, crop_size=50, num_epochs=10, crop_shape=[224,224,3])
#
# ImageList = dataset._prase_file()
# with open('/home/wangkai/logs_save/ppmiqa_IRccyn_on_syndata_every_picture.txt','w') as file:
#     for im_name,demos in ImageList:
#         model = PPMIQAmodel(im_name,demos,0,'IrccynDataset')
#         type_save,prediction_save,demos_save,loss_save=model.test_single_image(0)
#         file.write(im_name+" "+str(prediction_save)+" "+str(demos_save)+" "+str(loss_save)+"\n")

#training for without sampling
# model = PPMIQAmodel(123,123,123,'SynIQADataset')
# model.train_and_test()

# 3.1测每张图的预测分数
# dataset = SynIQADataset(1, shuffle=True, crop_size=50, num_epochs=10, crop_shape=[224,224,3])
#
# ImageList = dataset._prase_file()
# with open('/home/wangkai/logs_save/ppmiqa_without_upsample_syn_on_syndata_every_picture_10.txt','w') as file:
#     for im_name,demos in ImageList:
#         model = PPMIQAmodel(im_name,demos,0,'SynIQADataset')
#         type_save,prediction_save,demos_save,loss_save=model.test_single_image(0)
#         file.write(im_name+" "+str(prediction_save)+" "+str(demos_save)+" "+str(loss_save)+"\n")

# 3.2测每张图的预测分数
# dataset = SynIQADataset(1, shuffle=True, crop_size=50, num_epochs=10, crop_shape=[224,224,3])
#
# ImageList = dataset._prase_file()
# with open('/home/wangkai/logs_save/ppmiqa_with_downsample_syn_on_syndata_every_picture.txt','w') as file:
#     for im_name,demos in ImageList:
#         model = PPMIQAmodel(im_name,demos,0,'SynIQADataset')
#         type_save,prediction_save,demos_save,loss_save=model.test_single_image(0)
#         file.write(im_name+" "+str(prediction_save)+" "+str(demos_save)+" "+str(loss_save)+"\n")

# 3.3测每张图的预测分数
# dataset = SynIQADataset(1, shuffle=True, crop_size=50, num_epochs=10, crop_shape=[224,224,3])
#
# ImageList = dataset._prase_file()
# with open('/home/wangkai/logs_save/ppmiqa_without_upsample_and_block1_syn_on_syndata_every_picture_10.txt','w') as file:
#     for im_name,demos in ImageList:
#         model = PPMIQAmodel(im_name,demos,0,'SynIQADataset')
#         type_save,prediction_save,demos_save,loss_save=model.test_single_image(0)
#         file.write(im_name+" "+str(prediction_save)+" "+str(demos_save)+" "+str(loss_save)+"\n")

# 3.4测每张图的预测分数
# dataset = SynIQADataset(1, shuffle=True, crop_size=50, num_epochs=10, crop_shape=[224,224,3])
#
# ImageList = dataset._prase_file()
# with open('/home/wangkai/logs_save/ppmiqa_without_upsample_and_block4_syn_on_syndata_every_picture_10.txt','w') as file:
#     for im_name,demos in ImageList:
#         model = PPMIQAmodel(im_name,demos,0,'SynIQADataset')
#         type_save,prediction_save,demos_save,loss_save=model.test_single_image(0)
#         file.write(im_name+" "+str(prediction_save)+" "+str(demos_save)+" "+str(loss_save)+"\n")

# 3.4测每张图的预测分数
dataset = SynIQADataset(1, shuffle=True, crop_size=50, num_epochs=10, crop_shape=[224,224,3])

ImageList = dataset._prase_file()
with open('/home/wangkai/logs_save/ppmiqa_vgg_syn_on_syndata_every_picture_10.txt','w') as file:
    for im_name,demos in ImageList:
        model = PPMIQAmodel(im_name,demos,0,'SynIQADataset')
        type_save,prediction_save,demos_save,loss_save=model.test_single_image(0)
        file.write(im_name+" "+str(prediction_save)+" "+str(demos_save)+" "+str(loss_save)+"\n")
# 输出特征图

# image_path = '/home/wangkai/Paper_MultiFeature_Data/syn_data/single_image/55.bmp'
# model = PPMIQAmodel(image_path,0,0,'SynIQADataset')
# block1_feature,block4_feature= model.get_feature_map()
# np.save('/home/wangkai/logs_save/feautre1_55.npy',block1_feature)
# np.save('/home/wangkai/logs_save/feautre4_55.npy',block4_feature)