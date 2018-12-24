# _*_ coding:utf-8 _*_
# @Time     :12/21/18 5:42 PM
# @Author   :zer01ike
# @FileName : build_PPMIQA.py
# @gitHub   : https://github.com/zer01ike

from __future__ import print_function,unicode_literals

import tensorflow as tf
import numpy as np
from data.LiveIQADataset import LiveIQADataset
from nets.PPMIQA_network import PPMIQA
from scipy.stats import spearmanr
from scipy.stats import pearsonr

class PPMIQAmodel(object):
    @classmethod
    def default_params(cls):
        return {
            'root_dir': "/home/wangkai/",
            'resnet_ckpt': "/home/wangkai/Paper_MultiFeature_Data/resnet/resnet_v1_50.ckpt",
            'summary_dir': "../logs/multib1b4",
            'save_dir': "../save/multib1b4/",
            'orginal_learning_rate': 0.001,
            'decay_steps': 10,
            'decay_rate': 0.1,
            'momentum': 0.9,
            'epochs': 100,
            'corp_size': 10,
            'batch_size': 32,
            'height': 224,
            'width': 224,
            'channels': 3}

    def __init__(self):

        self.params = self.default_params()

        self.graph = tf.Graph()

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)

        self.sess = tf.Session(graph=self.graph,config=tf.ConfigProto(gpu_options=gpu_options))

        with self.graph.as_default():
            self.ops = {}
            self.data = {}
            self.get_DataSet()

            self.build_PPMIQA_net()

            self.make_train_step()

            self.initial_model()

    def get_DataSet(self):
        '''
        this is the function to get the Dataset
        :return: None But set the dict for this class
        '''

        # get the dataset from the LiveIQA Dataset
        dataset = LiveIQADataset(mode='training', batch_size=self.params['batch_size'], shuffle=True, crop_size=50,
                                 num_epochs=self.params['corp_size'],
                                 crop_shape=[self.params['height'], self.params['width'], self.params['channels']])

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

    def build_PPMIQA_net(self):
        self.net = PPMIQA()

        self.current_epoch = tf.Variable(0, name="current_epoch")

        self.ops['predictions'] = self.net.inference(self.data['image'])

        #self.ops['loss'] = tf.losses.mean_squared_error(self.ops['predictions'],self.data['demos'])
        self.ops['loss'] = tf.reduce_sum(tf.square(self.ops['predictions']-self.data['demos']))


        # set the writer for summary
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
        with self.graph.as_default():
            for epochs in range(self.params['epochs']):
                self.current_epoch = epochs + 1
                for corp_time in range(5):
                    self.sess.run(self.ops['train_init_op'])
                    #print(self.sess.run([self.data['image']]))

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
                                self.train_writer.add_summary(summary_v, total_step)
                        except tf.errors.OutOfRangeError:
                            break
                if (epochs + 1) % self.params['decay_steps'] == 0:
                    self.sess.run(self.ops['learning_rate_decay'])
                self.saver.save(self.sess, self.params['save_dir'] + 'saved_' + str(epochs) + 'ckpt')

                # test sequence
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

                print("Test :Eochs = " + str(epochs + 1).ljust(10) +
                      "P_s =" + str(p_s).ljust(25) +
                      "SRCC = " + str(srcc).ljust(25) +
                      "P_P =" + str(p_p).ljust(25) +
                      "PLCC = " + str(plcc).ljust(25))

model = PPMIQAmodel()
model.train()