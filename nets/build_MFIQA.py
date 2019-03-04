from __future__ import print_function, unicode_literals

import tensorflow as tf
from nets.MFIQA_network import MFIQA_network
from data.LiveIQADataset import LiveIQADataset
from data.TID2013Dataset import TID2013Dataset
from data.IETRDataset import IETRDataset
from data.SynIQADataset import SynIQADataset
import numpy as np
import os
import time
from scipy.stats import spearmanr
from scipy.stats import pearsonr


class MFIQAmodel(object):
    @classmethod
    def default_params(cls):
        '''
        this is the default_params to the params for the this model
        please set the your own params to replace the following dict
        :return: a dict with params
        '''
        return {
            'root_dir': "/home/wangkai/",
            'resnet_ckpt': "/home/wangkai/Paper_MultiFeature_Data/resnet/resnet_v1_50.ckpt",
            'summary_dir': "/home/wangkai/logs_save/logs/mfiqa_tid2013_train_test_sigmod/",
            'save_dir': "/home/wangkai/logs_save/save/mfiqa_tid2013_train_test_sigmod/",
            'orginal_learning_rate': 0.001,
            'restore_file': '/home/wangkai/logs_save/save/mfiqa_tid2013_train_test_sigmod/',
            'restore_name': 'saved_27ckpt',
            'mode': 'test_single',
            'dataset': 'TID2013Dataset',
            'train': False,
            'decay_steps': 10,
            'decay_rate': 0.1,
            'momentum': 0.9,
            'epochs': 100,
            'corp_size': 10,
            'batch_size': 32,
            'height': 224,
            'width': 224,
            'channels': 3

        }

    def __init__(self,image_name,demos,type):
        # get the params to use
        self.image_name = image_name
        self.demos = demos
        self.type = type
        self.params = self.default_params()

        # set the grpah
        self.graph = tf.Graph()

        # set the gpu options
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)

        # set the sess
        self.sess = tf.Session(graph=self.graph, config=tf.ConfigProto(gpu_options=gpu_options))

        # define the network
        with self.graph.as_default():
            # define the placeholder
            self.placeholders = {}

            # get the content of the placeholder
            # self.build_placeholders()

            # define the operations dict
            self.ops = {}

            # define the data dict
            self.data = {}
            #self.get_DataSet()
            self.get_dataSet_with_name_mode()

            # build the network
            self.build_MFIQA_net()

            # build the train options
            self.make_train_step()

            # paramater initilizer or restore model
            #self.initial_model()

            if self.params['train'] is True:
                self.initial_model()
            else:
                self.restore_model(self.params['save_dir'])

    # def build_placeholders(self):
    #     self.placeholders['X'] = tf.placeholder(tf.float32, shape=[self.params['batch_size'], self.params['height'], self.params['width'], self.params['channels']], name='x_input')
    #     self.placeholders['Y'] = tf.placeholder(tf.float32, (self.params['batch_size'], 1))

    def get_DataSet(self):
        '''
        this is the function to get the Dataset
        :return: None But set the dict for this class
        '''

        # get the dataset from the LiveIQA Dataset
        # dataset = LiveIQADataset(mode='training', batch_size=self.params['batch_size'], shuffle=True, crop_size=50,
        #                          num_epochs=self.params['corp_size'],
        #                          crop_shape=[self.params['height'], self.params['width'], self.params['channels']])
        dataset = TID2013Dataset(batch_size=self.params['batch_size'], shuffle=True, crop_size=50,
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

        else:
            assert("Error: Your should define the Mode first!")
            exit(code=25)

    def build_MFIQA_net(self):
        '''
        this is the function to build net of MFIQA
        contain the net structure and the summary witer
        :return: None
        '''

        # get the net work from MFIQA class
        self.net = MFIQA_network()

        # get the current_eopch for store the summary
        self.current_epoch = self.net.init()

        # get the predictions from the inference function
        self.ops['predictions'] = self.net.inference(self.data['image'])

        # self.ops['loss'] = tf.reduce_sum(tf.square(self.ops['predictions']-self.data['demos']))
        self.ops['loss'] = tf.losses.mean_squared_error(self.ops['predictions'], self.data['demos'])

        # set the writer for summary
        self.train_writer = tf.summary.FileWriter(self.params['summary_dir'], self.sess.graph)

        # add the loss to the summary
        tf.summary.scalar('loss_totoal', self.ops['loss'])

    def initial_model(self):
        '''
        this is the function to initial model
        * get the list of resnet 50
        * set the loader
        * set the saver
        * run init_op
        * restore paramenter of the resnet 50
        :return: None
        '''

        MFIQA_list = self.net.get_resent50_var()
        loader = tf.train.Saver(var_list=MFIQA_list)
        self.saver = tf.train.Saver()
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess.run(init_op)
        loader.restore(self.sess, self.params['resnet_ckpt'])

        # add the ops with merged
        self.ops['merged'] = tf.summary.merge_all()

    def make_train_step(self):
        '''
        define the train step
        :return: 
        '''

        # get the variable to train
        trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

        # set the learning_rate
        learning_rate = tf.Variable(self.params['orginal_learning_rate'], trainable=False, dtype=tf.float32)

        # set the learning_rate_decay options
        self.ops['learning_rate_decay'] = learning_rate.assign(tf.multiply(learning_rate, self.params['decay_rate']))
        self.ops['learning_rate'] = learning_rate

        # add to the summary
        tf.summary.scalar('learning_rate', self.ops['learning_rate'])

        # add encoder block weights to the hsitogram
        for encoder_var in trainable_vars:
            if 'encoder' in encoder_var.name:
                tf.summary.histogram(encoder_var.name, encoder_var)

        # define the optimizer
        momOp = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=self.params['momentum'])
        train_step = momOp.minimize(self.ops['loss'], var_list=trainable_vars, global_step=self.current_epoch)

        # add to ops
        self.ops['train_step'] = train_step

    def train_old_style(self):
        with self.graph.as_default():
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)
            try:
                total_step = 0
                # random crop size
                for i in range(1000000):
                    total_step += 1
                    if not coord.should_stop():
                        #         image_v,demos_v = self.sess.run([self.data['image'], self.data['demos']])
                        #         _,loss_v,predicitons_v,summary_v = self.sess.run([self.ops['train_step'],
                        # self.ops['loss'],self.ops['predictions'],self.ops['merged']],feed_dict={self.placeholders['X']:image_v,self.placeholders['Y'] : demos_v})
                        _, loss_v, predicitons_v, summary_v = self.sess.run(
                            [self.ops['train_step'], self.ops['loss'], self.ops['predictions'], self.ops['merged']])
                        # print(demos_v)
                        # print(img_v.shape)
                        # plt.figure()
                        # plt.imshow(Image.fromarray(img_v,'RGB'))
                        # plt.show()
                    if i % 20 == 0:
                        print("Step = " + str(total_step).ljust(15) +
                              "Loss = " + str(loss_v).ljust(15) +
                              "mean_prediction = " + str(predicitons_v[0][0]).ljust(20))
                        self.train_writer.add_summary(summary_v, total_step)

            except tf.errors.OutOfRangeError:
                print('Catch OutRangeError')
            finally:
                coord.request_stop()
                print('Finish reading')

            coord.join(threads)
            # for i in range(self.params['corp_size']):
            #     _,loss_v,predicitons_v =self.sess.run()
    def restore_model(self, path: str) -> None:
        full_path = os.path.join(path, self.params['restore_name'])
        print("Restoring weights from file %s." % full_path)
        loader = tf.train.Saver()

        loader.restore(self.sess, full_path)

        self.ops['merged'] = tf.summary.merge_all()

    def train_and_test(self):
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



    def test(self):
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

        print("Test :Eochs = " + str(1 + 1).ljust(10) +
              "P_s =" + str(p_s).ljust(25) +
              "SRCC = " + str(srcc).ljust(25) +
              "P_P =" + str(p_p).ljust(25) +
              "PLCC = " + str(plcc).ljust(25))

    def test_single_image(self,type):
        self.sess.run(self.ops['init_op'])
        predictions_list =[]
        while True:
            try:
                predictions_v, demos_v, loss_v = self.sess.run([self.ops['predictions'],
                                                                self.data['demos'],
                                                                self.ops['loss']])
                predictions_list.append(predictions_v)
            except tf.errors.OutOfRangeError:
                break
        predictions_array = np.asarray(predictions_list)
        print("type=%2s. predictions=%10s. gt=%10s. loss=%10s." % (type,predictions_array.mean(),demos_v[0][0],loss_v))

        return type,predictions_array.mean(),demos_v[0][0],loss_v


# model = MFIQAmodel(123,123,123)
# model.train_and_test()
dataset = TID2013Dataset(1, shuffle=True, crop_size=50, num_epochs=10, crop_shape=[224,224,3])
#
ImageList = dataset.get_test_list()
with open('/home/wangkai/logs_save/mfiqa_tid2013_sigmod_type_final.txt','w') as file:
    for im_name,demos,distort_type in ImageList:
        model = MFIQAmodel(im_name,demos,distort_type)
        type_save,prediction_save,demos_save,loss_save=model.test_single_image(distort_type)
        file.write(str(type_save)+" "+str(prediction_save)+" "+str(demos_save)+" "+str(loss_save)+"\n")
        #time.sleep(2)

# dataset = SynIQADataset(1, shuffle=True, crop_size=50, num_epochs=10, crop_shape=[224,224,3])
#
# ImageList = dataset.get_test_list()
# with open('/home/wangkai/logs_save/mfiqa_tid2013_on_tid2013_no_type.txt','w') as file:
#     for im_name,demos in ImageList:
#         model = MFIQAmodel(im_name,demos,0)
#         type_save,prediction_save,demos_save,loss_save=model.test_single_image(0)
#         file.write(str(type_save)+" "+str(prediction_save)+" "+str(demos_save)+" "+str(loss_save)+"\n")

# dataset = IETRDataset(1)
#
# ImageList = dataset._prase_file()
#
# with open('/home/wangkai/logs_save/mfiqa_itear_type.txt','w') as file:
#     for image,demos,type in ImageList:
#         model = MFIQAmodel(image,demos,type)
#         type_save,prediction_save,demos_save,loss_save = model.test_single_image(type)
#         file.write(str(type_save) + " " + str(prediction_save) + " " + str(demos_save) + " " + str(loss_save) + "\n")
