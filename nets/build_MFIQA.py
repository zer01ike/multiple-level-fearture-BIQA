from __future__ import print_function,unicode_literals

import tensorflow as tf
from nets.MFIQA_network import MFIQA_network


class MFIQAmodel(object):
    @classmethod
    def default_params(cls):
        return{
            'root_dir':"/home/wangkai/",
            'resnet_ckpt':"/home/wangkai/disk_seg/Paper_MultiFeature_Data/resnet/resnet_v1_50.ckpt",
            'orginal_learing_rate':0.0001,
            'decay_steps':10,
            'decay_rate':0.1,
            'momentum':0.9,
            'num_epochs':10

        }
    
    def __init__(self):
        self.params = self.default_params()

        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph,config=tf.ConfigProto(allow_soft_placement=True))

        with self.graph.as_default():
            self.placeholders = {}

            #define the operations
            self.ops = {}

            #build the network
            self.build_MFIQA_net()
            

            #build the train options
            self.make_train_step()

            #paramater initilizer or restore model
            self.initial_model()

    def build_MFIQA_net(self):
        net = MFIQA_network()
        self.current_epoch,self.MFIQA_list = net.init()

        self.ops['predictions'] = net.inference(self.placeholders['inputs'])
        self.ops['loss'] = tf.reduce_sum(tf.square(self.ops['predictions']-self.data['inputs_Y']))
        
    
    def initial_model(self):
        saver = tf.train.Saver(var_list=self.MFIQA_list)
        self.sess.run(tf.global_variables_initializer())
        saver.restore(self.sess,self.params['resnet_ckpt'])
    
    def make_train_step(self):
        trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        learing_rate = tf.train.exponential_decay(self.params['orginal_learing_rate'],self.current_epoch,decay_steps = self.params['decay_steps'],decay_rate = self.params['decay_rate'])
        momOp = tf.train.MomentumOptimizer(learing_rate = learing_rate,momentum = self.params['momentum'])
        train_step = momOp.minimize(self.ops['loss'],var_list=trainable_vars,global_step=self.current_epoch)
        self.ops['train_step'] = train_step
    def train(self):
        with self.graph.as_default():
            tf.train.start_queue_runners(sess=self.sess)
            for i in range(self.params['num_epochs']):
                _,loss_v,predicitons_v =self.sess.run([self.ops['train_step'],
                self.ops['loss'],self.ops['predictions']])


model = MFIQAmodel()
MFIQAmodel.train()
