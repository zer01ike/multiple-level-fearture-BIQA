from __future__ import print_function,unicode_literals

import tensorflow as tf
from nets.MFIQA_network import MFIQA_network
from data.LiveIQADataset import LiveIQADataset


class MFIQAmodel(object):
    @classmethod
    def default_params(cls):
        return{
            'root_dir': "/home/wangkai/",
            'resnet_ckpt': "/home/wangkai/Paper_MultiFeature_Data/resnet/resnet_v1_50.ckpt",
            'summary_dir': "../logs/batch128epochs40",
            'orginal_learing_rate': 0.0001,
            'decay_steps': 1,
            'decay_rate': 0.1,
            'momentum': 0.9,
            'num_epochs': 10,
            'batch_size': 8,
            'height': 224,
            'width': 224,
            'channels': 3

        }
    
    def __init__(self):
        self.params = self.default_params()

        self.graph = tf.Graph()
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction =0.5)
        self.sess = tf.Session(graph=self.graph,config=tf.ConfigProto(gpu_options = gpu_options))

        with self.graph.as_default():
            self.placeholders = {}

            self.train_writer = tf.summary.FileWriter(self.params['summary_dir'],self.sess.graph)
            self.build_placehoders()

            #define the operations
            self.ops = {}

            self.data = {}
            self.get_DataSet()

            #build the network
            self.build_MFIQA_net()
            

            #build the train options
            self.make_train_step()

            #paramater initilizer or restore model
            self.initial_model()
    def build_placehoders(self):
        self.placeholders['X'] = tf.placeholder(tf.float32, shape=[self.params['batch_size'], self.params['height'], self.params['width'], self.params['channels']], name='x_input')
        self.placeholders['Y'] = tf.placeholder(tf.float32, (self.params['batch_size'], 1))

    def get_DataSet(self):
        dataset = LiveIQADataset(mode='training',batch_size=self.params['batch_size'],shuffle=True,crop_size=50, num_epochs=self.params['num_epochs'],crop_shape=[self.params['height'], self.params['width'], self.params['channels']])
        self.data['demos'], self.data['image'] = dataset.get()


    def build_MFIQA_net(self):
        self.net = MFIQA_network()
        self.current_epoch = self.net.init()

        self.ops['predictions'] = self.net.inference(self.data['image'])
        self.ops['loss'] = tf.reduce_sum(tf.square(self.ops['predictions']-self.data['demos']))

        tf.summary.scalar('loss_totoal',self.ops['loss'])
        self.ops['merged'] = tf.summary.merge_all()
        
    
    def initial_model(self):
        MFIQA_list = self.net.get_resent50_var()
        loader = tf.train.Saver(var_list=MFIQA_list)
        # for i in MFIQA_list:
        #     print(i)
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess.run(init_op)
        loader.restore(self.sess, self.params['resnet_ckpt'])
    
    def make_train_step(self):
        trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        learing_rate = tf.train.exponential_decay(self.params['orginal_learing_rate'],self.current_epoch,decay_steps = self.params['decay_steps'],decay_rate = self.params['decay_rate'])
        momOp = tf.train.MomentumOptimizer(learning_rate=learing_rate,momentum = self.params['momentum'])
        train_step = momOp.minimize(self.ops['loss'],var_list=trainable_vars,global_step=self.current_epoch)
        self.ops['train_step'] = train_step

    # def variable_summaries(self,var):
    #     with tf.name_scope('summaries'):


    def train(self):
        with self.graph.as_default():
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=self.sess,coord=coord)
            try:
                total_step = 0
                #random crop size
                for i in range(1000000):
                    total_step +=1
                    if not coord.should_stop():
                #         image_v,demos_v = self.sess.run([self.data['image'], self.data['demos']])
                #         _,loss_v,predicitons_v,summary_v = self.sess.run([self.ops['train_step'],
                # self.ops['loss'],self.ops['predictions'],self.ops['merged']],feed_dict={self.placeholders['X']:image_v,self.placeholders['Y'] : demos_v})
                        _,loss_v,predicitons_v,summary_v = self.sess.run([self.ops['train_step'],self.ops['loss'],self.ops['predictions'], self.ops['merged']])
                        # print(demos_v)
                        # print(img_v.shape)
                        # plt.figure()
                        # plt.imshow(Image.fromarray(img_v,'RGB'))
                        # plt.show()
                    if i%20 == 0:
                        print("Step = " + str(total_step).ljust(15) +
                              "Loss = " + str(loss_v).ljust(15) +
                              "mean_prediction = " + str(predicitons_v[0][0]).ljust(20))
                        self.train_writer.add_summary(summary_v,total_step)

            except tf.errors.OutOfRangeError:
                print('Catch OutRangeError')
            finally:
                coord.request_stop()
                print('Finish reading')

            coord.join(threads)
            # for i in range(self.params['num_epochs']):
            #     _,loss_v,predicitons_v =self.sess.run()


model = MFIQAmodel()
model.train()
