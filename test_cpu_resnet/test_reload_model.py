import tensorflow as tf

sess = tf.Session()
saver = tf.train.import_meta_graph('./checkpoint_dir/MyModel.meta')
saver.restore(sess,tf.train.latest_checkpoint('./checkpoint_dir'))
print(sess.run('w1:0'))
graph = tf.get_default_graph()
w1 = graph.get_tensor_by_name("w1:0")
w2 = graph.get_tensor_by_name("w2:0")
