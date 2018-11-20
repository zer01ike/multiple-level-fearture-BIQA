def tensor_name_v1():
    from tensorflow.python import pywrap_tensorflow
    import os
    checkpoint_path = "K:\\resnet\\resnet_v1_50.ckpt"
    reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
    var_to_shape_map = reader.get_variable_to_shape_map()
    file = open('node_info.txt','a')
    for key in var_to_shape_map:
        print("tensor_name:",key)
        file.write(key+"\n")
    file.close()
        #print(reader.get_tensor(key))
def tensor_name_v2():
    import tensorflow as tf
    checkpoint_path = "K:\\resnet\\resnet_v1_50.ckpt"
    saver = tf.train.Saver()
    file = open("node_info.txt")
    with tf.Session() as sess:
        saver.restore(sess, checkpoint_path)
        for n in tf.get_default_graph.as_graph_def().node:
            file.write(n)
    file.close()

if __name__ == '__main__':
    tensor_name_v1()