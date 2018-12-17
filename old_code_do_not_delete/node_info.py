def tensor_name_v1():
    from tensorflow.python import pywrap_tensorflow
    import os
    checkpoint_path = "K:\\resnet\\resnet_v2_50.ckpt"
    reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
    var_to_shape_map = reader.get_variable_to_shape_map()
    file = open('node_info_v2.txt','w')
    for key in var_to_shape_map:
        print("tensor_name:",key)
        file.write(key+"\n")
    file.close()
        #print(reader.get_tensor(key))
def tensor_name_v2():
    import tensorflow as tf
    import os
    from tensorflow.python import pywrap_tensorflow
    ckpt = tf.train.get_checkpoint_state("K:\\resnet\\resnet_v2_50.ckpt")
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_file = os.path.basename(ckpt.model_checkpoint_path)
        print(ckpt_file)
    #reader = pywrap_tensorflow.NewCheckpointReader(os.path.join("K:\\resnet\\resnet_imagenet_v2_fp32_20181001\\",ckpt_file))
    #var_to_shape_map = reader.get_variable_to_shape_map()
    #file = open('node_v2_info.txt', 'a')
    #for key in var_to_shape_map:
        #print("tensor_name:", key)
        #file.write(key + "\n")
    #file.close()

def tensor_name_v2_new():
    from tensorflow.python.tools import inspect_checkpoint as chkp
    chkp.print_tensors_in_checkpoint_file("K:\\resnet\\resnet_v2_50.ckpt",tensor_name='', all_tensors=True)

if __name__ == '__main__':
    tensor_name_v2_new()