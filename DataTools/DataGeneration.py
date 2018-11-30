import tensorflow as tf
import cv2
import numpy as np
data_dir = "J:\\databaserelease2\\"
IQA_name = "LIVE_IQA.txt"

def image_score():
    image=[]
    score=[]

    with open(data_dir+IQA_name) as f:
        lines = f.readlines()

        for line in lines:
            imagename = line.split(" ")[3]
            imagescore = line.split(" ")[4]
            image.append(imagename)
            score.append(imagescore)
    return image, score

def random_crop_image(image_file,num,start):
    #with tf.Graph.as_default():
    tf.set_random_seed(66)
    file_contents = tf.read_file(image_file)
    image = tf.image.decode_image(file_contents,channels=3)
    image_crop_en_list = []

    for i in range(num):
        image_crop = tf.random_crop(image,[224,224,3])
        image_crop_en_list.append(tf.image.encode_png(image_crop))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        results = sess.run(image_crop_en_list)
        for idx,re in enumerate(results):
            with open(data_dir+"total_data\\"+str(start)+'.png','wb') as f:
                f.write(re)
            start+=1
    return start

def datageneration():
    x_train=[]
    y_train=[]
    x_test =[]
    y_test =[]

    file_dir = "J:\\databaserelease2\\total_data\\"
    iqa_dir = "J:\\databaserelease2\\score.txt"

    file_num = 1000
    image_list =[]
    score_list =[]
    with open(iqa_dir) as f:
        line = f.readline()
        iqa_list = line.split(" ")

    for i in range(0,file_num):
        score_list.append([iqa_list[i]])

    for index in range(0,file_num):
        file = file_dir+str(i)+".png"
        image_list.append(cv2.imread(file))

    x_train = np.array(image_list)
    y_train = np.array(score_list)

    return x_train,y_train

def get_all_mean_val(dir,file_num):
    mean = np.zeros([3],dtype=np.float32)

    for i in range(0,file_num):
        image_each = dir+str(i)+".png"
        mean +=np.mean(image_each,axis=(0,1))




if __name__ == '__main__':
    # image,score = image_score()
    # score_list = []
    # start = 0
    # for x, y in zip(image, score):
    #     start = random_crop_image(data_dir+x, 5, start)
    #     for i in range(0, 5):
    #         score_list.append(y)
    #
    # with open(data_dir+"score.txt", 'w') as f:
    #     for i in score_list:
    #         f.write(str(i)+" ")

    x_train,y_train =datageneration()
    print(x_train.shape)
    print(y_train.shape)
