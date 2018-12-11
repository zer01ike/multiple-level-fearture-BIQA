from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
#tf.enable_eager_execution()
import tensorflow as tf
from PIL import Image
import cv2
import numpy as np
import random
import base64

tf.logging.set_verbosity(tf.logging.INFO)

#####here for LIVE Database Path#####
data_dir = "/home/wangkai/Paper_MultiFeature_Data/databaserelease2/"
IQA_name = "LIVE_IQA.txt"
######end############################

class image():
    def __init__(self,path,ref,distoration,name,demos_normal):
        self.path=path
        self.ref = ref
        self.distoration = distoration
        self.name = name
        self.demos_normal = demos_normal
    def storeName(self,patch_number,store_fmt):
        slash = "_"
        return self.ref+slash+self.distoration+slash+self.name+slash+str(patch_number)+"."+store_fmt


def preporcessing(data_dir,summary_file,meanpatch_file,train_percent,test_percent):


    X_train ,Y_train,X_test,Y_test = generateTrainTest(data_dir=data_dir,
                                                       summary_file = summary_file,
                                                       train_percent= train_percent,
                                                       test_percent = test_percent)


    meanPatch = readImage(meanpatch_file)
    meanPatch = np.array(meanPatch)

    X_train,X_test = minusMeanPtacth(X_train,X_test,meanPatch)

    return X_train, Y_train,X_test,Y_test

def generatelist(data_dir,train_dir,meanpath_dir):
    X=[]
    Y=[]
    with open(train_dir,'r') as train:
        lines = train.readlines()
        for line in lines:
            line = line.replace("\n","")
            dmos, name = line.split(" ")
            class_name = name.split("_")[0]
            X.append(data_dir+name)
            Y.append(dmos)
    meanBatch = readImage(meanpath_dir)

    return X,Y,meanBatch


def readBatchSizeImage(start,batch_size,X,Y,meanBatch):
    X_train = []
    Y_train = []

    for x,y in zip(X[start:start+batch_size],Y[start:start+batch_size]):
        X_train.append(readImage(x) - meanBatch)
        #X_train.append(readImage(x))
        Y_train.append([y])

    X_train = np.array(X_train)
    Y_train = np.array(Y_train)

    return X_train,Y_train

def generateTrainTest(data_dir,summary_file,train_percent,test_percent):
    X=[]
    Y=[]
    X_train = []
    Y_train = []
    X_test = []
    Y_test = []

    train_file = "/home/wangkai/Paper_MultiFeature_Data/databaserelease2/train.txt"
    test_file = "/home/wangkai/Paper_MultiFeature_Data/databaserelease2/test.txt"

    train_instance = open(train_file,'w')
    test_instance = open(test_file,'w')

    X_train_np = np.empty([0,224,224,3])
    X_test_np = np.empty([0,224,224,3])
    Y_train_np = np.empty([0,1])
    Y_test_np = np.empty([0,1])

    # need to make sure the ref category is ok
    # so split method is base on the percentge of ref images

    category = []

    if(train_percent + test_percent != 1) :
        print("Error: The train and test percent is not equal to 100%")
        exit(code= 5)



    #read all images from the data_dir + summary_file
    with open(summary_file,'r') as summary:
        lines = summary.readlines()
        for line in lines:
            line = line.replace("\n","")
            dmos, name = line.split(" ")
            class_name = name.split("_")[0]
            X.append(name)
            Y.append(dmos)
            category.append(class_name)
    category = set(category)

    category = list(category)
    #split the category
    indices = int(len(category) * train_percent)
    train_category = category[:indices]
    test_category = category[indices:]
    print("Category Contain:".ljust(25)+str(category))
    print("Train Category Contain:".ljust(25)+str(train_category))
    print("Test Category Contain:".ljust(25)+str(test_category))


    counter = 0
    total = len(X)

    for unsplit_x,unsplit_y in zip(X,Y):
        isTrain = True if unsplit_x.split("_")[0] in train_category else False
        isTest = True if unsplit_x.split("_")[0] in test_category else False
        if isTrain :
            X_train.append(readImage(str(data_dir+unsplit_x)))
            #X_train_np = np.append(X_train_np,readImage(str(data_dir+unsplit_x))[np.newaxis,:,:,:],axis=0)
            Y_train.append([unsplit_y])
            #y = np.array([unsplit_y])
            #Y_train_np = np.append(Y_train_np,y[np.newaxis,:],axis=0)
            train_instance.write(str(unsplit_y)+" "+unsplit_x+"\n")
        elif isTest :
            X_test.append(readImage(str(data_dir+unsplit_x)))
            #X_test_np = np.append(X_test_np,readImage(str(data_dir+unsplit_x))[np.newaxis,:,:,:],axis =0)
            Y_test.append([unsplit_y])
            #y = np.array([unsplit_y])
            #Y_test_np = np.append(Y_test_np, y[np.newaxis, :], axis=0)
            test_instance.write(str(unsplit_y) + " " + unsplit_x+"\n")
        counter +=1

        if counter % 100 == 0:
            print("read_image:".ljust(15) + str(counter).rjust(6) + "/" + str(total).ljust(10))

    # X_train = np.array(X_train)
    # Y_train = np.array(Y_train)
    # X_test = np.array(X_test)
    # Y_test = np.array(Y_test)

    train_instance.close()
    test_instance.close()
    print("X_train_size:".ljust(25)+ str(len(X_train)).ljust(8) + "Y_train_size:".ljust(25) + str(len(Y_train)))
    print("X_test_size:".ljust(25) + str(len(X_test)).ljust(8) + "Y_test_size:".ljust(25) + str(len(Y_test)))
    # print("X_train_size:".ljust(25) + str(X_train_np.shape[0]).ljust(8) + "Y_train_size:".ljust(25) + str(Y_train_np.shape[0]))
    # print("X_test_size:".ljust(25) + str(X_test_np.shape[0]).ljust(8) + "Y_test_size:".ljust(25) + str(Y_test_np.shape[0]))

    return X_train,Y_train,X_test,Y_test

def minusMeanPtacth(X_train,X_test,meanPatch):
    X_train = X_train - meanPatch
    X_test = X_test - meanPatch
    return X_train, X_test


def generateMeanpatch(patch_dir):
    # read all Patch
    import os
    Patchs = []
    for root, dirs,files in os.walk(patch_dir):
        for file in files:
            Patchs.append(os.path.join(root,file))

    sum_mean = np.zeros([3], dtype=np.float64)
    index = 0
    for patch in Patchs:
        patch_content = readImage(patch)
        sum_mean += np.mean(patch_content, axis=(0, 1))
        if (index %100 == 0):
            print("Proced:".ljust(10)+str(index+1).rjust(8)+"/"+str(len(Patchs)))
        index+=1

    total = np.array([len(Patchs),len(Patchs),len(Patchs)])

    all_mean = np.true_divide(sum_mean,total)

    average_patch = np.tile(all_mean, (224, 224, 1))

    return average_patch

def getImageInfo(data_dir,IQA_name):
    '''
    get the path of each image from IQA_name file
    :param IQA_name:
    :return: the imagePath list
    '''

    imagesinfo =[]
    parentdir = data_dir
    with open(parentdir + IQA_name) as pathfile:
        lines_pathfile = pathfile.readlines()

        # this is the pattern of the file in IQA_name.txt
        for eachline in lines_pathfile:
            single_imagepath = eachline.split(" ")[3]
            single_ref = eachline.split(" ")[2].split("/")[1].split('.')[0]
            single_distoration = single_imagepath.split("/")[0]
            single_name = single_imagepath.split("/")[1].split('.')[0]
            single_dmos_normalize = eachline.split(" ")[4]

            single_image = image(parentdir+single_imagepath,single_ref,single_distoration,single_name,single_dmos_normalize)

            imagesinfo.append(single_image)
    return imagesinfo


def generateInfoFile(data_dir,IQA_name,file_name,randomcut_number):
    imagesinfo = getImageInfo(data_dir,IQA_name)
    index = 0
    with open(data_dir+file_name,'w') as saved:
        for image in imagesinfo:
            for i in range(0,randomcut_number):
                croped_image_name = image.storeName(index % randomcut_number, 'png')
                croped_image_dmos = image.demos_normal
                index +=1
                saved.write(str(croped_image_dmos)+" "+croped_image_name+"\n")


def readImage(patch):
    image = cv2.imread(patch)

    if image is None:
        print("error Read:".ljust(10) + patch)
        exit(15)

    return image


def cropImagerandom_tf(imagesinfo,num):
    tf.set_random_seed(66)

    image_crop_en_list = []

    for image in imagesinfo:
        image_file = tf.read_file(image.path)
        image_tf = tf.image.decode_image(image_file,channels=3)
        for i in range(num):
            image_crop = tf.random_crop(image_tf,[224,224,3])
            image_crop_en_list.append(tf.image.encode_png(image_crop))


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        croped_images = sess.run(image_crop_en_list)
        index = 0
        total = len(croped_images)
        #print(len(imagesinfo))
        for result in croped_images:
            info_image = imagesinfo[index//50]
            #print(index%50)
            croped_image_name = info_image.storeName(index%50,'png')

            with open(data_dir+"Patched_data/"+croped_image_name,'wb') as stored:
                stored.write(result)

            print("croped_image:".ljust(15)+ str(index+1).rjust(6) + "/" + str(total).ljust(10) + croped_image_name)

            index +=1

def shuffleList(X,Y):
    if len(X) != len(Y):
        print("Error: The length of X and Y is Not Equal!")
        exit(25)
    random_seed = random.randint(0,100)
    random.seed(random_seed)
    random.shuffle(X)
    random.seed(random_seed)
    random.shuffle(Y)

    return X,Y

def savedasTfRecord():
    def _bytes_feature(value):
        """Returns a bytes_list from a string / byte."""
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _float_feature(value):
        """Returns a float_list from a float / double."""
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    def _int64_feature(value):
        """Returns an int64_list from a bool / enum / int / uint."""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def save(file_dir,name):
        data_dir = "/home/wangkai/Paper_MultiFeature_Data/databaserelease2/Patched_data/"
        mean_patch_file = "/home/wangkai/Paper_MultiFeature_Data/databaserelease2/average_mean.png"
        X, Y, batchfile = generatelist(data_dir, train_file, mean_patch_file)

        writer = tf.python_io.TFRecordWriter(name)

        batch = Image.open(mean_patch_file)
        batch = np.array(batch)
        count = 0
        for im_dir, demos in zip(X, Y):
            image = Image.open(im_dir)
            width,height = image.size
            count +=1
            image = np.array(image)
            image_raw = Image.fromarray(image-batch).tobytes()


            feature = {
                'height': _int64_feature(height),
                'width': _int64_feature(width),
                'channel': _int64_feature(3),
                'dmos': _float_feature(float(demos)),
                'image_raw': _bytes_feature(image_raw)
            }

            tf_example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(tf_example.SerializeToString())
            print("write:".ljust(10) + str(count)+"Down!".rjust(4))
        writer.close()

    train_name = "/home/wangkai/Paper_MultiFeature_Data/TFrecords/train.tfrecords"
    train_file = "/home/wangkai/Paper_MultiFeature_Data/databaserelease2/train.txt"
    test_name = "/home/wangkai/Paper_MultiFeature_Data/TFrecords/test.tfrecords"
    test_file = "/home/wangkai/Paper_MultiFeature_Data/databaserelease2/test.txt"

    save(train_file,train_name)
    save(test_file,test_name)


def readTfRecord():
    # TODO ::need to be down
    train_name = "/home/wangkai/Paper_MultiFeature_Data/TFrecords/train.tfrecords"
    test_name = "/home/wangkai/Paper_MultiFeature_Data/TFrecords/test.tfrecords"
    def read(name):
        record_iterator = tf.python_io.tf_record_iterator(path=name)

        train_X =[]
        train_Y =[]

        for string_record in record_iterator:
            example = tf.train.Example()
            example.ParseFromString(string_record)

            label = example.features.feature['']
        reader = tf.TFRecordReader()
        _ ,serialized_example = reader.read(name)
        features = tf.parse_single_example(serialized_example,
                                           features={
                                               'height': tf.FixedLenFeature([],tf.int64),
                                               'width': tf.FixedLenFeature([],tf.int64),
                                               'channel': tf.FixedLenFeature([],tf.int64),
                                               'dmos': tf.FixedLenFeature([],tf.float32),
                                               'image_raw': tf.FixedLenFeature([],tf.string)
                                           })




###################################test case ##############33

def test_getImageInfo():
    data_dir = "/home/wangkai/Paper_MultiFeature_Data/databaserelease2/"
    IQA_name = "LIVE_IQA.txt"
    imageinfo = getImageInfo(data_dir, IQA_name)
    print(imageinfo[0].path)
    print(imageinfo[0].storeName(40,"png"))

def test_cropImage():
    data_dir = "/home/wangkai/Paper_MultiFeature_Data/databaserelease2/"
    IQA_name = "LIVE_IQA.txt"
    imageinfo = getImageInfo(data_dir, IQA_name)

    #print(len(imageinfo))
    print("ImageInfo Readed!!!")
    cropImagerandom_tf(imageinfo[0:5],50)

def test_generateMeanpatch():
    data_dir = "/home/wangkai/Paper_MultiFeature_Data/databaserelease2/Patched_data/"
    average_image = generateMeanpatch(data_dir)
    cv2.imwrite("average_mean.png",average_image)

def test_generateinfofile():
    data_dir = "/home/wangkai/Paper_MultiFeature_Data/databaserelease2/"
    IQA_name = "LIVE_IQA.txt"
    infofile = "croped_info.txt"
    generateInfoFile(data_dir,IQA_name,infofile,50)

def test_generateTrainTest():
    data_dir = "/home/wangkai/Paper_MultiFeature_Data/databaserelease2/Patched_data/"
    summary_file = "/home/wangkai/Paper_MultiFeature_Data/databaserelease2/croped_info.txt"

    generateTrainTest(data_dir,summary_file,0.8,0.2)
def test_imageread():
    a = readImage("/home/wangkai/Paper_MultiFeature_Data/databaserelease2/Patched_data/bikes_jp2k_img15_0.png")
    b = readImage("/home/wangkai/Paper_MultiFeature_Data/databaserelease2/Patched_data/bikes_jp2k_img15_0.png")

def test_shuffleList():
    X=[1,2,3,4,5,6,7]
    Y=[11,12,13,14,15,16,17]
    print(shuffleList(X,Y))

def test_readbatchsizeimage():
    train_file = "/home/wangkai/Paper_MultiFeature_Data/databaserelease2/train.txt"
    data_dir = "/home/wangkai/Paper_MultiFeature_Data/databaserelease2/Patched_data/"
    mean_patch_file = "/home/wangkai/Paper_MultiFeature_Data/databaserelease2/average_mean.png"
    X,Y,batchfile = generatelist(data_dir, train_file, mean_patch_file)

    X_train,Y_train = readBatchSizeImage(0,8,X,Y,batchfile)
    cv2.namedWindow("test_im")
    cv2.imshow("test_im",X_train[7])
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def test_readtestImage():
    test_file = "/home/wangkai/Paper_MultiFeature_Data/databaserelease2/test.txt"
    data_dir = "/home/wangkai/Paper_MultiFeature_Data/databaserelease2/Patched_data/"
    mean_patch_file = "/home/wangkai/Paper_MultiFeature_Data/databaserelease2/average_mean.png"
    X_test, Y_test, meanBatch = generatelist(data_dir, test_file, mean_patch_file)
    X_test_np, Y_test_np = readBatchSizeImage(0, len(X_test), X_test, Y_test, meanBatch)
    cv2.namedWindow("test_image")
    for i in range(len(X_test)):
        cv2.imshow("test_image",X_test_np[i])
        cv2.waitKey(500)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    #cropImage()
    #test_generateinfofile()
    #test_generateTrainTest()
    #test_imageread()
    #test_shuffleList()
    #test_readbatchsizeimage()
    #savedasTfRecord()
    test_readtestImage()
    pass
