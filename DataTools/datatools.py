import tensorflow as tf
import cv2
import numpy as np

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


def generateMeanImage():
    pass


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

        for eachline in lines_pathfile:
            single_imagepath = eachline.split(" ")[3]
            single_ref = eachline.split(" ")[2].split("/")[1].split('.')[0]
            single_distoration = single_imagepath.split("/")[0]
            single_name = single_imagepath.split("/")[1].split('.')[0]
            single_dmos_normalize = eachline.split(" ")[4]

            single_image = image(parentdir+single_imagepath,single_ref,single_distoration,single_name,single_dmos_normalize)

            imagesinfo.append(single_image)
    return imagesinfo


def getImagePatch():
    pass

def readImage(imagespath):
    import cv2

    pass

def readLabel():
    pass

def checkConsistency():
    pass

def saveImage():
    pass

def saveLabel():
    pass

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


def cropImagerandom():
    pass

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


def cropImage():
    data_dir = "/home/wangkai/Paper_MultiFeature_Data/databaserelease2/"
    IQA_name = "LIVE_IQA.txt"
    imageinfo = getImageInfo(data_dir, IQA_name)
    cropImagerandom_tf(imageinfo,50)





if __name__ == '__main__':
    cropImage()
