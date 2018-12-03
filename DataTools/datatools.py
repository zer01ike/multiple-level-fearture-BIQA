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


def preporcessing(data_dir,summary_file,meanpatch_file,train_percent,test_percent):


    X_train ,Y_train,X_test,Y_test = generateTrainTest(data_dir=data_dir,
                                                       summary_file = summary_file,
                                                       train_percent= train_percent,
                                                       test_percent = test_percent)


    meanPatch = readImage(meanpatch_file)
    meanPatch = np.array(meanPatch)

    X_train,X_test = minusMeanPtacth(X_train,X_test,meanPatch)

    return X_train, Y_train,X_test,Y_test

def generateTrainTest(data_dir,summary_file,train_percent,test_percent):
    X=[]
    Y=[]
    X_train = []
    Y_train = []
    X_test = []
    Y_test = []

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

    for unsplit_x,unsplit_y in zip(X,Y):
        isTrain = True if unsplit_x.split("_")[0] in train_category else False
        isTest = True if unsplit_x.split("_")[0] in test_category else False
        if isTrain :
            X_train.append(readImage(str(data_dir+unsplit_x)))
            Y_train.append(unsplit_y)
        elif isTest :
            X_test.append(readImage(str(data_dir+unsplit_x)))
            Y_test.append(unsplit_y)

    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    X_test = np.array(X_test)
    Y_test = np.array(Y_test)

    print("X_train_size:".ljust(25)+str(len(X_train)) +"  "+ str(X_train.shape) )
    print("Y_train_size:".ljust(25) + str(len(Y_train))+"  "+ str(Y_train.shape))
    print("X_test_size:".ljust(25) + str(len(X_test))+"  "+ str(X_test.shape))
    print("Y_test_size:".ljust(25) + str(len(Y_test))+"  "+ str(Y_test.shape))
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

if __name__ == '__main__':
    #cropImage()
    #test_generateinfofile()
    #test_generateTrainTest()
    #test_imageread()
    pass
