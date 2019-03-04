import numpy as np
import matplotlib.pyplot as plt
block1_feature = np.load('/home/wangkai/logs_save/feautre1_55.npy')
block4_feature = np.load('/home/wangkai/logs_save/feautre4.npy')
#print(block1_feature.shape)
block1_feature_each_channel =[]
block4_feature_each_channel =[]
for i in range(256):
    block1_feature_each_channel.append(block1_feature[0,:,:,i])
# print(block1_feature_each_channel[0])

for j in range(2048):
    block4_feature_each_channel.append(block4_feature[0, :, :, j])

# current = 0
# for each_image in block1_feature_each_channel:
#     plt.figure('Image')
#     current +=1
#     plt.imshow(each_image,cmap='gray')
#     #plt.title('image'+str(current))
#     plt.savefig('/home/wangkai/logs_save/feature_map_55_gray/block1_feature_'+str(current)+'.png')
#     print(str(current)+"::Feature_1")

current = 0
for i in range(1,len(block4_feature_each_channel)+1):
#for each_image in block4_feature_each_channel:
    plt.figure('Image')
    current +=1
    plt.imshow(block4_feature_each_channel[i],cmap='gray')
    #plt.title('image'+str(current))
    plt.savefig('/home/wangkai/logs_save/feature_map_55_gray/block4_feature_'+str(current)+'.png')
    #plt.show()
    print(str(i) + "::Feature_4")

# for each_image in block4_feature_each_channel:
#     print(each_image)
# plt.figure('Image')
# plt.imshow(block1_feature_each_channel[16],cmap='gray')
# plt.show()