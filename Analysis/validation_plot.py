import matplotlib.pyplot as plt

root_path = '/home/wangkai/logs_save/'

file_list = ['tid2013_type.txt','ppmiqa_itear_on_syndata_type.txt','mfiqa_tid2013_sigmod_type_final.txt','ppmiqa_itear_type.txt',
             'ppmiqa_tid2013_on_syndata_type.txt']
title_list = ['PPMIQA Train on TID2013 evaluate on TID2013 Dataset','PPMIQA Train on SynData evaluate on IETR',
              'MFIQA Train on TID2013 evaluate on TID2013 Dataset(sigmod)','PPMIQA Train on TID2013 evaluate on IETR',
              'PPMIQA Train on SynData evaluate on TID2013']

def _parse_file(path):
    result = []

    with open(path) as file:
        lines = file.readlines()
        for line in lines:
            line = line.strip("\n")
            result.append(line.split(" "))
    demos_list = []
    predictions_list = []
    for one_result in result:
        predictions_list.append(float(one_result[1]))
        demos_list.append(float(one_result[2]))
    return demos_list,predictions_list

# type prediction gt


#plt.title('Train on TID2013 evaluate on TID2013 Dataset')
for i in range(0,len(file_list)):
    plt.subplot(len(file_list),1,i+1)
    plt.title(title_list[i])
    demos_list, predictions_list = _parse_file(root_path+file_list[i])
    plt.scatter(demos_list, predictions_list)

plt.show()



# def plt_graph(title,index,prediction_list,demos_list):
#     plt.subplot()
#     plt.scatter(prediction_list,demos_list)
#     plt.show()