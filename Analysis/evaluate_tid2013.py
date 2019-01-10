from scipy.stats import spearmanr
from scipy.stats import pearsonr

result = []

with open('/home/wangkai/logs_save/mfiqa_tid2013_type.txt') as file:
    lines = file.readlines()
    for line in lines:
        line = line.strip("\n")
        result.append(line.split(" "))

#print(result)

def get_result(type):
    #type = 5
    demos_list=[]
    predictions_list=[]

    for one_result in result:
        if int(one_result[0]) == type:
           predictions_list.append(float(one_result[1]))
           demos_list.append(float(one_result[2]))

    #print(predictions_list)

    srcc, p_s = spearmanr(predictions_list, demos_list)
    plcc, p_p = pearsonr(predictions_list, demos_list)

    print("type=%2s. PLCC = %10s. SROCC = %10s." %(type,plcc,srcc))

type_list = []
for one_result in result:
    type_list.append(int(one_result[0]))
type_list =set(type_list)

type_list = list(type_list)

for single_type in type_list:
    get_result(single_type)



