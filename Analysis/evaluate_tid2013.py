from scipy.stats import spearmanr
from scipy.stats import pearsonr

result = []

with open('/home/wangkai/logs_save/mfiqa_tid2013_sigmod_type_final.txt') as file:
    lines = file.readlines()
    for line in lines:
        line = line.strip("\n")
        result.append(line.split(" "))

#print(result)

def rmse(target,prediction):
    error = []
    for i in range(len(target)):
        error.append(target[i] - prediction[i])

    squaredError = []
    absError = []
    for val in error:
        squaredError.append(val * val)  # target-prediction之差平方
        absError.append(abs(val))  # 误差绝对值
    from math import sqrt
    print("RMSE = ", sqrt(sum(squaredError) / len(squaredError)))

def get_result(type):
    #type = 5
    demos_list=[]
    predictions_list=[]

    for one_result in result:
        if int(one_result[0]) == type:
           predictions_list.append(float(one_result[1]))
           demos_list.append(float(one_result[2]))

    #print(predictions_list)

    rmse(demos_list,predictions_list)

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



