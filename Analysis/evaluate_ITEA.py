from scipy.stats import spearmanr
from scipy.stats import pearsonr

result = []

with open('/home/wangkai/logs_save/ppmiqa_vgg_syn_on_syndata_every_picture_10.txt') as file:
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
    return sqrt(sum(squaredError) / len(squaredError))  # 均方根误差RMSE

def get_result(type):
    #type = 5
    demos_list=[]
    predictions_list=[]

    for one_result in result:
        if one_result[0] == type:
           predictions_list.append(float(one_result[1]))
           demos_list.append(float(one_result[2]))

    #print(predictions_list)

    # print(predictions_list)
    # print(demos_list)

    srcc, p_s = spearmanr(predictions_list, demos_list)
    plcc, p_p = pearsonr(predictions_list, demos_list)
    rmse_result = rmse(predictions_list,demos_list)

    print("rmse=%10s. PLCC = %10s. SROCC = %10s." %(rmse_result,plcc,srcc))

def all_result():
    demos_list = []
    predictions_list = []
    for one_result in result:
        predictions_list.append(float(one_result[1]))
        demos_list.append(float(one_result[2]))
    print(predictions_list)
    print(demos_list)
    srcc, p_s = spearmanr(predictions_list, demos_list)
    plcc, p_p = pearsonr(predictions_list, demos_list)
    rmse_result = rmse(predictions_list, demos_list)

    print("PLCC = %10s. SROCC = %10s. RMSE = %10s." % (plcc, srcc,rmse_result))

# type_list = []
# for one_result in result:
#     type_list.append(one_result[0])
# type_list =set(type_list)
#
# type_list = list(type_list)
#
# for single_type in type_list:
#     get_result(single_type)

all_result()



