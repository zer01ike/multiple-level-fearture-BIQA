

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
    print("RMSE = ", sqrt(sum(squaredError) / len(squaredError)),'\n')  # 均方根误差RMSE

if __name__ == '__main__':
    root_path  = "/home/wangkai/logs_save/"
    test_file = ['ppmiqa_tid2013_on_syndata_type.txt',
                 'ppmiqa_itear_on_syndata_type.txt',
                 'ppmiqa_syn_on_syndata_type.txt',
                 'ppmiqa_IRccyn_on_syndata_type.txt',
                 'mfiqa_syn_on_syndata_type.txt',
                 'mfiqa_tid2013_sigmod_type_final.txt',
                 'ppmiqa_tid2013_on_tid2013_type.txt']
    for each_file in test_file:
        with open(root_path+each_file) as file:
            demos_list = []
            predictions_list = []
            lines = file.readlines()
            for each_line in lines:
                each_line = each_line.strip("\n")
                result = each_line.split(" ")
                predictions_list.append(float(result[1]))
                demos_list.append(float(result[2]))

            rmse(demos_list,predictions_list)

