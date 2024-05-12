import os
from pathlib import Path

feature_name = 'pdm_nonreactive_trajectory'
root = '/home/guojiayu/nuplan_data'
find_path = False
types = ['05','06','07','08','09','10']
# if find_path:
#     for type in types:
#         target_path = os.path.join('./data/',type+'.txt')
#         search_path = os.path.join(root,type)
#         with open(target_path, "w") as file:
#             lognames = os.listdir(search_path)
#             for logname in lognames:
#                 file.write(logname + "\n")
# else:
import shutil
target_root = '/data2/nuplan_data'
cache_train_pdm_path = './data/cache_train_pdm.txt'
cache_val_pdm_path = './data/cache_val_pdm.txt'
path_ls = [cache_train_pdm_path, cache_val_pdm_path]
logname_token_ls = []
for path in path_ls:
    with open(path, "r") as f:
        while True:
            line = f.readline()
            if not line:
                break;
            logname_token_ls.append(line.strip())
source_root = '/home/guojiayu/nuplan_data/'
# source_root = '/rbs/guojy/'
target_root = '/data2/nuplan_data/'
# target_root = '/rbs/guojy/ppm/'
def move_files(i, local_logname_token_ls):
    for logname_token in local_logname_token_ls:
        month = logname_token[5:7]
        file_name = os.path.join(month,logname_token+'_'+feature_name+'.gz')
        source_file_path = os.path.join(source_root, file_name)
        target_file_path = os.path.join(target_root, logname_token+'_'+feature_name+'.gz')
        shutil.copy(source_file_path ,target_file_path)

import multiprocessing
num_logs = len(logname_token_ls)
num_process = 12
num_logs_each_process = (num_logs/num_process)+1
p = multiprocessing.Pool(num_process)
res = []
for i in range(num_process):
    start = int(i*num_logs_each_process)
    end = int((i+1)*num_logs_each_process)
    res.append(p.apply_async(move_files,args=(i,logname_token_ls[start:end],)))
dl_path = []
for _res in res:
    dl_path+=_res.get()
p.close()
p.join()