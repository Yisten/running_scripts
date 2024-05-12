# from data.dataset import ProcessedDataset
import os
import hydra
from hydra.utils import instantiate
CONFIG_PATH = "config"
CONFIG_NAME = "nuplan.yaml"

def search_lognames_and_token(i, root, log_list,feature_names,target_names,log_splitter):
    train_path_list,val_path_list = [], []
    for log_name in log_list:
        dir_path = os.path.join(root, log_name)
        file_names = os.listdir(dir_path)
        dict = {}
        for file_name in file_names:
            token = file_name[:16]
            if token not in dict.keys():
                dict[token] = []

            dict[token].append(file_name[17:-3])
        
        valid_token_list = []
        for key, item in dict.items():
            
            logits = [(feature_name in item) for feature_name in feature_names]+\
                    [(target_name in item) for target_name in target_names]
            if all(logits):
                valid_token_list.append(key)
        temp_list = [os.path.join(log_name,valid_token) for valid_token in valid_token_list]
        if log_name in log_splitter.train_logs:
            train_path_list +=temp_list
        elif log_name in log_splitter.val_logs:

            val_path_list += temp_list
    return train_path_list, val_path_list

@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def main(cfg):
    feature_names = ["hivt_pyg"]
    target_names = ["trajectory","pdm_nonreactive_trajectory"]
    data_root = "/data2/nuplan_data/"
    log_splitter = instantiate(cfg)
    log_names = os.listdir(data_root)

    import multiprocessing
    num_logs = len(log_names)
    num_process = 12
    num_logs_each_process = (num_logs/num_process)+1
    p = multiprocessing.Pool(num_process)
    res = []
    for i in range(num_process):
        start = int(i*num_logs_each_process)
        end = int((i+1)*num_logs_each_process)
        res.append(p.apply_async(
            search_lognames_and_token,
            args=(i,data_root,log_names[start:end],feature_names,target_names,log_splitter))
            )
    train_list, val_list = [],[]
    for _res in res:
        _train_paths, _val_paths =_res.get()
        train_list += _train_paths
        val_list += _val_paths
    p.close()
    p.join()
    with open("./cache_train.txt", "w") as file:
        for path in train_list:
            file.write(path+"\n")
    with open("./cache_val.txt", "w") as file:
        for path in val_list:
            file.write(path+"\n") 
    a=1
if __name__ == '__main__':
    main()
# search_lognames_and_token(data_root,log_list,feature_names,target_names)
# ProcessedDataset(data_root,)
