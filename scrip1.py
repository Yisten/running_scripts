path ="/data2/nuplan_data/2021.09.06.03.27.22_veh-53_00803_01004/0101c2beb0bd5b8f"
feature_name = ["hivt_pyg"]
target_name = ["trajectory"]
feature_path = path+"_"+feature_name[0]+".gz"

import gzip
import pickle

with gzip.open(feature_path, 'rb') as f:
    data = pickle.load(f)
a=1