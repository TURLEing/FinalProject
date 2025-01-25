import pickle
import os

with open('/home/xtz/datasets/_file/belongs_id.pickle', 'rb') as f:
    belongs = pickle.load(f)

for belong in belongs:
    data_dir = f'datasets/denoised_data/{belong}'
    # 删除 data_dir 下的 data.list 文件
    os.system(f"rm {data_dir}/data.list")