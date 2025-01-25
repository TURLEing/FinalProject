# Create a class that supports selecting data from a dataset in a given path,
# such that the data meets a certain length condition, and save this data to another path.
# The relevant information of the data can be found in the `data.json` file

import os
import json
import pickle

# 更改当前文件的运行路径
os.chdir('/home/xtz/')
print(os.getcwd())

class DataSelector:
    def __init__(self, min_duration=0, max_duration=float('inf')):
        self.min_duration = min_duration
        self.max_duration = max_duration
    
    def load_data_info(self, data_dir=None):
        data_info_path = os.path.join(data_dir, "data.json")
        with open(data_info_path, "r") as f:
            data_info = json.load(f)
        return data_info
    
    # Function to get the list of selected data
    def select_data(self, data_dir=None):
        data_info = self.load_data_info(data_dir)
        selected_data = []
        for data in data_info:
            if self.min_duration <= data['Audio_Length'] <= self.max_duration:
                selected_data.append(data)
        
        return selected_data

    # 将选中的数据复制到输出目录，并保存选中数据的信息
    def copy_data(self, selected_data):
        self.save_selected_data_info(selected_data)
        for data in selected_data:
            file_name = data['File_Name']
            src_path = os.path.join(self.data_dir, file_name)
            dst_path = os.path.join(self.output_dir, file_name)
            os.system(f"cp {src_path} {dst_path}")
    
    # Dataset Format:
    # The TTS annotation .list file format:
    # vocal_path|speaker_name|language|text
    def save_selected_data_info(self, selected_data, output_dir=None):
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "data.list")

        with open(output_path, "w") as f:
            for data in selected_data:
                file_name = data['Unique_ID']
                file_path = os.path.join(output_dir, file_name)
                Speaker_ID = data['Speaker_ID']
                text = data['Paragraph']
                f.write(f"{file_path}|{Speaker_ID}|en|{text}\n")

if __name__ == '__main__':
    # Example of usage
    data_dir = "datasets/sample_data/303"
    output_dir = "datasets/text_data/303"
    min_duration = 1.0
    max_duration = 15.0

    data_selector = DataSelector(min_duration, max_duration)
    selected_data = data_selector.select_data(data_dir)
    data_selector.save_selected_data_info(selected_data, output_dir)


    # with open('/home/xtz/datasets/_file/belongs_id.pickle', 'rb') as f:
    #     belongs = pickle.load(f)

    # for belong in belongs:
    #     data_dir = f'datasets/sample_data/{belong}'
    #     output_dir = f'datasets/text_data/{belong}'
        
    #     selected_data = data_selector.select_data(data_dir)
    #     data_selector.save_selected_data_info(selected_data, output_dir)