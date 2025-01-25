# Create a class that supports inputting a folder (containing multiple wav files), denoising them, and outputting to a specified folder
# Denoising process: call the cmd-denoise.py script from GPT_SoVits

import os
import argparse
import pickle
import traceback
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from tqdm import tqdm


# 更改当前文件的运行路径
os.chdir('/home/xtz/GPT-SoVITS')
print(os.getcwd())

class DataDenoiser:
    def __init__(self, model_path='tools/denoise-model/speech_frcrn_ans_cirm_16k'):
        self.model_path = model_path if os.path.exists(model_path) else "damo/speech_frcrn_ans_cirm_16k"
        self.ans = pipeline(Tasks.acoustic_noise_suppression, model=self.model_path)

    def execute_denoise(self, input_folder, output_folder):
        os.makedirs(output_folder, exist_ok=True)
        for name in tqdm(os.listdir(input_folder)):
            try:
                self.ans(os.path.join(input_folder, name), output_path=os.path.join(output_folder, name))
            except:
                traceback.print_exc()

if __name__ == '__main__':

    with open('/home/xtz/datasets/_file/belongs_id.pickle', 'rb') as f:
        belongs = pickle.load(f)

    denoiser = DataDenoiser()
    for belong in belongs:
        input_folder = f'/home/xtz/datasets/sample_data/{belong}'
        output_folder = f'/home/xtz/datasets/denoised_data/{belong}'
        denoiser.execute_denoise(
            input_folder=input_folder,
            output_folder=output_folder,
        )
