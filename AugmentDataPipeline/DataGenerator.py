import subprocess
import os
import json
from time import sleep
from augment_audio_gen import augment_audio_generation, tts_stop, set_gpt_weights, set_sovits_weights
from augment_text_gen import text_generation
from DataFormator import read_participant_id, find_model_path

# 创建一个 DataGenerator 类，用于生成数据集。
# 要求1：可以开启模型服务（python api_v2.py -a 127.0.0.1 -p 9880 ）
# 要求3：可以停止模型服务（import tts_stop from augment_audio_gen.py）
# 要求4：可以调用GPT-4生成文本（import text_generation from augment_text_gen.py）
# 要求2：可以调用TTS模型生成音频（import tts_request from augment_audio_gen.py）
# 针对每个语音模型，生成对应的文本，然后调用TTS模型生成音频，最后将音频和文本保存到指定路径。

# 更改当前文件的运行路径
os.chdir('/home/xtz')
print(os.getcwd())

class DataGenerator:
    def __init__(self,  model_service_host='127.0.0.1', model_service_port=9880):
        self.model_service_host = model_service_host
        self.model_service_port = model_service_port
        self.model_service_process = None

        self.GPT_model_path = None
        self.SoVITs_model_path = None
        self.ref_audio_path = None
        self.prompt_text = None
        self.spk_id = -1

    def set_model_info(self, spk_id, model_info_path="/home/xtz/datasets/_file/model_list.json"):
        self.spk_id = spk_id

        # 从 /home/xtz/datasets/_file/model_list.json 中读取 spk_id 的信息
        with open(model_info_path, "r") as f:
            model_list = json.load(f)
        
        self.GPT_model_path = None
        self.SoVITs_model_path = None
        self.ref_audio_path = None
        self.prompt_text = None
        
        for model in model_list:
            if model["SpeakerID"] == spk_id:
                self.GPT_model_path = model["GPT_model_path"]
                self.SoVITs_model_path = model["SoVITs_model_path"]
                self.ref_audio_path = model["ref_audio_path"]
                self.prompt_text = model["ref_audio_text"]
                break
        
        if self.ref_audio_path is None:
            print(f"Ref audio path not found for speaker {spk_id}.")

    def start_model_service(self):
        if self.model_service_process is None:
            self.model_service_process = subprocess.Popen(
                ['python', 'GPT-SoVITS/api_v2.py', 
                 '-a', self.model_service_host, 
                 '-p', str(self.model_service_port)
                #  '-t', self.GPT_model_path,
                #  '-v', self.SoVITs_model_path 
                ]
            )
        else:
            print("Model Service is already running.")

    def stop_model_service(self):
        if self.model_service_process is not None:
            flag = tts_stop()
            if flag:
                self.model_service_process.terminate()
                self.model_service_process = None
                print("Model Stoped Successfully.")
            else:
                print("Model Stoped Failed.")
        else:
            print("Model Service is not running.")

    def switch_model_service(self):
        print(f"Switching model service to {self.spk_id}...")
        if self.model_service_process is not None:
            set_gpt_weights(self.GPT_model_path)
            set_sovits_weights(self.SoVITs_model_path)
        else:
            print("Model Service is not running.")


    def augment_audio_generation(self, generated_texts=None):
        augment_audio_generation(self.spk_id, self.ref_audio_path, self.prompt_text, generated_texts)
        print(f"{self.spk_id} audio generation finished.")

    def generate_text(self):
        return text_generation()

if __name__ == "__main__":

    # Create a DataGenerator instance
    generator = DataGenerator()
    generator.start_model_service()
    sleep(10)  # 等待10秒，确保模型服务已经启动

    spk_ids = read_participant_id("/home/xtz/datasets/_file/train_split_Depression_AVEC2017.csv")

    for spk_id in spk_ids:
        if spk_id <= 385: continue
        generator.set_model_info(spk_id)
        generator.switch_model_service()
        sleep(10)  # 等待10秒，确保模型服务已经启动

        generated_texts = generator.generate_text()
        generator.augment_audio_generation(generated_texts)
    # generator = DataGenerator()
    # generator.start_model_service()
    # sleep(10)

    # generator.set_model_info(385)
    # sleep(5)
    # generator.augment_audio_generation()
