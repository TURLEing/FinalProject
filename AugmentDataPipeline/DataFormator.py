# 整理所有语音模型(GPT，SoVITs)的路径
# 并从 sample_data_path 中挑选一个满足时长要求的（3<=时长<=10s）语音文件，作为 ref_audio_path.
# 将上述检索信息保存在 data_generated_list.json 中,
# 属性：SpeakerID, GPT_model_path, SoVITs_model_path, ref_audio_path, ref_audio_text.

import os
import json
import random

# 0. 从 train_split_Depression_AVEC2017.csv 读取所有 Participant_ID
def read_participant_id(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()
    participant_ids = []
    for line in lines[1:]:
        participant_ids.append(int(line.split(",")[0]))
    return participant_ids

# 1. 整理所有语音模型(GPT，SoVITs)的路径
def find_model_path(spk_ids):

    model_info = []

    Gpt_base_path = "/home/xtz/GPT-SoVITS/GPT_weights_v2"
    SoVITs_base_path = "/home/xtz/GPT-SoVITS/SoVITS_weights_v2"

    for spk_id in spk_ids: 
        Gpt_model_path = os.path.join(Gpt_base_path, f"{spk_id}-e20.ckpt")
        
        def find_SoVITs_model_path(SoVITs_base_path, spk_id):
            for file in os.listdir(SoVITs_base_path):
                if file.startswith(str(spk_id)):
                    return os.path.join(SoVITs_base_path, file)

            print(f"Model not found for speaker {spk_id}")
            return None

        SoVITs_model_path = find_SoVITs_model_path(SoVITs_base_path, spk_id)

        if os.path.exists(Gpt_model_path) and os.path.exists(SoVITs_model_path):
            model_info.append({
                "SpeakerID": spk_id,
                "GPT_model_path": Gpt_model_path,
                "SoVITs_model_path": SoVITs_model_path,
                "ref_audio_path": None,
                "ref_audio_text": None
            })
        else :
            print(f"Model not found for speaker {spk_id}")
    return model_info

# 2. 从 data.json 中挑选一个满足时长要求的（3<=时长<=10s）语音文件，作为 ref_audio_path.
# 如果有多个满足要求的语音文件，选择最长的那个    
def find_ref_audio_path(model_num):
    audio_data_path = f"/home/xtz/datasets/sample_data/{model_num}/data.json"
    with open(audio_data_path, "r") as f:
        data = json.load(f)

    audio_info = []
    for audio in data:
        if 3.0 <= audio["Audio_Length"] <= 9.5:
            audio_info.append(audio)
    
    if len(audio_info) == 0:
        print(f"No audio file found for speaker {model_num}")
        return None, None
    
    audio_info.sort(key=lambda x: x["Audio_Length"], reverse=True)
    ref_audio_path = audio_info[0]["Audio_Path"]
    ref_audio_text = audio_info[0]["Paragraph"]
    return ref_audio_path, ref_audio_text

if __name__ == "__main__":
    spk_ids = read_participant_id("/home/xtz/datasets/_file/train_split_Depression_AVEC2017.csv")
    model_info = find_model_path(spk_ids)

    for model in model_info:
        model_num = model["SpeakerID"]
        ref_audio_path, ref_audio_text = find_ref_audio_path(model_num)
        model["ref_audio_path"] = ref_audio_path
        model["ref_audio_text"] = ref_audio_text

    with open("/home/xtz/datasets/_file/model_list.json", "w") as f:
        json.dump(model_info, f, indent=4)