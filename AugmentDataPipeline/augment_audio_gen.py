import requests
import random
import json
import os

# 更改当前文件的运行路径
os.chdir('/home/xtz')
print(os.getcwd())

import sys
sys.path.append('/home/xtz/codebase')

from Tools.test_time_len import get_wav_duration

def tts_get_request(text, ref_audio_path, prompt_text, aux_ref_audio_paths):
    """调用 TTS API 的 GET 请求"""
    url = "http://127.0.0.1:9880/tts"
    params = {
        "text": text,
        "text_lang": "en",
        "ref_audio_path": ref_audio_path,
        "prompt_lang": "en",
        "prompt_text": prompt_text,
        "text_split_method": "cut4",
        "batch_size": 1,
        "media_type": "wav",
        "streaming_mode": "true"
    }
    
    response = requests.get(url, params=params, stream=True)
    if response.status_code == 200:
        with open("output_get.wav", "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                f.write(chunk)
        print("GET 请求成功，音频保存为 output_get.wav")
    else:
        print(f"GET 请求失败: {response.status_code}, 错误信息: {response.json()}")


def tts_request(text, ref_audio_path, prompt_text, aux_ref_audio_paths, save_path, save_name):
    """调用 TTS API 的 POST 请求"""
    seed = -1
    actual_seed = seed if seed not in [-1, "", None] else random.randrange(1 << 32)

    url = "http://127.0.0.1:9880/tts"
    aux_ref_audio_paths = []
    inputs = {'text': text, 
        'text_lang': 'en', 
        'ref_audio_path': ref_audio_path, 
        'aux_ref_audio_paths': aux_ref_audio_paths, 
        'prompt_text': prompt_text, 
        'prompt_lang': 'en', 
        'top_k': 5, 
        'top_p': 1, 
        'temperature': 1, 
        'text_split_method': 'cut4', 
        "batch_size": 1,
        "batch_threshold": 0.75,
        "split_bucket": True,
        "speed_factor": 1.0,
        "streaming_mode": False,
        "seed": actual_seed,
        "parallel_infer": True,
        "repetition_penalty": 1.35
    }

    headers = {
        "Content-Type": "application/json"
    }
    
    response = requests.post(url, json=inputs, headers=headers)

    if response.status_code == 200:
        save_full_path = os.path.join(save_path, save_name)
        with open(save_full_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                f.write(chunk)

        print(f"POST 请求成功，音频保存为{save_full_path}")

        # 返回音频信息
        return {
            "Speaker_ID": None,
            "Unique_ID": save_name,
            "Audio_Path": save_full_path,
            "Audio_Length": get_wav_duration(save_full_path),
            "Paragraph": text,
        }
    else:
        print(f"POST 请求失败: {response.status_code}, 错误信息: {response.json()}")
    
def tts_stop():
    """ 停止 TTS 服务 """

    url = "http://127.0.0.1:9880/control?command=exit"
    response = requests.get(url, timeout=5)
    if response.status_code == 200: return True
    else:
        print(f"停止 TTS 服务失败: {response.status_code}, 错误信息: {response.text}")
        return False

### 切换GPT模型
# endpoint: `/set_gpt_weights`

# GET:
# ```
# http://127.0.0.1:9880/set_gpt_weights?weights_path=GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt
# ```
# RESP: 
# 成功: 返回"success", http code 200
# 失败: 返回包含错误信息的 json, http code 400

def set_gpt_weights(weights_path):
    url = f"http://127.0.0.1:9880/set_gpt_weights?weights_path={weights_path}"
    response = requests.get(url)
    if response.status_code == 200:
        print("切换 GPT 模型成功")
    else:
        print(f"切换 GPT 模型失败: {response.status_code}, 错误信息: {response.json()}")
    
# ### 切换Sovits模型
# endpoint: `/set_sovits_weights`

# GET:
# ```
# http://127.0.0.1:9880/set_sovits_weights?weights_path=GPT_SoVITS/pretrained_models/s2G488k.pth
# ```

# RESP: 
# 成功: 返回"success", http code 200
# 失败: 返回包含错误信息的 json, http code 400

def set_sovits_weights(weights_path):
    url = f"http://127.0.0.1:9880/set_sovits_weights?weights_path={weights_path}"
    response = requests.get(url)
    if response.status_code == 200:
        print("切换 SoVITs 模型成功")
    else:
        print(f"切换 SoVITs 模型失败: {response.status_code}, 错误信息: {response.json()}")


def augment_audio_generation(spk_id, ref_audio_path, ref_audio_text, generated_texts=None):
    """生成 spk_id 的音频"""

    save_path = f"/home/xtz/datasets/augment_data/{spk_id}/"
    os.makedirs(save_path, exist_ok=True)

    if generated_texts is None:
        target_file_path = "/home/xtz/datasets/augment_data/generated_texts.json"
        with open(target_file_path, "r") as f:
            generated_texts = json.load(f)
    
    # 遍历 generated_texts，对其中每个元素的 text 调用 TTS API 生成音频
    generated_audio_info = []
    for generated_text in generated_texts:
        save_name = f"{spk_id}_{generated_text['scenario']}_{generated_text['emotion']}_{generated_text['max_tokens']}.wav"
        text = generated_text["text"]
        audio_info = tts_request(text, ref_audio_path, ref_audio_text, None, save_path, save_name)
        audio_info["Speaker_ID"] = spk_id
        generated_audio_info.append(audio_info)
    
    # 将生成的音频信息保存到 /home/xtz/datasets/augment_audio/{spk_id}/generated_audio_info.json
    with open(os.path.join(save_path, "data.json"), "w") as f:
        json.dump(generated_audio_info, f, indent=4)
    
    return generated_audio_info
    
if __name__ == "__main__":
    pass
