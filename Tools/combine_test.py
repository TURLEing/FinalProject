# 将两个 wav 文件连续拼接在一起，生成一个新的 wav 文件

import os
import wave
import numpy as np
def combine_wav(wav1, wav2, output):
    # 打开两个 wav 文件
    f1 = wave.open(wav1, 'rb')
    f2 = wave.open(wav2, 'rb')
    # 创建一个新的 wav 文件
    f3 = wave.open(output, 'wb')
    # 设置参数
    f3.setnchannels(f1.getnchannels())
    f3.setsampwidth(f1.getsampwidth())
    f3.setframerate(f1.getframerate())
    f3.setnframes(f1.getnframes() + f2.getnframes() + int(0.5 * f1.getframerate()))
    # 读取数据
    data1 = f1.readframes(f1.getnframes())
    data2 = f2.readframes(f2.getnframes())
    # 创建0.5秒的静音数据
    silence = np.zeros(int(0.5 * f1.getframerate()) * f1.getnchannels() * f1.getsampwidth(), dtype=np.uint8)
    # 写入数据
    f3.writeframes(data1)
    f3.writeframes(silence.tobytes())
    f3.writeframes(data2)
    # 关闭文件
    f1.close()
    f2.close()
    f3.close()

# 更改当前文件的运行路径
os.chdir('/home/xtz')
print(os.getcwd())

# Example of usage

wav1 = "datasets/sample_data/385/385_s37_AUDIO.wav"
wav2 = "datasets/sample_data/385/385_s38_AUDIO.wav"
output = "datasets/sample_data/385/385_s37,38_AUDIO.wav"

combine_wav(wav1, wav2, output)