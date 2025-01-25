# 作者：刘成广
# 时间：2024/8/7 下午4:32
# 1-----------最初的处理，将完整对话语音提取被采访者的单句话，并保存。
import psutil
import os
from shutil import copyfile, rmtree
import sys
from zipfile import ZipFile
import numpy as np
import librosa
import h5py
import pickle
import gc
import math
import soundfile as sf


def get_meta_data(dataset_path):
    """
    Grabs meta data from the dataset including, a list of the folders,
    a list of the audio paths, and a list of the transcription files for all
    the files in the dataset

    Input
        dataset_path: str - Location of the dataset

    Outputs
        folder_list: list - The complete list of folders in the dataset
        audio_paths: list - The complete list of audio locations for the data
        transcript_paths: list - The complete list of locations of the
                          transcriptions
    """
    folder_list = []
    audio_files = []
    audio_paths = []
    transcript_paths = []
    list_dir_dataset_path = os.listdir(dataset_path)
    list_dir_dataset_path.sort()
    counter = 0

    for i in list_dir_dataset_path:
        if i.endswith('_P'):
            folder_list.append(i)
            for j in os.listdir(os.path.join(dataset_path, i)):
                if 'wav' in j:
                    audio_files.append(j)
                    audio_paths.append(os.path.join(dataset_path, i, j))
                if 'TRANSCRIPT' in j:
                    if 'lock' in j or '._' in j:
                        pass
                    else:
                        transcript_paths.append(os.path.join(dataset_path, i, j))

    return folder_list, audio_paths, transcript_paths

# 从 transcriptions 文件中提取出每一段由 Participate 说话的开始和结束时间，并保存在on_off_times.pickle里
def obtaining_paragraph_time(transcript_paths, current_dir,
                               mode_for_bkgnd=False, remove_background=True):
    on_off_times = []
    paragraph_all = []

    # Interruptions during the session, these intervals should be removed
    special_case = {373: [395, 428],
                 444: [286, 387]}

    # Files with missing virtual agent transcriptions - onset/offset timings
    special_case_2 = [451, 458, 480]

    # Misaligned transcript timings
    special_case_3 = {318: 34.319917,
                      321: 3.8379167,
                      341: 6.1892,
                      362: 16.8582}
    
    belongs_id = []
    for i in transcript_paths:
        print(i)
        trial = i.split('\\')[-2]
        trial = int(trial.split('_')[0])
        with open(i, 'r') as file:
            data = file.readlines()
        participant_count = 0
        inter = []
        paragraph = []
        temp_sentence = []

        # 读取每行数据
        for j, values in enumerate(data):
            # The headers are in first position
            if j == 0: continue
            
            # The values in this list are actually strings, not int
            # Create temp which holds onset/offset times and the speaker id
            # temp = [start_time	stop_time	speaker]
            temp = values.split()[0:3]
            sentence = values.split()[3:]

            # This corrects misalignment errors
            if len(temp) == 0:
                    time_start = time_end = 0
            else:
                if trial in special_case_3:
                        time_start = float(temp[0]) + special_case_3[trial]
                        time_end = float(temp[1]) + special_case_3[trial]
                else:
                    time_start = float(temp[0])
                    time_end = float(temp[1])
            
            # 意义不明
            # if len(values) > 1:
            #     sync = values.split()[-1]
            # else:
            #     sync = ''
            # if sync == '[sync]' or sync == '[syncing]':
            #     sync = True
            # else:
            #     sync = False
            
            if len(temp) > 0 and temp[-1] == ('Participant' or 'participant'):
                participant_count += 1
                
                # if trial in special_case:  # 存在中断情况的样本
                #     inter_start = special_case[trial][0]
                #     inter_end = special_case[trial][1]
                #     if inter_start < time_start < inter_end or inter_start < time_end < inter_end:
                #         continue

                inter.append([str(time_start), str(time_end)])
                paragraph.append(" ".join(sentence))

        assert len(inter) == len(paragraph)

        belongs_id.append(trial)
        on_off_times.append(inter)
        paragraph_all.append(paragraph)

    with open(os.path.join(current_dir, 'on_off_times.pickle'), 'wb') as f:
        pickle.dump(on_off_times, f)

    with open(os.path.join(current_dir, 'belongs_id.pickle'), 'wb') as f:
        pickle.dump(belongs_id, f)

    with open(os.path.join(current_dir, 'paragraph_all.pickle'), 'wb') as f:
        pickle.dump(paragraph_all, f)

    return on_off_times, paragraph_all, belongs_id

def modify_audio_file(data, timings, sr, mode=False):
    """
    Function to remove segments from an audio file by creating a new audio
    file containing audio segments from the onset/offset time markers
    specified in the 'timings' variable

    Inputs
        data: numpy.array - The audio data to modify
        timings: list - The onset and offset time markers
        sr: int - The original sampling rate of the audio
        mode: bool - Set True if only considering the background information
              for the audio signal (this is the opening of the audio to the
              point where the first interaction begins)

    Output
        updateed_audio: numpy.array - The updated audio file
    """
    timings = np.array(timings, float)
    samples = timings * sr
    samples = np.array(samples, int)
    pointer = 0
    if mode:
        updated_audio = data[0:samples[0][1]]
    else:
        for i in samples:
            if pointer == 0:
                updated_audio = data[i[0]:i[1]]
                pointer += 1
            else:
                updated_audio = np.hstack((updated_audio, data[i[0]:i[1]]))

    return updated_audio


def save_audio_sentence(data, timings, id, sr, sen_path):
    timings = np.array(timings, float)
    samples = timings * sr
    samples = np.array(samples, int)
    for iterator, i in enumerate(samples):
        sentence_audio = data[i[0]:i[1]]
        path = os.path.join(sen_path, str(id)+"_s"+str(iterator+1) + "_AUDIO.wav")
        sf.write(path, sentence_audio, sr)


DATASET = 'D:\\buptAI\\datasets'  # 修改路径为申请到的DAIC-woz 数据集文件夹
current_directory = 'D:\\buptAI\\datasets\\PyUtils\\datasets'

if os.path.exists(current_directory):
    option = input('A directory at this location exists, do you want to delete? ')
    if option == ('y' or 'Y' or 'yes' or 'Yes'):
        rmtree(current_directory, ignore_errors=False, onerror=None)
    else:
        print('Please choose a different path, program will now terminate.')
        sys.exit()

os.makedirs(current_directory)
folders_to_make = ['audio_data', 'sample_data']
for i in folders_to_make:
    os.mkdir(os.path.join(current_directory, i))

folder_list, audio_paths, transcript_paths = get_meta_data(DATASET)

# on_off_times: 查查回答者每一次回话的开始和结束时间，并保存在on_off_times.pickle里
on_off_times, paragraph_all, belongs_id = obtaining_paragraph_time(transcript_paths, current_directory)
# on_off_times_np = np.array(on_off_times, dtype=object)
# np.save(current_directory + '/on_times.npy', on_off_times_np)

print('Processing Audio Files\n')
output_data = np.zeros((len(audio_paths), 4))
for iterator, filename in enumerate(audio_paths):
    print(f"iterator is {iterator}, and filename is {filename}")
    audio_data, sample_rate = librosa.load(filename, sr=None, mono=False)  # sample_rate = 16000
    if audio_data.ndim > 1:
        input('2 Channels were detected')
    mod_audio = modify_audio_file(audio_data, on_off_times[iterator], sample_rate)

    if sys.platform == 'win32':
        folder_name = filename.split('\\')[-2]
    else:
        folder_name = filename.split('\\')[-2]

    # Save the the audio_data
    path = os.path.join(current_directory, folders_to_make[0],
                        folder_name + '_audio_data.npy')
    np.save(path, mod_audio)
    path_wav = os.path.join(current_directory, folders_to_make[0],
                            folder_name + '_audio_data.wav')
    sf.write(path_wav, mod_audio, sample_rate)

    # 在sample_paragraph文件夹下，创建这个文件夹
    folder_number = int(folder_name[0:3])
    sen_path = os.path.join(current_directory, folders_to_make[1], str(folder_number))
    os.makedirs(sen_path)
    # --------将每段落分开保存-----------
    save_audio_sentence(audio_data, on_off_times[iterator], belongs_id[iterator], sample_rate, sen_path)







