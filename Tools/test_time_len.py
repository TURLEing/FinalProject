import os
import wave


# Function to get the duration of a .wav file
def get_wav_duration(file_path):
    with wave.open(file_path, 'r') as wav_file:
        frames = wav_file.getnframes()
        rate = wav_file.getframerate()
        duration = frames / float(rate)
        return duration

# Function to get the total duration of all .wav files in a directory
def get_total_duration(directory):
    total_duration = 0.0
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.wav'):
                file_path = os.path.join(root, file)
                total_duration += get_wav_duration(file_path)
    return total_duration

if __name__ == "__main__":
    directory = "D:\\buptAI\\augment_audio\\319"
    total_duration = get_total_duration(directory)
    print(f"Total duration of all .wav files: {total_duration} seconds")