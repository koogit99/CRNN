import os
import librosa
import numpy as np
import soundfile as sf

# 오디오 파일 형식: 모노, 샘플레이트(sr) : 16kHz, 샘플당 비트 : 16
def load_and_pad_audio(file_path, sr=16000, max_length=65536):
    audio, _ = librosa.load(file_path, sr=sr)
    if len(audio) < max_length:
        audio = np.pad(audio, (0, max_length - len(audio)), 'constant')
    else:
        audio = audio[:max_length]
    return audio

def preprocess_audio_files(input_dir, sr=16000, max_length=65536):
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.wav'):
                file_path = os.path.join(root, file)
                audio = load_and_pad_audio(file_path, sr, max_length)
                output_path = file_path  # 파일을 동일한 위치에 저장
                sf.write(output_path, audio, sr)
                print(f"Processed {file_path}")

if __name__ == "__main__":
    directories = [
        "Datasets/train_dataset",
        "Datasets/validation_dataset",
        "Datasets/noisy_testset"
    ]
    sample_rate = 16000
    max_audio_length = 65536 #25920 -> (161) / 65536 -> (410)

    for directory in directories:
        preprocess_audio_files(directory, sample_rate, max_audio_length)
