import os
import numpy as np
import librosa
import tensorflow as tf
from pydub import AudioSegment
import subprocess
import argparse
import time
import concurrent.futures
import shutil

def convert_audio_to_wav(audio_path):
    if not audio_path.lower().endswith('.wav'):
        audio = AudioSegment.from_file(audio_path)
        wav_path = os.path.splitext(audio_path)[0] + ".wav"
        audio.export(wav_path, format="wav")
        return wav_path
    return audio_path

def preprocess_and_detect_2_pop(audio_path, model, classes, target_sr=16000, duration_ms=1500):
    audio, sr = librosa.load(audio_path, sr=target_sr, mono=True)
    silence_duration = int(0.5 * sr)
    audio = np.pad(audio, (silence_duration, 0), 'constant')
    audio = librosa.util.normalize(audio)
    onset_frames = librosa.onset.onset_detect(y=audio, sr=sr, units='samples', backtrack=True)
    onset_sample = onset_frames[0] if onset_frames.size > 0 else silence_duration
    audio_segment = audio[onset_sample:onset_sample + int(sr * (duration_ms / 1000.0))]

    inp = tf.constant([audio_segment], dtype='float32')
    class_scores = model(inp)[0].numpy()
    return float(class_scores[classes.index("2pop")]) > 0.5

def process_audio_file(file_path, model, classes, temp_folder):
    wav_path = convert_audio_to_wav(file_path)
    is_2_pop = preprocess_and_detect_2_pop(wav_path, model, classes)
    
    audio = AudioSegment.from_file(wav_path)
    export_path = os.path.join(temp_folder, os.path.basename(file_path))

    if is_2_pop:
        audio[1500:].export(export_path, format="wav")
    else:
        onset_frames = librosa.onset.onset_detect(y=audio, sr=audio.frame_rate, units='time',
                                                  pre_max=1, post_max=1, pre_avg=1, post_avg=1, delta=0.01, wait=1)
        music_start = int(onset_frames[0] * 1000) if onset_frames.any() else 0
        silence_duration = max(500 - music_start, 0)
        trimmed_audio = AudioSegment.silent(duration=silence_duration, frame_rate=audio.frame_rate) + audio[max(0, music_start - 500):]
        trimmed_audio.export(export_path, format="wav")

def normalize_loudness_ffmpeg(input_file, output_file, export_level):
    FNULL = open(os.devnull, 'w')
    cmd = ['ffmpeg', '-i', input_file, '-filter_complex', f'loudnorm=I={export_level}:TP=-1.5:LRA=11', '-y', output_file]
    subprocess.run(cmd, stdout=FNULL, stderr=subprocess.STDOUT)
    FNULL.close()

def normalize_audio_file(input_file, exports_folder, export_level):
    output_file = os.path.join(exports_folder, os.path.basename(input_file))
    normalize_loudness_ffmpeg(input_file, output_file, export_level)

def process_folder(folder_path, model, classes, export_level):
    start_time = time.time()
    temp_folder, exports_folder = os.path.join(folder_path, "Temp"), os.path.join(folder_path, "Exports")

    shutil.rmtree(temp_folder, ignore_errors=True)
    shutil.rmtree(exports_folder, ignore_errors=True)
    os.makedirs(temp_folder, exist_ok=True)
    os.makedirs(exports_folder, exist_ok=True)

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.wav', '.aif', '.aiff', '.mp3', '.mp4')):
            process_audio_file(os.path.join(folder_path, filename), model, classes, temp_folder)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(normalize_audio_file, os.path.join(temp_folder, file), exports_folder, export_level) for file in os.listdir(temp_folder)]
        for future in concurrent.futures.as_completed(futures):
            future.result()

    shutil.rmtree(temp_folder)
    print(f"Total processing time: {time.time() - start_time:.2f} seconds.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process audio files for 2 pop detection and normalization.")
    parser.add_argument("folder_path", type=str, help="Path to the folder containing audio files")
    parser.add_argument("--level", type=float, default=-14, help="Export loudness level")
    args = parser.parse_args()
    model = tf.saved_model.load('ModelsTrained/2popmodel60000')
    classes = ["Music", "2pop"]
    process_folder(args.folder_path, model, classes, args.level)