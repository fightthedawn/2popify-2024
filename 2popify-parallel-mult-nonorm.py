import os
import numpy as np
import librosa
import tensorflow as tf
from pydub import AudioSegment
import soundfile as sf
import subprocess
import argparse
import time
import multiprocessing
import concurrent.futures
import shutil

# Function to convert audio file to WAV format
def convert_to_wav(audio_path):
    audio = AudioSegment.from_file(audio_path)
    wav_path = os.path.splitext(audio_path)[0] + ".wav"
    audio.export(wav_path, format="wav")
    return wav_path

# Function to preprocess audio for the model
def preprocess_audio_for_model(audio_path, target_sr=16000, duration_ms=1500, normalize=True, load_duration=None, load_offset=0):
    audio, sr = librosa.load(audio_path, sr=target_sr, mono=True, offset=load_offset, duration=load_duration if load_duration else None)
    silence_duration = int(0.5 * sr)
    audio = np.concatenate([np.zeros(silence_duration), audio])
    if normalize:
        audio = librosa.util.normalize(audio)
    onset_frames = librosa.onset.onset_detect(y=audio, sr=sr, units='samples', backtrack=True)
    onset_sample = onset_frames[0] if onset_frames.size > 0 else silence_duration
    end_sample = min(onset_sample + int(sr * (duration_ms / 1000.0)), len(audio))
    return audio[onset_sample:end_sample]

# Function to detect a 2 pop in the audio using the trained model
def detect_2_pop_with_model(audio_path, model, classes, detection_threshold):
    waveform = preprocess_audio_for_model(audio_path)
    inp = tf.constant(np.array([waveform]), dtype='float32')
    class_scores = model(inp)[0].numpy()
    return float(class_scores[classes.index("2pop")]) > detection_threshold

# Function to find the onset of music in an audio file
def find_music_onset(audio_path):
    if not audio_path.endswith(('.wav', '.aif', '.aiff')):
        audio_path = convert_to_wav(audio_path)
    y, sr = librosa.load(audio_path, sr=None, mono=True)
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, units='time', pre_max=1, post_max=1, pre_avg=1, post_avg=1, delta=0.01, wait=1)
    return int(onset_frames[0] * 1000) if onset_frames.any() else 0

# Function to process an audio file and export it to a temporary folder
def process_audio_file(file_path, model, classes, temp_folder, detection_threshold=0.95):
    audio = AudioSegment.from_file(file_path)
    temp_file = os.path.join(temp_folder, os.path.basename(file_path))

    if detect_2_pop_with_model(file_path, model, classes, detection_threshold):
        audio[1500:].export(temp_file, format="wav")
    else:
        start_of_music = find_music_onset(file_path)
        silence_duration = max(500 - start_of_music, 0)
        trimmed_audio = AudioSegment.silent(duration=silence_duration, frame_rate=audio.frame_rate) + audio[max(0, start_of_music - 500):]
        trimmed_audio.set_frame_rate(audio.frame_rate).set_channels(audio.channels).export(temp_file, format="wav")

def process_folder(folder_path, model_path, classes, detection_threshold=0.95):
    start_time = time.time()
    temp_folder = os.path.join(folder_path, "Temp")
    shutil.rmtree(temp_folder, ignore_errors=True)
    os.makedirs(temp_folder, exist_ok=True)

    file_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(('.wav', '.aif', '.aiff', '.mp3', '.mp4'))]
    process_args = [(file_path, model_path, classes, temp_folder, detection_threshold) for file_path in file_paths]

    with multiprocessing.Pool() as pool:
        pool.starmap(process_audio_file, process_args)

    # Instead of moving, now we directly rename the folder
    os.rename(temp_folder, os.path.join(folder_path, "Exports"))
    print(f"Total processing time: {time.time() - start_time:.2f} seconds.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process audio files for 2 pop detection and normalization.")
    parser.add_argument("folder_path", type=str, help="Path to the folder containing audio files")
    parser.add_argument("--level", type=float, default=-14, help="Export loudness level")
    args = parser.parse_args()
    model_path = 'ModelsTrained/2popmodel100000-20240110'
    process_folder(args.folder_path, model_path, ["Music", "2pop"], args.level)