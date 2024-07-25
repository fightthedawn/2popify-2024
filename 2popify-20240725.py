import os
import numpy as np
from scipy.signal import butter, sosfilt
import librosa
import tensorflow as tf
import soundfile as sf
from pydub import AudioSegment
import argparse
import time
import concurrent.futures
import shutil
from functools import lru_cache

# Function to convert audio file to WAV format
def convert_to_wav(audio_path):
    audio = AudioSegment.from_file(audio_path)
    wav_path = os.path.splitext(audio_path)[0] + ".wav"
    audio.export(wav_path, format="wav")
    return wav_path

@lru_cache(maxsize=None)
def get_butter_bandpass_sos(lowcut, highcut, sr, order=5):
    nyq = 0.5 * sr
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], btype='band', output='sos')
    return sos

def butter_bandpass_filter(data, lowcut, highcut, sr, order=5):
    sos = get_butter_bandpass_sos(lowcut, highcut, sr, order)
    y = sosfilt(sos, data)
    return y

@lru_cache(maxsize=None)
def load_audio(audio_path, target_sr=16000):
    return librosa.load(audio_path, sr=target_sr, mono=True)

def preprocess_audio_for_model(audio_path, target_sr=16000):
    # Load the audio file
    audio, sr = load_audio(audio_path, target_sr)
    
    # Apply bandpass filter
    audio_filtered = butter_bandpass_filter(audio, 140, 2200, sr)
    
    # Slice the first 1500ms for 2pop detection
    audio_slice_for_2pop_detection = audio_filtered[:int(1.5 * sr)]
    
    return audio_slice_for_2pop_detection, sr

def detect_2_pop_with_model(audio, sr, model, classes, detection_threshold=0.95):
    inp = tf.constant(np.array([audio]), dtype='float32')
    class_scores = model(inp)[0].numpy()
    return float(class_scores[classes.index("2pop")]) > detection_threshold

def find_music_onset(audio, sr):
    onset_frames = librosa.onset.onset_detect(y=audio, sr=sr, units='time', pre_max=1, post_max=1, pre_avg=1, post_avg=1, delta=0.01, wait=1)
    return onset_frames[0] if onset_frames.size > 0 else 0

def process_audio_file(file_path, model, classes, temp_folder, detection_threshold=0.95):
    audio = AudioSegment.from_file(file_path)
    temp_file = os.path.join(temp_folder, os.path.basename(file_path))

    waveform, sr = preprocess_audio_for_model(file_path)

    if detect_2_pop_with_model(waveform, sr, model, classes, detection_threshold):
        # If a 2 pop is detected, trim the first 1500ms from the audio
        audio_trimmed_for_onset_detection = audio[1500:]
        
        # Export the trimmed audio temporarily for onset detection
        temp_trimmed_path = os.path.join(temp_folder, "temp_for_onset_detection.wav")
        audio_trimmed_for_onset_detection.export(temp_trimmed_path, format="wav")
        
        # Now use the trimmed audio to find the music onset
        audio_for_onset, sr = librosa.load(temp_trimmed_path, sr=None, mono=True)
        start_of_music = find_music_onset(audio_for_onset, sr)
        
        # Calculate the actual start position in the original audio
        actual_start_position = max(int(start_of_music * 1000) + 1000, 0)  # Ensure it doesn't go negative
        
        # Trim the original audio based on the calculated start position and export
        trimmed_audio = audio[actual_start_position:]
        trimmed_audio.export(temp_file, format="wav")
        
        # Clean up the temporary file used for onset detection
        os.remove(temp_trimmed_path)
    else:
        # If no 2 pop is detected, use the original logic for music onset detection
        audio_for_onset, sr = librosa.load(file_path, sr=None, mono=True)
        start_of_music = find_music_onset(audio_for_onset, sr)
        start_of_music_ms = int(start_of_music * 1000)
        silence_duration = max(500 - start_of_music_ms, 0)
        trimmed_audio = AudioSegment.silent(duration=silence_duration, frame_rate=audio.frame_rate) + audio[max(0, start_of_music_ms - 500):]
        trimmed_audio.export(temp_file, format="wav")

def normalize_audio(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(('.wav', '.aif', '.aiff')):
            file_path = os.path.join(folder_path, filename)
            audio = AudioSegment.from_file(file_path)
            normalized_audio = audio.normalize()
            normalized_audio.export(file_path, format="wav")

def process_folder(folder_path, model, classes):
    start_time = time.time()
    temp_folder = os.path.join(folder_path, "Temp")
    exports_folder = os.path.join(folder_path, "Exports")

    # Setup folders
    shutil.rmtree(temp_folder, ignore_errors=True)
    os.makedirs(temp_folder, exist_ok=True)
    shutil.rmtree(exports_folder, ignore_errors=True)
    os.makedirs(exports_folder, exist_ok=True)

    # Process files in parallel
    audio_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.wav', '.aif', '.aiff', '.mp3', '.mp4'))]
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_audio_file, os.path.join(folder_path, f), model, classes, temp_folder) for f in audio_files]
        concurrent.futures.wait(futures)

    # Normalize audio
    normalize_audio(temp_folder)

    # Move processed files to exports folder
    for file_name in os.listdir(temp_folder):
        shutil.move(os.path.join(temp_folder, file_name), os.path.join(exports_folder, file_name))

    # Cleanup
    shutil.rmtree(temp_folder)
    
    print(f"Total processing time: {time.time() - start_time:.2f} seconds.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process audio files for 2 pop detection and music onset.")
    parser.add_argument("folder_path", type=str, help="Path to the folder containing audio files")
    args = parser.parse_args()
    
    # Load the original model
    model = tf.saved_model.load('/Users/giovonnilobato/Documents/GitHub/2popify-2024/ModelsTrained/2popmodel100000-20240110')
    classes = ["Music", "2pop"]
    
    process_folder(args.folder_path, model, classes)