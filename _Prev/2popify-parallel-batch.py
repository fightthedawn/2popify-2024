import os
import numpy as np
import librosa
import tensorflow as tf
from pydub import AudioSegment, silence
import soundfile as sf
import subprocess
import argparse
import time
import concurrent.futures
import shutil

# Function to convert audio file to WAV format
def convert_to_wav(audio_path):
    audio = AudioSegment.from_file(audio_path)
    wav_path = os.path.splitext(audio_path)[0] + ".wav"
    audio.export(wav_path, format="wav")
    return wav_path

# Function to preprocess audio for the model
def preprocess_audio_for_model(audio_path, target_sr=16000, duration_ms=1500, normalize=True):
    audio, sr = librosa.load(audio_path, sr=target_sr, mono=True)
    silence_duration = int(0.5 * sr)
    audio = np.concatenate([np.zeros(silence_duration), audio])
    if normalize:
        audio = librosa.util.normalize(audio)
    onset_frames = librosa.onset.onset_detect(y=audio, sr=sr, units='samples', backtrack=True)
    onset_sample = onset_frames[0] if onset_frames.size > 0 else silence_duration
    end_sample = min(onset_sample + int(sr * (duration_ms / 1000.0)), len(audio))
    return audio[onset_sample:end_sample]

def preprocess_audio_batch(folder_path, target_sr=16000, duration_ms=1500, normalize=True):
    batch = []
    file_paths = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.wav', '.aif', '.aiff', '.mp3', '.mp4')):
            file_path = os.path.join(folder_path, filename)
            print(f"Adding to batch: {file_path}")  # Debug print
            audio_data = preprocess_audio_for_model(file_path, target_sr, duration_ms, normalize)
            batch.append(audio_data)
            file_paths.append(file_path)
    print(f"Number of files in batch: {len(batch)}, Number of file paths: {len(file_paths)}")
    return np.array(batch), file_paths

# Function to detect a 2 pop in the audio using the trained model
def detect_2_pop_with_model(audio_path, model, classes, detection_threshold):
    waveform = preprocess_audio_for_model(audio_path)
    inp = tf.constant(np.array([waveform]), dtype='float32')
    class_scores = model(inp)[0].numpy()
    return float(class_scores[classes.index("2pop")]) > detection_threshold

"""
def detect_2_pop_batch_with_model(audio_batch, model, classes, detection_threshold):
    batch_input = tf.constant(audio_batch, dtype='float32')
    class_scores = model(batch_input).numpy()
    return [float(scores[classes.index("2pop")]) > detection_threshold for scores in class_scores]
"""

def detect_2_pop_batch_with_model(audio_batch, model, classes, detection_threshold):
    # Normalize length of each waveform in the batch
    max_length = max(len(waveform) for waveform in audio_batch)
    padded_batch = [np.pad(waveform, (0, max_length - len(waveform)), 'constant') for waveform in audio_batch]

    # Convert to numpy array and ensure float32 data type
    batch_input = np.array(padded_batch, dtype=np.float32)

    # Process the batch through the TensorFlow model
    class_scores_batch = model(tf.constant(batch_input)).numpy()

    # Interpret the results for each waveform
    detection_results = [float(scores[classes.index("2pop")]) > detection_threshold for scores in class_scores_batch]
    return detection_results


# Function to find the onset of music in an audio file
def find_music_onset(audio_path):
    if not audio_path.endswith(('.wav', '.aif', '.aiff')):
        audio_path = convert_to_wav(audio_path)

    y, sr = librosa.load(audio_path, sr=None, mono=True)
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, units='time', pre_max=1, post_max=1, pre_avg=1, post_avg=1, delta=0.01, wait=1)
    return int(onset_frames[0] * 1000) if onset_frames.any() else 0

# Function to normalize the loudness of an audio file using FFmpeg
def normalize_loudness_ffmpeg(input_file, output_file, export_level):
    FNULL = open(os.devnull, 'w')
    cmd = ['ffmpeg', '-i', input_file, '-filter_complex', f'loudnorm=I={export_level}:TP=-1.5:LRA=11', '-y', output_file]
    subprocess.run(cmd, stdout=FNULL, stderr=subprocess.STDOUT)
    FNULL.close()

# Function to normalize an audio file and move it to the exports folder
def normalize_audio_file(input_file, exports_folder, export_level):
    output_file = os.path.join(exports_folder, os.path.basename(input_file))
    normalize_loudness_ffmpeg(input_file, output_file, export_level)

# Function to process an audio file and export it to a temporary folder
def process_audio_file(file_path, has_2_pop, temp_folder):
    print(f"Processing file: {file_path}, 2 pop detected: {has_2_pop}")

    # Load the audio file
    audio = AudioSegment.from_file(file_path)
    temp_file = os.path.join(temp_folder, os.path.basename(file_path))

    # Process the audio file based on the 2 pop detection
    if has_2_pop:
        print("Trimming first 1500ms due to 2 pop detection")
        audio = audio[1500:]
    else:
        print("Finding music onset for audio editing")
        start_of_music = find_music_onset(file_path)
        silence_duration = max(500 - start_of_music, 0)
        audio = AudioSegment.silent(duration=silence_duration, frame_rate=audio.frame_rate) + audio[max(0, start_of_music - 500):]

    # Export the processed audio
    audio.export(temp_file, format="wav")
    print(f"Exported processed audio to {temp_file}")

# Main function to process all audio files in a folder
def process_folder(folder_path, model, classes, export_level, detection_threshold=0.95):
    start_time = time.time()
    temp_folder, exports_folder = os.path.join(folder_path, "Temp"), os.path.join(folder_path, "Exports")

    shutil.rmtree(temp_folder, ignore_errors=True)
    shutil.rmtree(exports_folder, ignore_errors=True)
    os.makedirs(temp_folder, exist_ok=True)
    os.makedirs(exports_folder, exist_ok=True)

    # Batch preprocess and 2 pop detection
    audio_batch, file_paths = preprocess_audio_batch(folder_path)
    detection_results = detect_2_pop_batch_with_model(audio_batch, model, classes, detection_threshold)

    if len(file_paths) != len(detection_results):
        print("Mismatch in the number of files and detection results")
        return

    for file_path, has_2_pop in zip(file_paths, detection_results):
        print(f"Processing file: {file_path}, 2 pop detected: {has_2_pop}")
        process_audio_file(file_path, has_2_pop, temp_folder)
        
    # Parallel processing for normalization
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(normalize_audio_file, os.path.join(temp_folder, os.path.basename(file_path)), exports_folder, export_level) for file_path in file_paths]
        for future in concurrent.futures.as_completed(futures):
            print(f"Normalizing file: {future}")

    shutil.rmtree(temp_folder)
    print(f"Total processing time: {time.time() - start_time:.2f} seconds.")

# Sets all settings, loads model, and runs Main function to process provided folder
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process audio files for 2 pop detection and normalization.")
    parser.add_argument("folder_path", type=str, help="Path to the folder containing audio files")
    parser.add_argument("--level", type=float, default=-14, help="Export loudness level")
    args = parser.parse_args()
    model = tf.saved_model.load('ModelsTrained/2popmodel100000-20240110')
    classes = ["Music", "2pop"]
    process_folder(args.folder_path, model, classes, args.level)