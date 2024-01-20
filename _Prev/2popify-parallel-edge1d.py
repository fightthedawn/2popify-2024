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
def preprocess_audio_for_model(audio_path, target_sr=48000, target_length=2600, normalize=True):
    audio, sr = librosa.load(audio_path, sr=target_sr, mono=True)
    if normalize:
        audio = librosa.util.normalize(audio)
    if len(audio) > target_length:
        audio = audio[:target_length]
    elif len(audio) < target_length:
        audio = np.pad(audio, (0, target_length - len(audio)))
    return audio

# Function to detect a 2 pop in the audio using the trained model
def detect_2_pop_with_model(audio_path, model, classes, detection_threshold):
    waveform = preprocess_audio_for_model(audio_path)
    # Add a batch dimension and ensure type is float32
    inp = np.expand_dims(waveform, axis=0).astype(np.float32)

    # Use the model's serving signature to make a prediction
    class_scores = model.signatures['serving_default'](x=tf.constant(inp))['output_0'].numpy()
    # Get the score for the "2pop" class
    two_pop_score = class_scores[0][classes.index("2pop")]

    return float(two_pop_score) > detection_threshold

def detect_2_pop_batch_with_model(audio_paths, model, classes, detection_threshold):
    # Preprocess all audio files and store in a list
    batch_waveform = [preprocess_audio_for_model(path) for path in audio_paths]
    
    # Stack all waveforms into a batch
    batch_waveform = np.stack(batch_waveform, axis=0).astype(np.float32)
    
    # Use the model's serving signature to make predictions for the batch
    class_scores = model.signatures['serving_default'](x=tf.constant(batch_waveform))['output_0'].numpy()
    
    # Interpret the results for each audio clip in the batch
    two_pop_scores = class_scores[:, classes.index("2pop")]
    
    # Return a list of boolean values indicating the presence of 2pop
    return two_pop_scores > detection_threshold

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

# Main function to process all audio files in a folder
def process_folder(folder_path, model, classes, export_level, detection_threshold=0.5, batch_size=32):
    start_time = time.time()
    temp_folder, exports_folder = os.path.join(folder_path, "Temp"), os.path.join(folder_path, "Exports")

    # Remove existing temporary and export folders, then create new ones
    shutil.rmtree(temp_folder, ignore_errors=True)
    shutil.rmtree(exports_folder, ignore_errors=True)
    os.makedirs(temp_folder, exist_ok=True)
    os.makedirs(exports_folder, exist_ok=True)

    # Get all audio file paths
    all_audio_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(('.wav', '.aif', '.aiff', '.mp3', '.mp4'))]
    
    # Process files in batches
    for i in range(0, len(all_audio_paths), batch_size):
        batch_paths = all_audio_paths[i:i + batch_size]
        
        # Detect 2pop in batch
        detection_results = detect_2_pop_batch_with_model(batch_paths, model, classes, detection_threshold)
        
        # Process each file in the batch
        for audio_path, has_2_pop in zip(batch_paths, detection_results):
            # Process and move to temp folder based on detection result
            processed_file_path = process_audio_file(audio_path, has_2_pop, temp_folder)
            if processed_file_path:
                # Normalize and move to exports folder
                normalize_audio_file(processed_file_path, exports_folder, export_level)

    # Cleanup: remove the temporary folder
    shutil.rmtree(temp_folder)
    total_time = time.time() - start_time
    print(f"Total processing time: {total_time:.2f} seconds.")

# Sets all settings, loads model, and runs Main function to process provided folder
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process audio files for 2 pop detection and normalization.")
    parser.add_argument("folder_path", type=str, help="Path to the folder containing audio files")
    parser.add_argument("--level", type=float, default=-14, help="Export loudness level")
    args = parser.parse_args()
    model = tf.saved_model.load('ModelsTrained/2popedge20240118-eon')
    classes = ["Music", "2pop"]
    process_folder(args.folder_path, model, classes, args.level)