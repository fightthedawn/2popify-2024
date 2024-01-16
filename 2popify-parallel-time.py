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

# Function to detect a 2 pop in the audio using the TensorFlow Lite model
def detect_2_pop_with_model_tflite(audio_path, tflite_model_path, classes, detection_threshold):
    # Load the TFLite model and allocate tensors
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()

    # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Preprocess the audio
    waveform = preprocess_audio_for_model(audio_path)
    inp = np.array([waveform], dtype=np.float32)

    # Check the required input shape and resize if necessary
    input_shape = input_details[0]['shape']
    inp = np.resize(inp, input_shape)

    # Set the tensor to point to the input data
    interpreter.set_tensor(input_details[0]['index'], inp)
    interpreter.invoke()

    # Retrieve the output and apply the threshold
    class_scores = interpreter.get_tensor(output_details[0]['index'])[0]
    return float(class_scores[classes.index("2pop")]) > detection_threshold

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
def process_audio_file(file_path, tflite_model_path, classes, temp_folder, detection_threshold=0.95):
    start_file_time = time.time()

    audio = AudioSegment.from_file(file_path)
    temp_file = os.path.join(temp_folder, os.path.basename(file_path))

    # Timer for 2 pop detection
    start_2pop_time = time.time()
    has_2_pop = detect_2_pop_with_model_tflite(file_path, tflite_model_path, classes, detection_threshold)
    end_2pop_time = time.time()
    print(f"2 pop detection time for {file_path}: {end_2pop_time - start_2pop_time:.2f} seconds")

    # Timer for onset detection
    start_onset_time = time.time()
    start_of_music = find_music_onset(file_path)
    end_onset_time = time.time()
    print(f"Music onset detection time for {file_path}: {end_onset_time - start_onset_time:.2f} seconds")

    # Timer for audio editing
    start_edit_time = time.time()
    if has_2_pop:
        audio = audio[1500:]
    else:
        silence_duration = max(500 - start_of_music, 0)
        audio = AudioSegment.silent(duration=silence_duration, frame_rate=audio.frame_rate) + audio[max(0, start_of_music - 500):]
    audio.set_frame_rate(audio.frame_rate).set_channels(audio.channels).export(temp_file, format="wav")
    end_edit_time = time.time()
    print(f"Audio editing time for {file_path}: {end_edit_time - start_edit_time:.2f} seconds")

    end_file_time = time.time()
    print(f"Total processing time for {file_path}: {end_file_time - start_file_time:.2f} seconds")
    return temp_file

# Main function to process all audio files in a folder
def process_folder(folder_path, model, classes, export_level):
    start_time = time.time()
    temp_folder, exports_folder = os.path.join(folder_path, "Temp"), os.path.join(folder_path, "Exports")

    shutil.rmtree(temp_folder, ignore_errors=True)
    shutil.rmtree(exports_folder, ignore_errors=True)
    os.makedirs(temp_folder, exist_ok=True)
    os.makedirs(exports_folder, exist_ok=True)

    # Start timer for audio file processing
    start_processing_time = time.time()

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.wav', '.aif', '.aiff', '.mp3', '.mp4')):
            file_path = os.path.join(folder_path, filename)
            process_audio_file(file_path, tflite_model_path, classes, temp_folder, detection_threshold=0.5)

    # End timer for audio file processing
    end_processing_time = time.time()
    print(f"Total time for processing audio files: {end_processing_time - start_processing_time:.2f} seconds")

    # Start timer for normalization
    start_norm_time = time.time()

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(normalize_audio_file, os.path.join(temp_folder, file), exports_folder, export_level) for file in os.listdir(temp_folder)]
        for future in concurrent.futures.as_completed(futures):
            future.result()

   # End timer for normalization
    end_norm_time = time.time()
    print(f"Total normalization time for all files: {end_norm_time - start_norm_time:.2f} seconds")

    shutil.rmtree(temp_folder)
    total_time = time.time() - start_time
    print(f"Overall total processing time: {total_time:.2f} seconds.")

# Sets all settings, loads model, and runs Main function to process provided folder
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process audio files for 2 pop detection and normalization.")
    parser.add_argument("folder_path", type=str, help="Path to the folder containing audio files")
    parser.add_argument("--level", type=float, default=-14, help="Export loudness level")
    args = parser.parse_args()

    tflite_model_path = 'ModelsTrained/2popmodel100000lite/model.tflite'  # Update with the actual path of your TFLite model
    classes = ["Music", "2pop"]

    process_folder(args.folder_path, tflite_model_path, classes, args.level)