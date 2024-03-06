import os
import numpy as np
import librosa
import tensorflow as tf
from pydub import AudioSegment, silence
import soundfile as sf
import subprocess
import argparse
import time
import multiprocessing
import concurrent.futures
import shutil
import tensorflow.lite as tflite

def load_tflite_model(model_path):
    # Load TFLite model and allocate tensors
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    return interpreter

def run_tflite_model(interpreter, input_data):
    # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Test the model on input data
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Extract the output
    output_data = interpreter.get_tensor(output_details[0]['index'])

    return output_data

# Function to convert audio file to WAV format
def convert_to_wav(audio_path):
    audio = AudioSegment.from_file(audio_path)
    wav_path = os.path.splitext(audio_path)[0] + ".wav"
    audio.export(wav_path, format="wav")
    return wav_path

# Function to preprocess audio for the model
def preprocess_audio_for_model(audio_path, target_sr=16000, duration_ms=1500, normalize=True, load_duration=None, load_offset=0):
    # Start timing
    start_time = time.time()

    # Load audio with SoundFile
    load_start = time.time()
    try:
        with sf.SoundFile(audio_path) as sound_file:
            frame_count = min(load_duration * target_sr if load_duration else sound_file.frames, sound_file.frames - load_offset * target_sr)
            audio = sound_file.read(frames=frame_count, dtype='float32', fill_value=0.0, always_2d=True, out=None)
            if audio.shape[1] > 1:  # Convert to mono if stereo
                audio = np.mean(audio, axis=1)
            sr = sound_file.samplerate
    except Exception as e:
        print(f"Error loading audio with SoundFile: {e}")
        # Fallback to librosa if SoundFile fails
        audio, sr = librosa.load(audio_path, sr=target_sr, mono=True, offset=load_offset, duration=load_duration)

    load_end = time.time()

    # Add silence
    silence_add_start = time.time()
    silence_duration = int(0.5 * sr)
    audio = np.concatenate([np.zeros(silence_duration), audio])
    silence_add_end = time.time()

    # Normalize audio
    normalize_start = time.time()
    if normalize:
        audio = librosa.util.normalize(audio)
    normalize_end = time.time()

    # Detect onset
    onset_detect_start = time.time()
    onset_frames = librosa.onset.onset_detect(y=audio, sr=sr, units='samples', backtrack=True)
    onset_sample = onset_frames[0] if onset_frames.size > 0 else silence_duration
    end_sample = min(onset_sample + int(sr * (duration_ms / 1000.0)), len(audio))
    onset_detect_end = time.time()

    # End timing
    end_time = time.time()

    # Print timings
    print(f"Total time: {end_time - start_time:.4f}s")
    print(f"Loading time: {load_end - load_start:.4f}s")
    print(f"Silence adding time: {silence_add_end - silence_add_start:.4f}s")
    print(f"Normalization time: {normalize_end - normalize_start:.4f}s")
    print(f"Onset detection time: {onset_detect_end - onset_detect_start:.4f}s")

    return audio[onset_sample:end_sample]

# Function to detect a 2 pop in the audio using the trained model
def detect_2_pop_with_model(audio_path, interpreter, classes, detection_threshold):
    waveform = preprocess_audio_for_model(audio_path)
    inp = np.array([waveform], dtype='float32')
    class_scores = run_tflite_model(interpreter, inp)
    return float(class_scores[0][classes.index("2pop")]) > detection_threshold

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

def process_audio_file_mp(file_path, model_path, classes, temp_folder, detection_threshold, export_level):
    try:
        # Load the TFLite model inside each subprocess
        interpreter = load_tflite_model(model_path)

        preprocess_start = time.time()
        waveform = preprocess_audio_for_model(file_path)
        inp = np.array([waveform], dtype='float32')
        preprocess_end = time.time()

        model_predict_start = time.time()
        class_scores = run_tflite_model(interpreter, inp)
        model_predict_end = time.time()

        has_2_pop = float(class_scores[0][classes.index("2pop")]) > detection_threshold

        audio_processing_start = time.time()
        audio = AudioSegment.from_file(file_path)
        processed_file = os.path.join(temp_folder, os.path.basename(file_path))

        if has_2_pop:
            audio = audio[1500:]
        else:
            start_of_music = find_music_onset(file_path)
            silence_duration = max(500 - start_of_music, 0)
            audio = AudioSegment.silent(duration=silence_duration) + audio[max(0, start_of_music - 500):]

        audio.export(processed_file, format="wav")
        audio_processing_end = time.time()

        normalization_start = time.time()
        normalize_loudness_ffmpeg(processed_file, processed_file, export_level)
        normalization_end = time.time()

        print(f"Timings: Model Load: {load_model_end - load_model_start}, Preprocessing: {preprocess_end - preprocess_start}, Model Predict: {model_predict_end - model_predict_start}, Audio Processing: {audio_processing_end - audio_processing_start}, Normalization: {normalization_end - normalization_start}")

    except Exception as e:
        print(f"Error processing file {file_path}: {e}")

def process_folder(folder_path, model_path, classes, export_level, detection_threshold=0.95):
    start_time = time.time()
    temp_folder = os.path.join(folder_path, "Temp")
    shutil.rmtree(temp_folder, ignore_errors=True)
    os.makedirs(temp_folder, exist_ok=True)

    file_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(('.wav', '.aif', '.aiff', '.mp3', '.mp4'))]
    process_args = [(file_path, model_path, classes, temp_folder, detection_threshold, export_level) for file_path in file_paths]

    with multiprocessing.Pool() as pool:
        pool.starmap(process_audio_file_mp, process_args)

    shutil.move(temp_folder, os.path.join(folder_path, "Exports"))
    print(f"Total processing time: {time.time() - start_time:.2f} seconds.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process audio files for 2 pop detection and normalization.")
    parser.add_argument("folder_path", type=str, help="Path to the folder containing audio files")
    parser.add_argument("--level", type=float, default=-14, help="Export loudness level")
    args = parser.parse_args()
    process_folder(args.folder_path, 'ModelsTrained/Export/converted_model.tflite', ["Music", "2pop"], args.level)