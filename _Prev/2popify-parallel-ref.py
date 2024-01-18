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

# Existing functions (no changes needed)

def convert_to_wav(audio_path):
    audio = AudioSegment.from_file(audio_path)
    wav_path = os.path.splitext(audio_path)[0] + ".wav"
    audio.export(wav_path, format="wav")
    return wav_path

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

def detect_2_pop_with_model(audio_path, model, classes, detection_threshold):
    waveform = preprocess_audio_for_model(audio_path)
    inp = tf.constant(np.array([waveform]), dtype='float32')
    class_scores = model(inp)[0].numpy()
    return float(class_scores[classes.index("2pop")]) > detection_threshold

def find_music_onset(audio_path):
    if not audio_path.endswith(('.wav', '.aif', '.aiff')):
        audio_path = convert_to_wav(audio_path)

    y, sr = librosa.load(audio_path, sr=None, mono=True)
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, units='time', pre_max=1, post_max=1, pre_avg=1, post_avg=1, delta=0.01, wait=1)
    return int(onset_frames[0] * 1000) if onset_frames.any() else 0

def normalize_loudness_ffmpeg(input_file, output_file, export_level, timeout=60):
    cmd = ['ffmpeg', '-i', input_file, '-filter_complex', f'loudnorm=I={export_level}:TP=-1.5:LRA=11', '-y', output_file]
    try:
        # Run FFmpeg and wait for it to complete with a timeout
        process = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
        print(f"FFmpeg Output: {process.stdout.decode('utf-8')}")
        print(f"FFmpeg Errors: {process.stderr.decode('utf-8')}")
    except subprocess.TimeoutExpired:
        print(f"FFmpeg command timed out on file {input_file}")
    except Exception as e:
        print(f"Error running FFmpeg on file {input_file}: {e}")

# Combined and parallelized function

def process_and_normalize_audio(file_path, model, classes, temp_folder, exports_folder, detection_threshold, export_level):
    # Process the audio file
    print(f"Starting processing for {file_path}")
    
    print(f"Reading file: {file_path}")
    audio = AudioSegment.from_file(file_path)
    
    temp_file = os.path.join(temp_folder, os.path.basename(file_path))

    print("Checking for 2 pop...")
    if detect_2_pop_with_model(file_path, model, classes, detection_threshold):
        print("2 pop detected. Processing...")
        audio[1500:].export(temp_file, format="wav")
    else:
        print("No 2 pop detected. Finding music onset...")
        start_of_music = find_music_onset(file_path)
        silence_duration = max(500 - start_of_music, 0)
        trimmed_audio = AudioSegment.silent(duration=silence_duration, frame_rate=audio.frame_rate) + audio[max(0, start_of_music - 500):]
        trimmed_audio.set_frame_rate(audio.frame_rate).set_channels(audio.channels).export(temp_file, format="wav")

    print(f"Completed processing for {file_path}. Now normalizing...")
    
    # Normalize and export the audio file
    output_file = os.path.join(exports_folder, os.path.basename(file_path))
    normalize_loudness_ffmpeg(temp_file, output_file, export_level)
    print(f"Processing completed for {file_path}")

# Main function with parallelization
def process_folder_parallel(folder_path, model, classes, export_level, detection_threshold=0.95):
    print("Starting processing with a single thread")
    start_time = time.time()
    
    temp_folder = os.path.join(folder_path, "Temp")
    exports_folder = os.path.join(folder_path, "Exports")

    shutil.rmtree(temp_folder, ignore_errors=True)
    shutil.rmtree(exports_folder, ignore_errors=True)
    os.makedirs(temp_folder, exist_ok=True)
    os.makedirs(exports_folder, exist_ok=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        futures = []
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(('.wav', '.aif', '.aiff', '.mp3', '.mp4')):
                file_path = os.path.join(folder_path, filename)
                print(f"Submitting {file_path} for processing")
                future = executor.submit(process_and_normalize_audio, file_path, model, classes, temp_folder, exports_folder, detection_threshold, export_level)
                futures.append(future)

        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
                print("One file processed")
            except Exception as e:
                print(f"Error processing a file: {e}")

    shutil.rmtree(temp_folder)
    total_time = time.time() - start_time
    print(f"Total processing time: {total_time:.2f} seconds.")

# Updated __main__ section to call the new parallel processing function

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process audio files for 2 pop detection and normalization.")
    parser.add_argument("folder_path", type=str, help="Path to the folder containing audio files")
    parser.add_argument("--level", type=float, default=-14, help="Export loudness level")
    args = parser.parse_args()
    model = tf.saved_model.load('ModelsTrained/2popmodel100000-20240110')
    classes = ["Music", "2pop"]
    process_folder_parallel(args.folder_path, model, classes, args.level)
