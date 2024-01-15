import os
import librosa
import tensorflow as tf
from pydub import AudioSegment, silence
import soundfile as sf
import subprocess
import argparse
import time
import concurrent.futures
import shutil
import numpy as np

# Function to convert audio file to WAV format
def convert_to_wav(audio_path):
    audio = AudioSegment.from_file(audio_path)
    wav_path = os.path.splitext(audio_path)[0] + ".wav"
    audio.export(wav_path, format="wav")
    return wav_path

# Function to process a single audio file
def process_audio_file(file_path, model, classes, temp_folder, detection_threshold):
    audio = AudioSegment.from_file(file_path)
    sr = audio.frame_rate
    temp_file = os.path.join(temp_folder, os.path.basename(file_path))

    # Process audio data
    audio_data = np.array(audio.get_array_of_samples()).astype(np.float32) / np.iinfo(audio.sample_width * 8).max
    if detect_2_pop_with_model(audio_data, sr, model, classes, detection_threshold):
        audio[1500:].export(temp_file, format="wav")
    else:
        start_of_music = find_music_onset(file_path)
        silence_duration = max(500 - start_of_music, 0)
        trimmed_audio = AudioSegment.silent(duration=silence_duration, frame_rate=sr) + audio[max(0, start_of_music - 500):]
        trimmed_audio.set_frame_rate(sr).set_channels(audio.channels).export(temp_file, format="wav")


def detect_2_pop_with_model(audio, sr, model, classes, detection_threshold):
    waveform = preprocess_audio_for_model(audio, sr)
    inp = tf.constant(np.array([waveform]), dtype='float32')
    class_scores = model(inp)[0].numpy()
    return float(class_scores[classes.index("2pop")]) > detection_threshold

def preprocess_audio_for_model(audio, sr, duration_ms=1500, normalize=True):
    silence_duration = int(0.5 * sr)
    audio = np.concatenate([np.zeros(silence_duration), audio])
    if normalize:
        audio = librosa.util.normalize(audio)
    onset_frames = librosa.onset.onset_detect(y=audio, sr=sr, units='samples', backtrack=True)
    onset_sample = onset_frames[0] if onset_frames.size > 0 else silence_duration
    return audio[onset_sample:min(onset_sample + int(sr * (duration_ms / 1000.0)), len(audio))]

def find_music_onset(audio_path):
    """Finds the onset of music in an audio file."""
    if not audio_path.endswith(('.wav', '.aif', '.aiff')):
        audio_path = convert_to_wav(audio_path)

    y, sr = librosa.load(audio_path, sr=None, mono=True)
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, units='time', pre_max=1, post_max=1, pre_avg=1, post_avg=1, delta=0.01, wait=1)
    return int(onset_frames[0] * 1000) if onset_frames.any() else 0

def audio_editorial(audio, sr, onset_frame, is_2_pop):
    if is_2_pop:
        return audio[1500:]
    else:
        silence_duration = max(500 - onset_frame, 0)
        return silence.silent(duration=silence_duration, frame_rate=sr) + audio[max(0, onset_frame - 500):]

def normalize_and_export_audio(audio, export_path, export_level):
    FNULL = open(os.devnull, 'w')
    cmd = ['ffmpeg', '-i', audio, '-filter_complex', f'loudnorm=I={export_level}:TP=-1.5:LRA=11', '-y', export_path]
    subprocess.run(cmd, stdout=FNULL, stderr=subprocess.STDOUT)
    FNULL.close()

# Main function to process all audio files in a folder
def process_folder(folder_path, model, classes, export_level):
    start_time = time.time()
    temp_folder, exports_folder = os.path.join(folder_path, "Temp"), os.path.join(folder_path, "Exports")

    shutil.rmtree(temp_folder, ignore_errors=True)
    shutil.rmtree(exports_folder, ignore_errors=True)
    os.makedirs(temp_folder, exist_ok=True)
    os.makedirs(exports_folder, exist_ok=True)

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.wav', '.aif', '.aiff', '.mp3', '.mp4')):
            full_path = os.path.join(folder_path, filename)
            process_audio_file(full_path, model, classes, temp_folder, detection_threshold=0.95)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(normalize_audio_file, os.path.join(temp_folder, file), exports_folder, export_level) for file in os.listdir(temp_folder)]
        for future in concurrent.futures.as_completed(futures):
            future.result()

    shutil.rmtree(temp_folder)
    print(f"Total processing time: {time.time() - start_time:.2f} seconds.")

# Command-line argument setup
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process audio files for 2 pop detection and normalization.")
    parser.add_argument("folder_path", type=str, help="Path to the folder containing audio files")
    parser.add_argument("--level", type=float, default=-14, help="Export loudness level")
    args = parser.parse_args()
    model = tf.saved_model.load('ModelsTrained/2popmodel100000-20240110')
    classes = ["Music", "2pop"]
    process_folder(args.folder_path, model, classes, args.level)