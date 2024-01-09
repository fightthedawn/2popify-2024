import os
import numpy as np
import librosa
import tensorflow as tf
from pydub import AudioSegment
import soundfile as sf
import subprocess
import argparse

def convert_to_wav(audio_path):
    audio = AudioSegment.from_file(audio_path)
    wav_path = os.path.splitext(audio_path)[0] + ".wav"
    audio.export(wav_path, format="wav")
    return wav_path

def preprocess_audio_for_model(audio_path, target_sr=16000, duration_ms=1500, normalize=True):
    audio, sr = librosa.load(audio_path, sr=target_sr, mono=True)
    prepended_audio = np.concatenate([np.zeros(int(0.5 * sr)), audio]) # 0.5 seconds of silence
    if normalize:
        prepended_audio = librosa.util.normalize(prepended_audio)
    onset_frames = librosa.onset.onset_detect(y=prepended_audio, sr=sr, units='samples', backtrack=True)
    onset_sample = onset_frames[0] if onset_frames.size > 0 else int(0.5 * sr)
    end_sample = min(onset_sample + int(sr * (duration_ms / 1000.0)), len(prepended_audio))
    return prepended_audio[onset_sample:end_sample]

def detect_2_pop_with_model(audio_path, model, classes, detection_threshold):
    waveform = preprocess_audio_for_model(audio_path)
    inp = tf.constant(np.array([waveform]), dtype='float32')
    class_scores = model(inp)[0].numpy()
    return float(class_scores[classes.index("2pop")]) > detection_threshold

def find_music_onset(audio_path):
    if not audio_path.endswith('.wav'):
        audio_path = convert_to_wav(audio_path)
    y, sr = librosa.load(audio_path, sr=None, mono=True)
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, units='time')
    return int(onset_frames[0] * 1000) if onset_frames.any() else 0

def normalize_loudness_ffmpeg(input_file, output_file, export_level):
    subprocess.run([
        'ffmpeg', '-i', input_file,
        '-filter_complex', f'loudnorm=I={export_level}:TP=-1.5:LRA=11',
        '-y', output_file
    ], check=True)

def process_audio_file(file_path, model, classes, exports_folder, detection_threshold=0.95, export_level=-14):
    audio = AudioSegment.from_file(file_path)
    is_2_pop = detect_2_pop_with_model(file_path, model, classes, detection_threshold)
    start_of_music = 1500 + find_music_onset(file_path) if is_2_pop else find_music_onset(file_path)
    start_of_music = max(start_of_music, 500) # Ensuring 0.5 seconds of silence
    trimmed_audio = AudioSegment.silent(duration=start_of_music - 500, frame_rate=audio.frame_rate) + audio[start_of_music:]
    temp_trimmed_file = "temp_trimmed_audio.wav"
    trimmed_audio.export(temp_trimmed_file, format="wav")
    normalized_file_path = os.path.join(exports_folder, os.path.basename(file_path))
    normalize_loudness_ffmpeg(temp_trimmed_file, normalized_file_path, export_level)
    os.remove(temp_trimmed_file)

def process_folder(folder_path, model, classes, export_level):
    exports_folder = os.path.join(folder_path, "Exports")
    os.makedirs(exports_folder, exist_ok=True)
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.wav', '.aiff', '.mp3')):
            process_audio_file(os.path.join(folder_path, filename), model, classes, exports_folder, export_level)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process audio files for 2 pop detection and normalization.")
    parser.add_argument("folder_path", type=str, help="Path to the folder containing audio files")
    parser.add_argument("--level", type=float, default=-14, help="Export loudness level (default: -14)")
    args = parser.parse_args()
    model = tf.saved_model.load('2popmodel20000')
    classes = ["Music", "2pop"]
    process_folder(args.folder_path, model, classes, args.level)