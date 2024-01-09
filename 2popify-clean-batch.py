import os
import numpy as np
import librosa
import tensorflow as tf
from pydub import AudioSegment, silence
import soundfile as sf
import subprocess
import argparse
import time


def convert_to_wav(audio_path):
    """Converts the audio file to WAV format."""
    audio = AudioSegment.from_file(audio_path)
    wav_path = os.path.splitext(audio_path)[0] + ".wav"
    audio.export(wav_path, format="wav")
    return wav_path

def preprocess_audio_for_model(audio_path, target_sr=16000, duration_ms=1500, normalize=True, export_temp=False):
    """Load audio, prepend silence, find the first sound onset, and preprocess a segment for the model."""
    audio, sr = librosa.load(audio_path, sr=target_sr, mono=True)
    silence_duration = int(0.5 * sr)
    audio = np.concatenate([np.zeros(silence_duration), audio])
    if normalize:
        audio = librosa.util.normalize(audio)
    onset_frames = librosa.onset.onset_detect(y=audio, sr=sr, units='samples', backtrack=True)
    onset_sample = onset_frames[0] if onset_frames.size > 0 else silence_duration
    end_sample = min(onset_sample + int(sr * (duration_ms / 1000.0)), len(audio))
    audio_segment = audio[onset_sample:end_sample]
    if len(audio_segment) < int(sr * (duration_ms / 1000.0)):
        audio_segment = np.pad(audio_segment, (0, max(0, int(sr * (duration_ms / 1000.0)) - len(audio_segment))), mode='constant')
    if export_temp:
        temp_filename = os.path.splitext(os.path.basename(audio_path))[0] + "_temp.wav"
        temp_filepath = os.path.join("temp_audio", temp_filename)
        if not os.path.exists("temp_audio"):
            os.makedirs("temp_audio")
        sf.write(temp_filepath, audio_segment, target_sr)
    return audio_segment

def detect_2_pop_with_model(audio_path, model, classes, detection_threshold):
    """Use the TensorFlow model to detect if the first 1.5 seconds of audio contains a 2 pop."""
    waveform = preprocess_audio_for_model(audio_path)
    inp = tf.constant(np.array([waveform]), dtype='float32')
    class_scores = model(inp)[0].numpy()
    two_pop_score = float(class_scores[classes.index("2pop")])
    return two_pop_score > detection_threshold

def find_music_onset(audio_path):
    """Finds the onset of music in an audio file."""
    if not audio_path.endswith('.wav'):
        audio_path = convert_to_wav(audio_path)

    y, sr = librosa.load(audio_path, sr=None, mono=True)

    # Highly sensitive parameters for onset detection
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, units='time',
                                              pre_max=1, post_max=1,
                                              pre_avg=1, post_avg=1, 
                                              delta=0.01, wait=1)

    return int(onset_frames[0] * 1000) if onset_frames.any() else 0

def normalize_loudness_ffmpeg(input_file, output_file, export_level):
    """Normalize the loudness of an audio file using FFmpeg."""
    FNULL = open(os.devnull, 'w')  # Use this to redirect FFmpeg's output
    cmd = [
        'ffmpeg',
        '-i', input_file,
        '-filter_complex', f'loudnorm=I={export_level}:TP=-1.5:LRA=11',
        '-y',  # Overwrite the output file if it exists
        output_file
    ]
    subprocess.run(cmd, stdout=FNULL, stderr=subprocess.STDOUT)
    FNULL.close()

def batch_normalize_loudness(temp_folder, exports_folder, export_level):
    """Normalizes the loudness of all audio files in the temporary folder using FFmpeg."""
    for file_name in os.listdir(temp_folder):
        input_file = os.path.join(temp_folder, file_name)
        output_file = os.path.join(exports_folder, file_name)

        cmd = [
            'ffmpeg',
            '-i', input_file,
            '-filter_complex', f'loudnorm=I={export_level}:TP=-1.5:LRA=11',
            '-y', output_file
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

def process_audio_file(file_path, model, classes, temp_folder, detection_threshold=0.95):
    """Processes an audio file and saves the trimmed audio to a temporary folder."""
    audio = AudioSegment.from_file(file_path)
    original_channels = audio.channels
    original_frame_rate = audio.frame_rate
    is_2_pop = detect_2_pop_with_model(file_path, model, classes, detection_threshold)

    if is_2_pop:
        print(f"2 pop detected in: {os.path.basename(file_path)}")
        post_2_pop_segment = audio[1500:]
        temp_file = os.path.join(temp_folder, os.path.basename(file_path))
        post_2_pop_segment.export(temp_file, format="wav")
    else:
        start_of_music = find_music_onset(file_path)
        if start_of_music < 500:
            required_silence_duration = 500 - start_of_music
            silence_audio = AudioSegment.silent(duration=required_silence_duration, frame_rate=original_frame_rate)
            audio = silence_audio + audio
        new_start_point = max(0, start_of_music - 500)
        trimmed_audio = audio[new_start_point:]
        temp_file = os.path.join(temp_folder, os.path.basename(file_path))
        trimmed_audio.set_frame_rate(original_frame_rate).set_channels(original_channels).export(temp_file, format="wav")

def process_folder(folder_path, model, classes, export_level):
    """Processes all audio files in a folder using batch normalization."""
    start_time = time.time()  # Start the timer

    temp_folder = os.path.join(folder_path, "Temp")
    exports_folder = os.path.join(folder_path, "Exports")
    os.makedirs(temp_folder, exist_ok=True)
    os.makedirs(exports_folder, exist_ok=True)

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.wav', '.aiff', '.mp3')):
            full_path = os.path.join(folder_path, filename)
            process_audio_file(full_path, model, classes, temp_folder)

    batch_normalize_loudness(temp_folder, exports_folder, export_level)

    # Clean up the temporary folder
    for file_name in os.listdir(temp_folder):
        os.remove(os.path.join(temp_folder, file_name))
    os.rmdir(temp_folder)

    end_time = time.time()  # End the timer
    print(f"Processing completed in {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process audio files for 2 pop detection and normalization.")
    parser.add_argument("folder_path", type=str, help="Path to the folder containing audio files")
    parser.add_argument("--level", type=float, default=-14, help="Export loudness level (default: -14)")
    args = parser.parse_args()
    model = tf.saved_model.load('2popmodel20000')
    classes = ["Music", "2pop"]
    process_folder(args.folder_path, model, classes, args.level)