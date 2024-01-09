import os
import numpy as np
import librosa
import tensorflow as tf
from pydub import AudioSegment
import soundfile as sf
import subprocess
import tkinter as tk
from tkinter import filedialog, simpledialog

def convert_to_wav(audio_path):
    """Converts the audio file to WAV format."""
    audio = AudioSegment.from_file(audio_path)
    wav_path = os.path.splitext(audio_path)[0] + ".wav"
    audio.export(wav_path, format="wav")
    return wav_path

def preprocess_audio_for_model(audio_path, target_sr=16000, duration_ms=1500, normalize=True):
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

    return audio_segment

def detect_2_pop_with_model(audio_path, model, classes, detection_threshold):
    waveform = preprocess_audio_for_model(audio_path)
    inp = tf.constant(np.array([waveform]), dtype='float32')
    class_scores = model(inp)[0].numpy()
    
    two_pop_score = float(class_scores[classes.index("2pop")])
    return two_pop_score > detection_threshold

def find_music_onset(audio_path):
    if not audio_path.endswith('.wav'):
        audio_path = convert_to_wav(audio_path)

    y, sr = librosa.load(audio_path, sr=None, mono=True)
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, units='time')

    if onset_frames.any():
        return int(onset_frames[0] * 1000)
    return 0

def normalize_loudness_ffmpeg(input_file, output_file, export_level):
    """Normalize the loudness of an audio file using FFmpeg."""
    cmd = [
        'ffmpeg',
        '-i', input_file,
        '-filter_complex', f'loudnorm=I={export_level}:TP=-1.5:LRA=11',
        '-y',
        output_file
    ]
    subprocess.run(cmd, check=True)

def process_audio_file(file_path, model, classes, exports_folder, detection_threshold=0.95, export_level=-14):
    """Processes an audio file."""
    audio = AudioSegment.from_file(file_path)
    original_frame_rate = audio.frame_rate

    is_2_pop = detect_2_pop_with_model(file_path, model, classes, detection_threshold)

    if is_2_pop:
        post_2_pop_segment = audio[1500:]
        temp_file = "temp_post_2_pop.wav"
        post_2_pop_segment.export(temp_file, format="wav")
        start_of_music_after_2_pop = find_music_onset(temp_file)
        os.remove(temp_file)
        start_of_music = 1500 + start_of_music_after_2_pop
    else:
        start_of_music = find_music_onset(file_path)

    start_of_music = max(start_of_music, 500)
    trimmed_audio = audio[start_of_music - 500:]
    export_path = os.path.join(exports_folder, os.path.basename(file_path))
    trimmed_audio.export(export_path, format="wav")

    normalize_loudness_ffmpeg(export_path, export_path, export_level)

def process_folder(folder_path, model, classes, export_level):
    """Processes all audio files in a folder."""
    exports_folder = os.path.join(folder_path, "Exports")
    if not os.path.exists(exports_folder):
        os.makedirs(exports_folder)

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.wav', '.aiff', '.mp3')):
            full_path = os.path.join(folder_path, filename)
            process_audio_file(full_path, model, classes, exports_folder, detection_threshold=0.95, export_level=export_level)

# Tkinter GUI for folder selection and setting export level
def select_folder():
    folder_path = filedialog.askdirectory()
    if folder_path:
        process_folder(folder_path, model, classes, export_level)

def set_export_level():
    global export_level
    level = simpledialog.askfloat("Export Level", "Enter the export level (e.g., -14):", minvalue=-100, maxvalue=0)
    if level is not None:
        export_level = level

# Load the TensorFlow model and class names
model = tf.saved_model.load('2popmodel20000')  # Replace with the path to your model directory
classes = ["Music", "2pop"]

export_level = -14  # Default export level

# Tkinter GUI setup
root = tk.Tk()
root.title("Audio Processing App")

select_folder_btn = tk.Button(root, text="Select Folder", command=select_folder)
select_folder_btn.pack(pady=10)

export_level_btn = tk.Button(root, text="Set Export Level", command=set_export_level)
export_level_btn.pack(pady=10)

root.mainloop()
