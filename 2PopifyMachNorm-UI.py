import os
import numpy as np
import librosa
import tensorflow as tf
from pydub import AudioSegment, silence
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

def preprocess_audio_for_model(audio_path, target_sr=16000, duration_ms=1500, normalize=True, export_temp=False):
    """Load audio, prepend silence, find the first sound onset, and preprocess a segment for the model."""
    # Load the entire audio file
    audio, sr = librosa.load(audio_path, sr=target_sr, mono=True)

    # Prepend 0.5 seconds of silence
    silence_duration = int(0.5 * sr)  # 0.5 seconds in samples
    audio = np.concatenate([np.zeros(silence_duration), audio])

    # Normalize audio if enabled
    if normalize:
        audio = librosa.util.normalize(audio)

    # Find the first onset of sound after the prepended silence
    onset_frames = librosa.onset.onset_detect(y=audio, sr=sr, units='samples', backtrack=True)
    onset_sample = onset_frames[0] if onset_frames.size > 0 else silence_duration

    # Calculate end sample for the duration from the onset
    end_sample = min(onset_sample + int(sr * (duration_ms / 1000.0)), len(audio))

    # Slice the audio segment
    audio_segment = audio[onset_sample:end_sample]

    # If the segment is shorter than the desired duration, pad with zeros
    if len(audio_segment) < int(sr * (duration_ms / 1000.0)):
        audio_segment = np.pad(audio_segment, (0, max(0, int(sr * (duration_ms / 1000.0)) - len(audio_segment))), mode='constant')

    # Export the preprocessed audio segment to a temporary file for inspection
    if export_temp:
        temp_filename = os.path.splitext(os.path.basename(audio_path))[0] + "_temp.wav"
        temp_filepath = os.path.join("temp_audio", temp_filename)
        if not os.path.exists("temp_audio"):
            os.makedirs("temp_audio")
        sf.write(temp_filepath, audio_segment, target_sr)

    return audio_segment

def detect_2_pop_with_model(audio_path, model, classes, detection_threshold):
    """Use the TensorFlow model to detect if the first 1.5 seconds of audio contains a 2 pop."""
    
    # Set export_temp to True for the file you want to inspect
    #export_temp = "2020-07-02 Gold Bond GL01 r2.wav" in audio_path  # Adjust the condition as needed
    #waveform = preprocess_audio_for_model(audio_path, export_temp=export_temp)

    waveform = preprocess_audio_for_model(audio_path)
    inp = tf.constant(np.array([waveform]), dtype='float32')
    class_scores = model(inp)[0].numpy()
    
    two_pop_score = float(class_scores[classes.index("2pop")])
    music_score = float(class_scores[classes.index("Music")])

    # Debug: Print scores and threshold comparison for inspection
    #print(f"File: {os.path.basename(audio_path)}")
    #print(f"  2 pop score: {two_pop_score}, Music score: {music_score}")
    #print(f"  Exceeds threshold: {two_pop_score > detection_threshold}")
    #print(f"  2 pop detected: {'Yes' if two_pop_score > detection_threshold else 'No'}")

    return two_pop_score > detection_threshold

def find_music_onset(audio_path):
    """Finds the onset of music in an audio file."""
    if not audio_path.endswith('.wav'):
        audio_path = convert_to_wav(audio_path)

    y, sr = librosa.load(audio_path, sr=None, mono=True)
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, units='time')

    if onset_frames.any():
        return int(onset_frames[0] * 1000)  # Convert to milliseconds
    return 0

def normalize_loudness_ffmpeg(input_file, output_file):
    """Normalize the loudness of an audio file using FFmpeg."""
    cmd = [
        'ffmpeg',
        '-i', input_file,
        '-filter_complex', 'loudnorm=I=-14:TP=-1.5:LRA=11',
        '-y',  # Overwrite the output file if it exists
        output_file
    ]
    subprocess.run(cmd, check=True)

def process_audio_file(file_path, model, classes, exports_folder, detection_threshold=0.95):
    """Processes an audio file."""
    audio = AudioSegment.from_file(file_path)
    original_channels = audio.channels
    original_frame_rate = audio.frame_rate

    is_2_pop = detect_2_pop_with_model(file_path, model, classes, detection_threshold)

    if is_2_pop:
        # If a 2 pop is detected, process the segment after the first 1.5 seconds
        print(f"2 pop detected in: {os.path.basename(file_path)}")
        post_2_pop_segment = audio[1500:]  # Segment after potential 2 pop
        temp_file = "temp_post_2_pop.wav"
        post_2_pop_segment.export(temp_file, format="wav")
        start_of_music_after_2_pop = find_music_onset(temp_file)
        os.remove(temp_file)
        start_of_music = 1500 + start_of_music_after_2_pop
    else:
        # If no 2 pop is detected, find the start of music in the entire file
        start_of_music = find_music_onset(file_path)

    # Determine the new start point of the audio
    if start_of_music < 500:
        required_silence_duration = 500 - start_of_music
        silence_audio = AudioSegment.silent(duration=required_silence_duration, frame_rate=original_frame_rate)
        audio = silence_audio + audio
        new_start_point = 0
    else:
        new_start_point = max(0, start_of_music - 500)

    trimmed_audio = audio[new_start_point:]
    temp_trimmed_file = "temp_trimmed_audio.wav"
    trimmed_audio.set_frame_rate(original_frame_rate).set_channels(original_channels).export(temp_trimmed_file, format="wav")

    # Normalize the loudness of the trimmed audio file
    normalized_file_path = os.path.join(exports_folder, os.path.basename(file_path))
    normalize_loudness_ffmpeg(temp_trimmed_file, normalized_file_path)

    # Remove the temporary file
    os.remove(temp_trimmed_file)

def process_folder(folder_path, model, classes):
    """Processes all audio files in a folder."""
    exports_folder = os.path.join(folder_path, "Exports")
    if not os.path.exists(exports_folder):
        os.makedirs(exports_folder)

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.wav', '.aiff', '.mp3')):
            full_path = os.path.join(folder_path, filename)
            process_audio_file(full_path, model, classes, exports_folder)

# Load the TensorFlow model and class names
model = tf.saved_model.load('2popmodel20000')  # Replace with the path to your model directory
classes = ["Music", "2pop"]

# Folder containing audio files
folder_path = "TestData/TestData5"  # Replace with your folder path

# Process the folder
process_folder(folder_path, model, classes)