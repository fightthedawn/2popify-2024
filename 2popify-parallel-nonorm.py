import os
import numpy as np
from scipy.signal import butter, lfilter
import noisereduce as nr
import librosa
import tensorflow as tf
import soundfile as sf
from pydub import AudioSegment
import argparse
import time
import concurrent.futures
import shutil
from autolevel import normalize_audio

# Function to convert audio file to WAV format
def convert_to_wav(audio_path):
    audio = AudioSegment.from_file(audio_path)
    wav_path = os.path.splitext(audio_path)[0] + ".wav"
    audio.export(wav_path, format="wav")
    return wav_path

def butter_bandpass(lowcut, highcut, sr, order=5):
    nyq = 0.5 * sr
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, sr, order=5):
    b, a = butter_bandpass(lowcut, highcut, sr, order=order)
    y = lfilter(b, a, data)
    return y

'''# Function to preprocess audio for the model
def preprocess_audio_for_model(audio_path, target_sr=16000, duration_ms=1500, normalize=True):
    # Load the audio file
    audio, sr = librosa.load(audio_path, sr=target_sr, mono=True)
    
    # Apply noise reduction
    audio_reduced_noise = nr.reduce_noise(y=audio, sr=sr)
    
    # Proceed with your existing preprocessing steps, starting with bandpass filtering
    lowcut = 140
    highcut = 2200
    order = 5
    nyq = 0.5 * sr
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    
    # Apply the bandpass filter on the noise-reduced audio
    audio_filtered = lfilter(b, a, audio_reduced_noise)
    
    # Continue with the addition of silence, normalization, and onset detection
    silence_duration = int(0.5 * sr)
    audio_filtered_with_silence = np.concatenate([np.zeros(silence_duration), audio_filtered])
    if normalize:
        audio_filtered_with_silence = librosa.util.normalize(audio_filtered_with_silence)
    
    onset_frames = librosa.onset.onset_detect(y=audio_filtered_with_silence, sr=sr, units='samples', backtrack=True)
    onset_sample = onset_frames[0] if onset_frames.size > 0 else silence_duration
    end_sample = min(onset_sample + int(sr * (duration_ms / 1000.0)), len(audio_filtered_with_silence))
    
    # Ensure the directory exists
    export_dir = "2popIso"
    os.makedirs(export_dir, exist_ok=True)
    
    # Change file extension to .wav for export
    export_basename = os.path.splitext(os.path.basename(audio_path))[0] + ".wav"
    export_path = os.path.join(export_dir, export_basename)
    
    # Export the processed audio as .wav
    sf.write(export_path, audio_filtered_with_silence[onset_sample:end_sample], sr)
    
    return audio_filtered_with_silence[onset_sample:end_sample], sr
'''

# Function to preprocess audio for the model
def preprocess_audio_for_model(audio_path, target_sr=16000, normalize=True):
    # Load the audio file
    audio, sr = librosa.load(audio_path, sr=target_sr, mono=True)
    
    # Apply noise reduction
    audio_reduced_noise = nr.reduce_noise(y=audio, sr=sr)
    
    # Proceed with your existing preprocessing steps, starting with bandpass filtering
    lowcut = 140
    highcut = 2200
    order = 5
    nyq = 0.5 * sr
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    
    # Apply the bandpass filter on the noise-reduced audio
    audio_filtered = lfilter(b, a, audio_reduced_noise)
    
    # Instead of adding silence and normalizing, directly slice the first 1500ms for 2pop detection
    # This step bypasses onset detection and uses the first 1500ms of the filtered audio
    audio_slice_for_2pop_detection = audio_filtered[:int(1.5 * sr)]
    
    '''
    # Optionally, export this slice for review
    export_dir = "2popIso"
    os.makedirs(export_dir, exist_ok=True)
    file_name = os.path.splitext(os.path.basename(audio_path))[0] + "_2pop_slice.wav"
    export_path = os.path.join(export_dir, file_name)
    sf.write(export_path, audio_slice_for_2pop_detection, sr)
    '''
    return audio_slice_for_2pop_detection, sr

# Function to detect a 2 pop in the audio using the trained model
def detect_2_pop_with_model(audio_path, model, classes, detection_threshold):
    waveform, sr = preprocess_audio_for_model(audio_path)  # Adjusted to unpack the returned tuple
    inp = tf.constant(np.array([waveform]), dtype='float32')  # waveform is now correctly referenced
    class_scores = model(inp)[0].numpy()
    return float(class_scores[classes.index("2pop")]) > detection_threshold

# Function to find the onset of music in an audio file
def find_music_onset(audio_path):
    if not audio_path.endswith(('.wav', '.aif', '.aiff')):
        audio_path = convert_to_wav(audio_path)

    y, sr = librosa.load(audio_path, sr=None, mono=True)
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, units='time', pre_max=1, post_max=1, pre_avg=1, post_avg=1, delta=0.01, wait=1)
    return int(onset_frames[0] * 1000) if onset_frames.any() else 0

# Function to process an audio file and export it to a temporary folder
def process_audio_file(file_path, model, classes, temp_folder, detection_threshold=0.95):
    audio = AudioSegment.from_file(file_path)
    temp_file = os.path.join(temp_folder, os.path.basename(file_path))

    if detect_2_pop_with_model(file_path, model, classes, detection_threshold):
        # If a 2 pop is detected, trim the first 1500ms from the audio
        audio_trimmed_for_onset_detection = audio[1500:]
        
        # Export the trimmed audio temporarily for onset detection
        temp_trimmed_path = os.path.join(temp_folder, "temp_for_onset_detection.wav")
        audio_trimmed_for_onset_detection.export(temp_trimmed_path, format="wav")
        
        # Now use the trimmed audio to find the music onset
        start_of_music = find_music_onset(temp_trimmed_path)  # Using the trimmed file
        
        # Calculate the actual start position in the original audio
        # 1500ms for the 2 pop + detected onset - 500ms before the onset for the cut
        actual_start_position = max(start_of_music + 1000, 0)  # Ensure it doesn't go negative
        
        # Trim the original audio based on the calculated start position and export
        trimmed_audio = audio[actual_start_position:]
        trimmed_audio.export(temp_file, format="wav")
        
        # Clean up the temporary file used for onset detection
        os.remove(temp_trimmed_path)

    else:
        # If no 2 pop is detected, use the original logic for music onset detection
        start_of_music = find_music_onset(file_path)
        silence_duration = max(500 - start_of_music, 0)
        trimmed_audio = AudioSegment.silent(duration=silence_duration, frame_rate=audio.frame_rate) + audio[max(0, start_of_music - 500):]
        trimmed_audio.export(temp_file, format="wav")

# Main function to process all audio files in a folder
def process_folder(folder_path, model, classes):
    start_time = time.time()
    temp_folder = os.path.join(folder_path, "Temp")

    # Remove the temp folder if it exists, then recreate it
    shutil.rmtree(temp_folder, ignore_errors=True)
    os.makedirs(temp_folder, exist_ok=True)

    # Process each audio file and save it to the temp folder
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.wav', '.aif', '.aiff', '.mp3', '.mp4')):
            process_audio_file(os.path.join(folder_path, filename), model, classes, temp_folder, detection_threshold=0.5)
    
    # Normalize the processed audio files in the temp folder before final export
    normalize_audio(temp_folder)  # This function is called to normalize files directly in the temp folder

    exports_folder = os.path.join(folder_path, "Exports")
    # Check if exports folder exists and remove it if it does
    if os.path.exists(exports_folder):
        shutil.rmtree(exports_folder)
    os.makedirs(exports_folder, exist_ok=True)  # Ensure the exports folder is created

    # Move normalized files from the temp folder to the exports folder
    for file_name in os.listdir(temp_folder):
        src_file_path = os.path.join(temp_folder, file_name)
        dst_file_path = os.path.join(exports_folder, file_name)
        shutil.move(src_file_path, dst_file_path)

    # Clean up the temporary folder after moving its contents
    shutil.rmtree(temp_folder)
    print(f"Total processing time: {time.time() - start_time:.2f} seconds.\n")

# Load model, parse arguments, and run the main function
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process audio files for 2 pop detection.")
    parser.add_argument("folder_path", type=str, help="Path to the folder containing audio files")
    args = parser.parse_args()
    model = tf.saved_model.load('ModelsTrained/2popmodel100000-20240110')
    classes = ["Music", "2pop"]
    process_folder(args.folder_path, model, classes)