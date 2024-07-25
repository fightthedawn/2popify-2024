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
import pyloudnorm as pyln
import onnxruntime as ort

# Function to convert audio file to WAV format
def convert_to_wav(audio_path):
    audio = AudioSegment.from_file(audio_path)
    wav_path = os.path.splitext(audio_path)[0] + ".wav"
    audio.export(wav_path, format="wav")
    return wav_path

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
    
    return audio_slice_for_2pop_detection, sr

def detect_2_pop_with_model(audio, sr, ort_session, classes, detection_threshold=0.95):
    # Ensure audio is the right shape (adjust as needed for your model)
    if len(audio) < 24000:
        audio = np.pad(audio, (0, 24000 - len(audio)))
    elif len(audio) > 24000:
        audio = audio[:24000]
    
    # Get input details
    input_details = ort_session.get_inputs()
    
    # Prepare inputs
    inputs = {}
    for input_detail in input_details:
        input_name = input_detail.name
        input_shape = input_detail.shape
        
        if input_name == 'input':
            inputs[input_name] = np.array([audio], dtype=np.float32)
        elif len(input_shape) == 0:  # Scalar inputs
            inputs[input_name] = np.array(0, dtype=np.float32 if input_detail.type == 'tensor(float)' else np.int32)
        else:
            inputs[input_name] = np.zeros(input_shape, dtype=np.float32 if input_detail.type == 'tensor(float)' else np.int32)
    
    try:
        # Run inference
        outputs = ort_session.run(None, inputs)
        
        # Process output
        if len(outputs) == 0:
            raise ValueError("Model produced no outputs")
        
        output = outputs[0]
        if output.shape != (1, 2):
            raise ValueError(f"Unexpected output shape: {output.shape}")
        
        class_scores = output[0]
        if len(class_scores) != 2:
            raise ValueError(f"Unexpected number of class scores: {len(class_scores)}")
        
        return float(class_scores[classes.index("2pop")]) > detection_threshold
    except Exception as e:
        print(f"Error in detect_2_pop_with_model: {str(e)}")
        return False

def find_music_onset(audio_path):
    try:
        if not audio_path.endswith(('.wav', '.aif', '.aiff')):
            audio_path = convert_to_wav(audio_path)

        y, sr = librosa.load(audio_path, sr=None, mono=True)
        if len(y) == 0:
            print(f"Warning: {audio_path} appears to be empty")
            return 0
        onset_frames = librosa.onset.onset_detect(y=y, sr=sr, units='time', pre_max=1, post_max=1, pre_avg=1, post_avg=1, delta=0.01, wait=1)
        return int(onset_frames[0] * 1000) if onset_frames.any() else 0
    except Exception as e:
        print(f"Error in find_music_onset for {audio_path}: {str(e)}")
        return 0

def process_audio_file(file_path, ort_session, classes, temp_folder, detection_threshold=0.95):
    try:
        print(f"Processing file: {file_path}")
        audio = AudioSegment.from_file(file_path)
        
        if len(audio) < 1500:  # Skip files shorter than 1.5 seconds
            print(f"Skipping {file_path}: File too short")
            return

        temp_file = os.path.join(temp_folder, os.path.basename(file_path))

        waveform, sr = preprocess_audio_for_model(file_path)
        has_2pop = detect_2_pop_with_model(waveform, sr, ort_session, classes, detection_threshold)

        if has_2pop:
            print(f"2 pop detected in {file_path}")
            audio_trimmed_for_onset_detection = audio[1500:]
            
            temp_trimmed_path = os.path.join(temp_folder, f"temp_for_onset_detection_{os.path.basename(file_path)}")
            audio_trimmed_for_onset_detection.export(temp_trimmed_path, format="wav")
            
            start_of_music = find_music_onset(temp_trimmed_path)
            
            actual_start_position = max(start_of_music + 1000, 0)
            
            trimmed_audio = audio[actual_start_position:]
            trimmed_audio.export(temp_file, format="wav")
            
            os.remove(temp_trimmed_path)
        else:
            print(f"No 2 pop detected in {file_path}")
            start_of_music = find_music_onset(file_path)
            silence_duration = max(500 - start_of_music, 0)
            trimmed_audio = AudioSegment.silent(duration=silence_duration, frame_rate=audio.frame_rate) + audio[max(0, start_of_music - 500):]
            trimmed_audio.export(temp_file, format="wav")

        print(f"Processed file saved to: {temp_file}")
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")

def normalize_audio(folder_path, target_peak=-0.5):
    lowest_lufs = float('inf')
    
    # First pass: normalize to peak and find the lowest LUFS
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.wav', '.aif', '.aiff')):
            file_path = os.path.join(folder_path, filename)
            audio = AudioSegment.from_file(file_path)
            
            # Normalize to target peak
            peak_normalized_audio = audio.normalize(target_peak)
            peak_normalized_audio.export(file_path, format="wav")
            
            # Measure LUFS
            data, rate = sf.read(file_path)
            meter = pyln.Meter(rate)
            loudness = meter.integrated_loudness(data)
            
            lowest_lufs = min(lowest_lufs, loudness)
    
    print(f"Lowest LUFS value across all files: {lowest_lufs}")
    
    # Second pass: adjust to the lowest LUFS
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.wav', '.aif', '.aiff')):
            file_path = os.path.join(folder_path, filename)
            data, rate = sf.read(file_path)
            meter = pyln.Meter(rate)
            loudness = meter.integrated_loudness(data)
            
            # Adjust to the lowest LUFS found
            gain_db = lowest_lufs - loudness
            adjusted_audio = AudioSegment.from_wav(file_path)
            final_audio = adjusted_audio + gain_db
            
            final_audio.export(file_path, format="wav")
            
            print(f"Normalized {filename} to {lowest_lufs:.2f} LUFS")

def process_folder(folder_path, ort_session, classes):
    start_time = time.time()
    temp_folder = os.path.join(folder_path, "Temp")

    # Remove the temp folder if it exists, then recreate it
    shutil.rmtree(temp_folder, ignore_errors=True)
    os.makedirs(temp_folder, exist_ok=True)

    # Process files in parallel
    audio_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.wav', '.aif', '.aiff', '.mp3', '.mp4'))]
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_audio_file, os.path.join(folder_path, f), ort_session, classes, temp_folder, detection_threshold=0.5) for f in audio_files]
        concurrent.futures.wait(futures)
    
    # Normalize the processed audio files in the temp folder before final export
    normalize_audio(temp_folder, target_peak=-0.5)

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process audio files for 2 pop detection.")
    parser.add_argument("folder_path", type=str, help="Path to the folder containing audio files")
    args = parser.parse_args()
    
    # Load the ONNX model
    try:
        ort_session = ort.InferenceSession("2pop_model.onnx")
        print("ONNX model loaded successfully")
    except Exception as e:
        print(f"Error loading ONNX model: {str(e)}")
        exit(1)

    classes = ["Music", "2pop"]
    
    process_folder(args.folder_path, ort_session, classes)