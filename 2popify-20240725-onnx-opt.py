import os
import numpy as np
from scipy import signal
import soundfile as sf
from pydub import AudioSegment
import argparse
import time
import concurrent.futures
import shutil
import pyloudnorm as pyln
import onnxruntime as ort
import math
import librosa
import noisereduce as nr
from scipy.signal import butter, lfilter

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = signal.butter(order, [low, high], btype='band', output='sos')
    return sos

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    sos = butter_bandpass(lowcut, highcut, fs, order=order)
    y = signal.sosfilt(sos, data)
    return y

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

def detect_2_pop_with_model(audio, sr, ort_session, detection_threshold=0.5):
    if len(audio) < 24000:
        audio = np.pad(audio, (0, 24000 - len(audio)))
    elif len(audio) > 24000:
        audio = audio[:24000]
    
    inputs = {}
    for input_meta in ort_session.get_inputs():
        input_name = input_meta.name
        input_shape = input_meta.shape
        if input_name == 'input':
            inputs[input_name] = np.array([audio], dtype=np.float32)
        elif ':0' in input_name:
            if input_meta.type == 'tensor(float)':
                if len(input_shape) == 0:
                    inputs[input_name] = np.array(0.0, dtype=np.float32)
                else:
                    inputs[input_name] = np.zeros(input_shape, dtype=np.float32)
            elif input_meta.type == 'tensor(int32)':
                if len(input_shape) == 0:
                    inputs[input_name] = np.array(0, dtype=np.int32)
                else:
                    inputs[input_name] = np.zeros(input_shape, dtype=np.int32)
            else:
                inputs[input_name] = np.zeros(input_shape, dtype=np.float32)

    outputs = ort_session.run(None, inputs)
    return outputs[0][0][1] > detection_threshold

def find_music_onset(audio_path):
    if not audio_path.endswith(('.wav', '.aif', '.aiff')):
        audio_path = convert_to_wav(audio_path)

    y, sr = librosa.load(audio_path, sr=None, mono=True)
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, units='time', pre_max=1, post_max=1, pre_avg=1, post_avg=1, delta=0.01, wait=1)
    return int(onset_frames[0] * 1000) if onset_frames.any() else 0

def process_audio_file(file_path, ort_session, temp_folder, detection_threshold):
    try:
        print(f"Processing: {os.path.basename(file_path)}")
        audio = AudioSegment.from_file(file_path)
        
        if len(audio) < 1500:  # Skip files shorter than 1.5 seconds
            print(f"Skipping {file_path}: File too short")
            return False
        
        temp_file = os.path.join(temp_folder, os.path.basename(file_path))
        waveform, sr = preprocess_audio_for_model(file_path)
        try:
            has_2pop = detect_2_pop_with_model(waveform, sr, ort_session, detection_threshold)
        except Exception as e:
            print(f"Error in 2-pop detection for {file_path}: {str(e)}")
            return False

        if has_2pop:
            print(f"2 pop detected in {os.path.basename(file_path)}")
            audio_trimmed_for_onset_detection = audio[1500:]
            
            temp_trimmed_path = os.path.join(temp_folder, f"temp_for_onset_detection_{os.path.basename(file_path)}")
            audio_trimmed_for_onset_detection.export(temp_trimmed_path, format="wav")
            
            start_of_music = find_music_onset(temp_trimmed_path)
            
            actual_start_position = max(start_of_music + 1000, 0)
            
            audio = audio[actual_start_position + 1500:]
        else:
            print(f"No 2 pop in {os.path.basename(file_path)}")
            start_of_music = find_music_onset(file_path)
            silence_duration = max(500 - start_of_music, 0)
            audio = AudioSegment.silent(duration=silence_duration, frame_rate=audio.frame_rate) + audio[max(0, start_of_music - 500):]
        
        # Ensure stereo output
        if audio.channels > 2:
            audio = audio.set_channels(2)
        elif audio.channels == 1:
            audio = audio.set_channels(2)

        audio.export(temp_file, format="wav")
        print(f"Saved: {os.path.basename(temp_file)}")
        return has_2pop
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return False

def normalize_audio(folder_path, target_peak=-1, target_lufs=None):
    files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.wav', '.aif', '.aiff'))]
    lowest_lufs = float('inf')
    
    # First pass: find the lowest LUFS
    for filename in files:
        file_path = os.path.join(folder_path, filename)
        audio = AudioSegment.from_file(file_path)
        
        # Convert to stereo if more than 2 channels
        if audio.channels > 2:
            audio = audio.set_channels(2)
        
        normalized_audio = audio.normalize(target_peak)
        normalized_audio.export(file_path, format="wav")
        
        data, rate = sf.read(file_path)
        
        # Ensure data is 2D (stereo)
        if data.ndim == 1:
            data = np.column_stack((data, data))
        elif data.ndim > 2:
            data = data[:, :2]
        
        meter = pyln.Meter(rate)
        loudness = meter.integrated_loudness(data)
        lowest_lufs = min(lowest_lufs, loudness)
    
    # Round down to the nearest whole number if no target_lufs is specified
    if target_lufs is None:
        target_lufs = math.floor(lowest_lufs)
    
    print(f"Lowest LUFS: {lowest_lufs:.2f}")
    print(f"Target LUFS: {target_lufs:.2f}")
    
    # Second pass: normalize all files to the target LUFS
    for filename in files:
        file_path = os.path.join(folder_path, filename)
        data, rate = sf.read(file_path)
        
        # Ensure data is 2D (stereo)
        if data.ndim == 1:
            data = np.column_stack((data, data))
        elif data.ndim > 2:
            data = data[:, :2]
        
        loudness = pyln.Meter(rate).integrated_loudness(data)
        normalized_audio = pyln.normalize.loudness(data, loudness, target_lufs)
        
        # Always write as WAV
        output_path = os.path.splitext(file_path)[0] + '.wav'
        sf.write(output_path, normalized_audio, rate)
        print(f"Normalized {filename} to {target_lufs:.2f} LUFS")
        
        # If the original file was not WAV, remove it
        if not file_path.lower().endswith('.wav'):
            os.remove(file_path)

def process_folder(folder_path, ort_session, detection_threshold, target_peak, target_lufs):
    start_time = time.time()
    temp_folder = os.path.join(folder_path, "Temp")
    exports_folder = os.path.join(folder_path, "Exports")

    shutil.rmtree(temp_folder, ignore_errors=True)
    os.makedirs(temp_folder, exist_ok=True)

    audio_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.wav', '.aif', '.aiff', '.mp3', '.mp4'))]
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_audio_file, os.path.join(folder_path, f), ort_session, temp_folder, detection_threshold) for f in audio_files]
        concurrent.futures.wait(futures)
    
    normalize_audio(temp_folder, target_peak, target_lufs)

    shutil.rmtree(exports_folder, ignore_errors=True)
    shutil.move(temp_folder, exports_folder)
    
    # Rename any remaining .aif or .aiff files to .wav
    for filename in os.listdir(exports_folder):
        if filename.lower().endswith(('.aif', '.aiff')):
            old_path = os.path.join(exports_folder, filename)
            new_path = os.path.join(exports_folder, os.path.splitext(filename)[0] + '.wav')
            os.rename(old_path, new_path)
    
    print(f"\nProcessed {len(audio_files)} files")
    print(f"2-pops detected: {sum(1 for f in futures if f.result())}")
    print(f"Total processing time: {time.time() - start_time:.2f} seconds")

def print_model_info(ort_session):
    print("Model Inputs:")
    for i, input_meta in enumerate(ort_session.get_inputs()):
        print(f"Input {i}:")
        print(f"  Name: {input_meta.name}")
        print(f"  Shape: {input_meta.shape}")
        print(f"  Type: {input_meta.type}")
    
    print("\nModel Outputs:")
    for i, output_meta in enumerate(ort_session.get_outputs()):
        print(f"Output {i}:")
        print(f"  Name: {output_meta.name}")
        print(f"  Shape: {output_meta.shape}")
        print(f"  Type: {output_meta.type}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process audio files for 2 pop detection and normalization.")
    parser.add_argument("folder_path", type=str, help="Path to the folder containing audio files")
    parser.add_argument("--threshold", type=float, default=0.5, help="2-pop detection threshold")
    parser.add_argument("--peak", type=float, default=-1, help="Target peak level in dBFS")
    parser.add_argument("--lufs", type=float, help="Target LUFS level (optional)")
    args = parser.parse_args()
    
    try:
        ort_session = ort.InferenceSession("2pop_model.onnx")
        print("ONNX model loaded successfully")
        print_model_info(ort_session)
    except Exception as e:
        print(f"Error loading ONNX model: {str(e)}")
        exit(1)

    process_folder(args.folder_path, ort_session, args.threshold, args.peak, args.lufs)