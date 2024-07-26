import os
import numpy as np
from scipy import signal
import noisereduce as nr
import librosa
import soundfile as sf
from pydub import AudioSegment
import argparse
import time
import concurrent.futures
import shutil
import pyloudnorm as pyln
import onnxruntime as ort
from scipy.io import wavfile
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

def preprocess_audio_for_model(audio_path, target_sr=16000):
    audio, sr = librosa.load(audio_path, sr=target_sr, mono=True)
    audio_reduced_noise = nr.reduce_noise(y=audio, sr=sr)
    sos = signal.butter(5, [140 / (0.5 * sr), 2200 / (0.5 * sr)], btype='band', output='sos')
    return signal.sosfilt(sos, audio_reduced_noise), sr

def detect_2_pop_with_model(audio_path, ort_session, detection_threshold=0.95):
    audio, _ = librosa.load(audio_path, sr=16000, duration=1.5)
    audio = np.pad(audio, (0, max(0, 24000 - len(audio))))[:24000]
    inputs = {ort_session.get_inputs()[0].name: np.array([audio], dtype=np.float32)}
    for input_detail in ort_session.get_inputs()[1:]:
        input_shape = input_detail.shape
        dtype = np.float32 if input_detail.type == 'tensor(float)' else np.int32
        inputs[input_detail.name] = np.zeros(input_shape, dtype=dtype)
    outputs = ort_session.run(None, inputs)
    return outputs[0][0][1] > detection_threshold

def find_music_onset(audio_path):
    y, sr = librosa.load(audio_path, sr=None, mono=True)
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, units='time', pre_max=1, post_max=1, pre_avg=1, post_avg=1, delta=0.01, wait=1)
    return int(onset_frames[0] * 1000) if onset_frames.size > 0 else 0

def process_audio_file(file_path, ort_session, temp_folder, detection_threshold=0.5):
    try:
        audio = AudioSegment.from_file(file_path)
        if len(audio) < 1500:
            return False, f"Skipped: {os.path.basename(file_path)} (too short)"

        temp_file = os.path.join(temp_folder, os.path.basename(file_path))
        has_2pop = detect_2_pop_with_model(file_path, ort_session, detection_threshold)

        if has_2pop:
            audio_trimmed = audio[1500:]
            temp_trimmed_path = os.path.join(temp_folder, f"temp_{os.path.basename(file_path)}")
            audio_trimmed.export(temp_trimmed_path, format="wav")
            start_of_music = find_music_onset(temp_trimmed_path)
            actual_start_position = max(start_of_music - 500, 0)
            trimmed_audio = audio[1500 + actual_start_position:]
            silence_duration = max(500 - actual_start_position, 0)
            os.remove(temp_trimmed_path)
        else:
            start_of_music = find_music_onset(file_path)
            silence_duration = max(500 - start_of_music, 0)
            trimmed_audio = audio[max(0, start_of_music - 500):]

        trimmed_audio = AudioSegment.silent(duration=silence_duration, frame_rate=audio.frame_rate) + trimmed_audio
        trimmed_audio = trimmed_audio.set_channels(2)
        trimmed_audio.export(temp_file, format="wav")
        return has_2pop, os.path.basename(file_path)
    except Exception as e:
        return False, f"Error: {os.path.basename(file_path)} - {str(e)}"

def normalize_audio(folder_path, target_peak=-0.5, executor=None):
    audio_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.wav', '.aif', '.aiff'))]
    
    def process_file(file_path):
        rate, data = wavfile.read(file_path)
        if data.dtype != np.float32:
            data = data.astype(np.float32) / np.iinfo(data.dtype).max
        peak = np.max(np.abs(data))
        normalized_data = data * (10 ** (target_peak / 20)) / peak
        wavfile.write(file_path, rate, (normalized_data * 32767).astype(np.int16))
        loudness = pyln.Meter(rate).integrated_loudness(normalized_data)
        return loudness, file_path

    if executor:
        futures = [executor.submit(process_file, os.path.join(folder_path, f)) for f in audio_files]
        results = [future.result() for future in concurrent.futures.as_completed(futures)]
    else:
        results = [process_file(os.path.join(folder_path, f)) for f in audio_files]
    
    lowest_lufs = min(result[0] for result in results)
    
    def adjust_loudness(file_path):
        rate, data = wavfile.read(file_path)
        data = data.astype(np.float32) / 32767
        loudness = pyln.Meter(rate).integrated_loudness(data)
        gain_db = lowest_lufs - loudness
        adjusted_data = data * (10 ** (gain_db / 20))
        wavfile.write(file_path, rate, (adjusted_data * 32767).astype(np.int16))

    if executor:
        list(executor.map(adjust_loudness, (result[1] for result in results)))
    else:
        for result in results:
            adjust_loudness(result[1])
    
    return lowest_lufs

def process_folder(folder_path, ort_session):
    start_time = time.time()
    temp_folder = os.path.join(folder_path, "Temp")
    exports_folder = os.path.join(folder_path, "Exports")

    shutil.rmtree(temp_folder, ignore_errors=True)
    os.makedirs(temp_folder, exist_ok=True)

    audio_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.wav', '.aif', '.aiff', '.mp3', '.mp4'))]
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        future_to_file = {executor.submit(process_audio_file, os.path.join(folder_path, f), ort_session, temp_folder, 0.5): f for f in audio_files}
        results = []
        for future in concurrent.futures.as_completed(future_to_file):
            file = future_to_file[future]
            try:
                results.append(future.result())
            except Exception as exc:
                print(f'{file} generated an exception: {exc}')
        
        final_lufs = normalize_audio(temp_folder, target_peak=-0.5, executor=executor)

    shutil.rmtree(exports_folder, ignore_errors=True)
    shutil.move(temp_folder, exports_folder)
    
    print("\nProcessing Results:")
    for has_2pop, filename in results:
        status = "2 pop detected" if has_2pop else "No 2 pop detected"
        print(f"{filename}: {status}")
    
    print(f"\nAll files normalized to {final_lufs:.2f} LUFS")
    print(f"Total processing time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process audio files for 2 pop detection.")
    parser.add_argument("folder_path", type=str, help="Path to the folder containing audio files")
    args = parser.parse_args()
    
    try:
        # Suppress ONNX Runtime output
        ort.set_default_logger_severity(3)
        available_providers = ort.get_available_providers()
        if 'CoreMLExecutionProvider' in available_providers:
            providers = ['CoreMLExecutionProvider', 'CPUExecutionProvider']
        elif 'CUDAExecutionProvider' in available_providers:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']
        
        ort_session = ort.InferenceSession("2pop_model.onnx", providers=providers)
        
        # Get the name of the provider actually being used
        used_provider = ort_session.get_providers()[0]
        
        if used_provider == 'CoreMLExecutionProvider':
            device = "Apple Neural Engine"
        elif used_provider == 'CUDAExecutionProvider':
            device = "NVIDIA GPU"
        else:
            device = "CPU"
        
        print(f"ONNX model loaded successfully using {device}")
    except Exception as e:
        print(f"Error loading ONNX model: {str(e)}")
        exit(1)

    process_folder(args.folder_path, ort_session)