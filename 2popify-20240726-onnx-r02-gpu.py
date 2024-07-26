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

def preprocess_audio_for_model(audio_path, target_sr=16000):
    audio, sr = librosa.load(audio_path, sr=target_sr, mono=True)
    audio_reduced_noise = nr.reduce_noise(y=audio, sr=sr)
    
    sos = signal.butter(5, [140 / (0.5 * sr), 2200 / (0.5 * sr)], btype='band', output='sos')
    audio_filtered = signal.sosfilt(sos, audio_reduced_noise)
    return audio_filtered, sr

def detect_2_pop_with_model(audio_path, ort_session, detection_threshold=0.95):
    # Load only the first 1.5 seconds of audio
    audio, sr = librosa.load(audio_path, sr=16000, duration=1.5)
    audio = np.pad(audio, (0, max(0, 24000 - len(audio))))[:24000]
    inputs = {ort_session.get_inputs()[0].name: np.array([audio], dtype=np.float32)}
    for input_detail in ort_session.get_inputs()[1:]:
        input_shape = input_detail.shape
        dtype = np.float32 if input_detail.type == 'tensor(float)' else np.int32
        inputs[input_detail.name] = np.zeros(input_shape, dtype=dtype)
    outputs = ort_session.run(None, inputs)
    return outputs[0][0][1] > detection_threshold

def find_music_onset(audio_path):
    if not audio_path.endswith(('.wav', '.aif', '.aiff')):
        audio_path = convert_to_wav(audio_path)
    y, sr = librosa.load(audio_path, sr=None, mono=True)
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, units='time', pre_max=1, post_max=1, pre_avg=1, post_avg=1, delta=0.01, wait=1)
    return int(onset_frames[0] * 1000) if onset_frames.size > 0 else 0

def process_audio_file(file_path, ort_session, temp_folder, detection_threshold=0.5):
    try:
        audio = AudioSegment.from_file(file_path)
        print(f"Processing {file_path}, length: {len(audio)} ms")
        
        if len(audio) < 1500:
            return False, f"Skipped: {os.path.basename(file_path)} (too short)"

        temp_file = os.path.join(temp_folder, os.path.basename(file_path))
        
        has_2pop = detect_2_pop_with_model(file_path, ort_session, detection_threshold)
        print(f"2-pop detected: {has_2pop}")

        if has_2pop:
            audio_trimmed = audio[1500:]
            temp_trimmed_path = os.path.join(temp_folder, f"temp_{os.path.basename(file_path)}")
            audio_trimmed.export(temp_trimmed_path, format="wav")
            start_of_music = find_music_onset(temp_trimmed_path)
            print(f"Start of music after 2-pop: {start_of_music} ms")
            actual_start_position = max(start_of_music - 500, 0)
            trimmed_audio = audio[1500 + actual_start_position:]
            silence_duration = max(500 - actual_start_position, 0)
            os.remove(temp_trimmed_path)
        else:
            start_of_music = find_music_onset(file_path)
            print(f"Start of music (no 2-pop): {start_of_music} ms")
            silence_duration = max(500 - start_of_music, 0)
            trimmed_audio = audio[max(0, start_of_music - 500):]

        trimmed_audio = AudioSegment.silent(duration=silence_duration, frame_rate=audio.frame_rate) + trimmed_audio
        trimmed_audio = trimmed_audio.set_channels(2)
        trimmed_audio.export(temp_file, format="wav")
        return has_2pop, os.path.basename(file_path)
    except Exception as e:
        return False, f"Error: {os.path.basename(file_path)} - {str(e)}"

def find_music_onset_from_array(audio_array, sr):
    onset_frames = librosa.onset.onset_detect(y=audio_array, sr=sr, units='time', pre_max=1, post_max=1, pre_avg=1, post_avg=1, delta=0.01, wait=1)
    return int(onset_frames[0] * 1000) if onset_frames.size > 0 else 0

def normalize_audio(folder_path, target_peak=-0.5):
    lowest_lufs = float('inf')
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.wav', '.aif', '.aiff')):
            file_path = os.path.join(folder_path, filename)
            audio = AudioSegment.from_file(file_path).normalize(target_peak)
            audio.export(file_path, format="wav")
            data, rate = sf.read(file_path)
            loudness = pyln.Meter(rate).integrated_loudness(data)
            lowest_lufs = min(lowest_lufs, loudness)
    
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.wav', '.aif', '.aiff')):
            file_path = os.path.join(folder_path, filename)
            data, rate = sf.read(file_path)
            loudness = pyln.Meter(rate).integrated_loudness(data)
            gain_db = lowest_lufs - loudness
            adjusted_audio = AudioSegment.from_wav(file_path) + gain_db
            adjusted_audio.export(file_path, format="wav")
    
    return lowest_lufs

def process_folder(folder_path, ort_session):
    start_time = time.time()
    temp_folder = os.path.join(folder_path, "Temp")
    exports_folder = os.path.join(folder_path, "Exports")

    shutil.rmtree(temp_folder, ignore_errors=True)
    os.makedirs(temp_folder, exist_ok=True)

    audio_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.wav', '.aif', '.aiff', '.mp3', '.mp4'))]
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [executor.submit(process_audio_file, os.path.join(folder_path, f), ort_session, temp_folder, 0.5) for f in audio_files]
        results = [future.result() for future in concurrent.futures.as_completed(futures)]
    
    final_lufs = normalize_audio(temp_folder, target_peak=-0.5)

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
        available_providers = ort.get_available_providers()
        print(f"Available ONNX Runtime providers: {available_providers}")
        
        if 'CoreMLExecutionProvider' in available_providers:
            providers = ['CoreMLExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']
        
        ort_session = ort.InferenceSession("2pop_model.onnx", providers=providers)
        print(f"ONNX model loaded successfully using {ort_session.get_providers()[0]}")
    except Exception as e:
        print(f"Error loading ONNX model: {str(e)}")
        exit(1)

    process_folder(args.folder_path, ort_session)