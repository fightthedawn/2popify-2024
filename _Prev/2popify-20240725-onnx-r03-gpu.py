import os
import numpy as np
from scipy import signal
import noisereduce as nr
import soundfile as sf
import argparse
import time
import concurrent.futures
import shutil
import pyloudnorm as pyln
import onnxruntime as ort
from pydub import AudioSegment
import multiprocessing

def convert_to_wav(file_path, temp_folder):
    try:
        audio = AudioSegment.from_file(file_path)
        wav_path = os.path.join(temp_folder, os.path.splitext(os.path.basename(file_path))[0] + ".wav")
        audio.export(wav_path, format="wav")
        return wav_path
    except Exception as e:
        print(f"Error converting {file_path} to WAV: {str(e)}")
        return None

def preprocess_audio_for_model(samples, sr, target_sr=16000):
    if sr != target_sr:
        samples = signal.resample(samples, int(len(samples) * target_sr / sr))
    
    audio_reduced_noise = nr.reduce_noise(y=samples, sr=target_sr)
    sos = signal.butter(5, [140 / (0.5 * target_sr), 2200 / (0.5 * target_sr)], btype='band', output='sos')
    audio_filtered = signal.sosfilt(sos, audio_reduced_noise)
    return audio_filtered[:int(1.5 * target_sr)], target_sr

def detect_2_pop_with_model(audio, ort_session, detection_threshold=0.95):
    audio = np.pad(audio, (0, max(0, 24000 - len(audio))))[:24000]
    inputs = {ort_session.get_inputs()[0].name: np.array([audio], dtype=np.float32)}
    for input_detail in ort_session.get_inputs()[1:]:
        input_shape = input_detail.shape
        dtype = np.float32 if input_detail.type == 'tensor(float)' else np.int32
        inputs[input_detail.name] = np.zeros(input_shape, dtype=dtype)
    outputs = ort_session.run(None, inputs)
    return outputs[0][0][1] > detection_threshold

def find_music_onset(audio, sr):
    frame_length = int(0.05 * sr)  # 50ms frames
    energy = np.array([np.sum(audio[i:i+frame_length]**2) for i in range(0, len(audio)-frame_length, frame_length)])
    threshold = np.mean(energy) + np.std(energy)
    onset_frames = np.where(energy > threshold)[0]
    return int((onset_frames[0] * frame_length) / sr * 1000) if len(onset_frames) > 0 else 0

def process_audio_file(args):
    file_path, temp_folder, detection_threshold = args
    try:
        print(f"Processing file: {file_path}")
        audio = AudioSegment.from_file(file_path)
        
        if len(audio) < 1500:  # Skip files shorter than 1.5 seconds
            print(f"Skipping {file_path}: File too short")
            return f"Skipped: {os.path.basename(file_path)} (too short)"

        temp_file = os.path.join(temp_folder, os.path.splitext(os.path.basename(file_path))[0] + ".wav")
        
        # Convert to mono for preprocessing
        audio_mono = audio.set_channels(1)
        samples = np.array(audio_mono.get_array_of_samples())
        
        waveform, model_sr = preprocess_audio_for_model(samples, audio.frame_rate)
        
        # Save preprocessed audio for 2pop detection
        np.save(temp_file + '.npy', waveform)

        # Ensure stereo output
        if audio.channels == 1:
            audio = audio.set_channels(2)

        audio.export(temp_file, format="wav")
        print(f"Saved processed file to: {temp_file}")
        return os.path.basename(temp_file)
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        import traceback
        traceback.print_exc()
        return f"Error: {os.path.basename(file_path)} - {str(e)}"

def normalize_audio_file(file_path, target_lufs):
    try:
        audio = AudioSegment.from_file(file_path)
        samples = np.array(audio.get_array_of_samples()).astype(np.float32)
        samples = samples / np.iinfo(audio.sample_width * 8).max  # Normalize to [-1.0, 1.0]
        
        if audio.channels == 2:
            samples = samples.reshape((-1, 2))
        
        meter = pyln.Meter(audio.frame_rate)
        loudness = meter.integrated_loudness(samples)
        gain_db = target_lufs - loudness
        
        # Apply gain
        normalized_samples = pyln.normalize.loudness(samples, loudness, target_lufs)
        
        # Convert back to int16
        normalized_samples = (normalized_samples * np.iinfo(np.int16).max).astype(np.int16)
        
        # Create a new AudioSegment
        normalized_audio = AudioSegment(
            normalized_samples.tobytes(),
            frame_rate=audio.frame_rate,
            sample_width=2,
            channels=audio.channels
        )
        
        normalized_audio.export(file_path, format="wav")
        return loudness
    except Exception as e:
        print(f"Error normalizing {file_path}: {str(e)}")
        return None

def normalize_audio(folder_path, target_lufs):
    files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.wav', '.aif', '.aiff'))]
    if not files:
        print("No audio files found in the temporary folder. Skipping normalization.")
        return None

    with concurrent.futures.ThreadPoolExecutor() as executor:
        loudness_values = list(executor.map(lambda f: normalize_audio_file(os.path.join(folder_path, f), target_lufs), files))

    valid_loudness = [l for l in loudness_values if l is not None]
    if not valid_loudness:
        print("No valid loudness values found. Skipping normalization.")
        return None

    return target_lufs

def process_folder(folder_path, ort_session):
    start_time = time.time()
    temp_folder = os.path.join(folder_path, "Temp")
    exports_folder = os.path.join(folder_path, "Exports")

    shutil.rmtree(temp_folder, ignore_errors=True)
    os.makedirs(temp_folder, exist_ok=True)

    audio_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.wav', '.aif', '.aiff', '.mp3', '.mp4'))]
    
    if not audio_files:
        print("No audio files found in the specified folder.")
        return

    print(f"Found {len(audio_files)} audio files to process.")

    num_processes = max(1, multiprocessing.cpu_count() - 1)

    with multiprocessing.Pool(processes=num_processes) as pool:
        file_args = [(os.path.join(folder_path, f), temp_folder, 0.5) for f in audio_files]
        results = pool.map(process_audio_file, file_args)

    processed_files = [r for r in results if not r.startswith("Error") and not r.startswith("Skipped")]
    print(f"Files in temp folder after processing: {processed_files}")

    if not processed_files:
        print("No files were successfully processed and saved to the temp folder.")
        return

    # Detect 2pops in the main process and apply trimming
    detection_results = []
    for filename in processed_files:
        try:
            npy_file = os.path.join(temp_folder, filename + '.npy')
            wav_file = os.path.join(temp_folder, filename)
            waveform = np.load(npy_file)
            has_2pop = detect_2_pop_with_model(waveform, ort_session, 0.5)
            detection_results.append((has_2pop, filename))
            
            # Apply trimming based on 2-pop detection
            audio = AudioSegment.from_file(wav_file)
            if has_2pop:
                print(f"2 pop detected in {filename}")
                audio = audio[1500:]
                start_of_music = find_music_onset(np.array(audio.get_array_of_samples()), audio.frame_rate)
                actual_start = max(start_of_music + 1000, 0)
                audio = audio[actual_start:]
            else:
                print(f"No 2 pop detected in {filename}")
                start_of_music = find_music_onset(np.array(audio.get_array_of_samples()), audio.frame_rate)
                silence_duration = max(500 - start_of_music, 0)
                audio = AudioSegment.silent(duration=silence_duration, frame_rate=audio.frame_rate) + audio[max(0, start_of_music - 500):]
            
            audio.export(wav_file, format="wav")
            os.remove(npy_file)  # Clean up the temporary .npy file
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
            detection_results.append((False, f"Error: {filename}"))

    try:
        final_lufs = normalize_audio(temp_folder, target_lufs=-14)  # or whatever target LUFS you prefer
    except Exception as e:
        print(f"Error during normalization: {str(e)}")
        final_lufs = None

    if final_lufs is not None:
        shutil.rmtree(exports_folder, ignore_errors=True)
        shutil.move(temp_folder, exports_folder)
        
        print("\nProcessing Results:")
        for has_2pop, filename in detection_results:
            status = "2 pop detected" if has_2pop else "No 2 pop detected"
            print(f"{filename}: {status}")
        
        print(f"\nAll files normalized to {final_lufs:.2f} LUFS")
    else:
        print("\nNormalization skipped due to errors. Check the temporary folder for processed files.")

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