import os
import numpy as np
from pydub import AudioSegment
from pyloudnorm import Meter
from pathlib import Path

def audio_segment_to_numpy_array(audio_segment):
    samples = np.array(audio_segment.get_array_of_samples())

    if audio_segment.channels == 2:
        samples = samples.reshape((-1, 2))
    
    # Ensure floating point representation matches the expected bit depth
    if audio_segment.sample_width == 2:  # 16-bit audio
        samples = samples.astype(np.float32) / 2**15
    elif audio_segment.sample_width == 3:  # 24-bit audio
        samples = samples.astype(np.float32) / 2**23
    elif audio_segment.sample_width == 4:  # 32-bit audio
        samples = samples.astype(np.float32) / 2**31
    else:
        raise ValueError("Unsupported sample width")

    return samples, audio_segment.frame_rate

def normalize_audio(folder_path):
    formats_to_convert = ['.wav', '.aif', '.aiff', '.mp3', '.m4a']
    loudness_values = []
    file_loudness_map = {}  # Create a map to store filename and its loudness

    file_paths = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if os.path.splitext(file)[1].lower() in formats_to_convert]

    for file_path in file_paths:
        audio = AudioSegment.from_file(file_path)
        peak_normalize_db = -1.0 - audio.max_dBFS
        normalized_audio = audio.apply_gain(peak_normalize_db)
        
        audio_data, sample_rate = audio_segment_to_numpy_array(normalized_audio)
        meter = Meter(sample_rate)
        loudness = meter.integrated_loudness(audio_data)
        
        loudness_values.append(loudness)
        file_loudness_map[file_path] = loudness

    if not loudness_values:
        return

    lowest_lufs = min(loudness_values)
    # Find the filename with the lowest LUFS
    lowest_lufs_file = [file for file, lufs in file_loudness_map.items() if lufs == lowest_lufs][0]
    
    print(f"\nLowest LUFS in the batch: {lowest_lufs} (File: {os.path.basename(lowest_lufs_file)})")

    # Apply normalization based on the lowest LUFS
    #print("\n\nStep 3: Normalizing files to the lowest LUFS value.")
    for file_path in file_paths:
        audio = AudioSegment.from_file(file_path)
        
        audio_data, sample_rate = audio_segment_to_numpy_array(audio)
        meter = Meter(sample_rate)
        current_loudness = meter.integrated_loudness(audio_data)
        gain_to_apply = lowest_lufs - current_loudness
        #print(f"{os.path.basename(file_path)} - Current LUFS: {current_loudness}, Gain to apply: {gain_to_apply} dB")
        
        final_normalized_audio = audio.apply_gain(gain_to_apply)
        
        # Correctly determine the format for export based on file extension
        file_extension = os.path.splitext(file_path)[1].lower()
        export_format = {
            '.wav': 'wav',
            '.aif': 'aiff',  # pydub uses 'aiff' for both .aif and .aiff
            '.aiff': 'aiff',
            '.mp3': 'mp3',
            '.m4a': 'ipod',  # For M4A, 'ipod' works with pydub if ffmpeg is correctly set up
        }.get(file_extension, 'wav')  # Default to 'wav' if not found

        # Overwrite the original file with its normalized version
        final_normalized_audio.export(file_path, format=export_format)

    print("\nNormalization completed. Files have been replaced in the original directory.\n")

if __name__ == "__main__":
    #folder_path = 'TestData/TestData7'  # Adjust as necessary
    normalize_audio(folder_path)