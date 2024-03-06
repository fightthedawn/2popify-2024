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
    output_dir = os.path.join(folder_path, "_Normalized")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    formats_to_convert = ['.wav', '.aif', '.aiff', '.mp3', '.m4a']
    loudness_values = []
    
    print("Step 1 & 2: Normalizing to -1 dB True Peak and measuring LUFS.\n\n")
    for file in os.listdir(folder_path):
        extension = os.path.splitext(file)[1].lower()
        if extension in formats_to_convert:
            file_path = os.path.join(folder_path, file)
            print(f"Processing {file_path}")
            audio = AudioSegment.from_file(file_path)
            
            peak_normalize_db = -1.0 - audio.max_dBFS
            normalized_audio = audio.apply_gain(peak_normalize_db)
            
            audio_data, sample_rate = audio_segment_to_numpy_array(normalized_audio)
            meter = Meter(sample_rate)
            loudness = meter.integrated_loudness(audio_data)
            loudness_values.append(loudness)
            print(f"Loudness of {file}: {loudness} LUFS\n")
            
            temp_path = os.path.join(output_dir, f"temp_{file}")
            normalized_audio.export(temp_path, format=extension.replace('.', ''))
    
    if not loudness_values:
        print("No audio files were processed for loudness measurement.")
        return
    
    lowest_lufs = min(loudness_values)
    print(f"\n\nLowest LUFS in the batch: {lowest_lufs}")
    
    print("\n\nStep 3: Normalizing files to the lowest LUFS value.")
    for temp_file in os.listdir(output_dir):
        if temp_file.startswith("temp_"):
            file_path = os.path.join(output_dir, temp_file)
            audio = AudioSegment.from_file(file_path)
            
            audio_data, sample_rate = audio_segment_to_numpy_array(audio)
            meter = Meter(sample_rate)
            current_loudness = meter.integrated_loudness(audio_data)
            gain_to_apply = lowest_lufs - current_loudness
            print(f"{temp_file} - Current LUFS: {current_loudness}, Gain to apply: {gain_to_apply} dB\n")
            
            final_normalized_audio = audio.apply_gain(gain_to_apply)
            
            final_path = os.path.join(output_dir, temp_file.replace("temp_", ""))
            export_format = temp_file.split('.')[-1].replace("temp_", "").lower()
            if export_format in ['aif', 'aiff']:
                export_format = 'aiff'
            final_normalized_audio.export(final_path, format=export_format)
            
            os.remove(file_path)
    
    print("Normalization completed. Files are saved in the output directory.")

if __name__ == "__main__":
    # Example usage; adjust as needed based on how you plan to call this script
    folder_path = 'TestData/TestData7'  # This should be modified based on actual use
    normalize_audio(folder_path)