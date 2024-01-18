import os
import librosa

def find_shortest_audio_file(folder_path):
    shortest_duration = float('inf')
    shortest_file = None

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.wav', '.mp3', '.aif', '.aiff', '.flac', '.m4a')):
            file_path = os.path.join(folder_path, filename)
            try:
                duration = librosa.get_duration(filename=file_path)
                if duration < shortest_duration:
                    shortest_duration = duration
                    shortest_file = filename
            except Exception as e:
                print(f"Error processing {filename}: {e}")

    if shortest_file is not None:
        print(f"The shortest audio file is '{shortest_file}' with a duration of {shortest_duration * 1000:.2f} milliseconds.")
    else:
        print("No audio files found in the folder.")

# Example usage
folder_path = 'ModelData/2024-01-17/2pop'  # Replace with your folder path
find_shortest_audio_file(folder_path)