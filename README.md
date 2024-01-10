# Audio Processing Script

This script is designed to automate the process of 2pop removal, normalization, and export. It features a model trained with Liner.AI to recognize 2pop sounds, remove them, then normalize audio loudness across a folder.

## What is a 2 Pop?
A 2 Pop is a short beep that is added before a piece of audio as a reference for visual sync. It helps align music and video in production workflows.

## Why did you build this?
I work with a large library of music, and some of it has 2 Pops, some doesn't. When sending raw audio to clients, I prefer to remove the 2 Pop so its just music right from the front, and then normalize so that there aren't large changes in music volume as they are listening.
This script allows me to accomplish both very quickly and in an automated fashion.

## Features

- **2 Pop Detection**: Automatically detects the presence of a 2 pop sound (using a TensorFlow model) in the first 1.5 seconds of the audio file, and removes it.
- **Music Onset Detection**: Identifies the onset of music in an audio track using librosa, and adds data so there is .5 seconds of silence before the music starts.
- **Loudness Normalization**: Adjusts the loudness of audio files to a standard LUFS level (-14 LUFS default) using FFmpeg.
- **Support for Multiple Formats**: Processes `.wav`, `.aif`, `.aiff`, and `.mp3` file formats.
- **Parallel Processing**: Normalizes multiple audio files in parallel for faster processing.

## How to Use

1. **Clone the Repository**
   
   Clone this repository to your local machine using `git clone`.

2. **Install Dependencies**

   Ensure you have the following dependencies installed:
   - Python 3.x
   - TensorFlow
   - Librosa
   - PyDub
   - FFmpeg (ensure it's added to your system's PATH)

3. **Running the Script**

   Navigate to the script's directory and run:

   ```
   python audio_processing_script.py --folder_path="path/to/your/audio/files" --level=-14
   ```

   - `folder_path`: Path to the folder containing your audio files.
   - `level`: The loudness level for normalization (default is -14 LUFS).

## Requirements

- Python 3.x
- Libraries: TensorFlow, Librosa, PyDub
- FFmpeg: This script uses FFmpeg for audio processing. Ensure it is installed and added to your system's PATH.

## Contributing

Contributions to improve this script are welcome. Please fork this repository, make your changes, and submit a pull request.

## Warning
This script is a work in progress, and while it does create a copy of audio files to work on them, I cannot and will not make any guarantees as it pertains to your data safety. When working with this script please keep proper backups of all data as necessary.
