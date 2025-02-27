import argparse
import os
import numpy as np
from pathlib import Path
import soundfile as sf
import librosa
import pyloudnorm as pyln
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm
import concurrent.futures
from contextlib import contextmanager
import time
import sys
import warnings
import tempfile
import joblib
from functools import partial

# Suppress non-critical warnings
warnings.filterwarnings("ignore", category=UserWarning)

# For TensorFlow Lite model loading (if installed)
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# Class for handling progress in multithreaded environment
class MultiThreadedProgress:
    def __init__(self, total, desc="Processing"):
        self.total = total
        self.desc = desc
        self.completed = 0
        self.lock = __import__('threading').Lock()
        self.start_time = time.time()
        
    def update(self, n=1):
        with self.lock:
            self.completed += n
            percentage = (self.completed / self.total) * 100
            elapsed = time.time() - self.start_time
            rate = self.completed / elapsed if elapsed > 0 else 0
            remaining = (self.total - self.completed) / rate if rate > 0 else 0
            
            bar_length = 30
            filled_length = int(bar_length * self.completed // self.total)
            bar = '█' * filled_length + '-' * (bar_length - filled_length)
            
            sys.stdout.write(f'\r{self.desc}: |{bar}| {percentage:.1f}% Complete ({self.completed}/{self.total}) [ETA: {remaining:.1f}s]')
            sys.stdout.flush()
            
    def close(self):
        sys.stdout.write('\n')
        sys.stdout.flush()

class AudioProcessor:
    """Class for detecting and removing 2-pops and normalizing audio files."""
    
    def __init__(self, target_lufs=-23.0, silence_duration=0.5, fade_samples=100, 
                 model_path=None, num_threads=None, advanced_onset=True):
        """
        Initialize the audio processor.
        
        Args:
            target_lufs (float): Target LUFS value for normalization
            silence_duration (float): Duration of silence to add before music in seconds
            fade_samples (int): Number of samples for fade-in
            model_path (str, optional): Path to ML model for 2-pop detection
            num_threads (int, optional): Number of threads for parallel processing
            advanced_onset (bool): Whether to use advanced onset detection algorithm
        """
        self.target_lufs = target_lufs
        self.silence_duration = silence_duration
        self.fade_samples = fade_samples
        self.advanced_onset = advanced_onset
        self.meter = pyln.Meter(44100)  # Initialize with default sample rate
        self.model = None
        self.input_details = None
        self.output_details = None
        self.classes = ["Music", "2pop"]  # Default classes
        self.num_threads = num_threads or max(1, os.cpu_count() - 1)
        self.sample_rate = 44100  # Default sample rate for reporting
        self.current_file = "unknown"  # Track current file for debugging
        
        # Try to load ML model if path is provided
        if model_path and os.path.exists(model_path) and TF_AVAILABLE:
            try:
                print(f"Loading model from {model_path}...")
                self.model = tf.lite.Interpreter(model_path=model_path)
                self.model.allocate_tensors()
                
                # Try to load classes from classes.json if available
                try:
                    with open('classes.json', 'r') as f:
                        import json
                        classes_data = json.load(f)
                        # Convert to list if it's a dict
                        if isinstance(classes_data, dict):
                            # Sort by value to get correct order
                            self.classes = [k for k, v in sorted(classes_data.items(), key=lambda item: item[1])]
                        else:
                            self.classes = classes_data
                except:
                    # Default classes based on provided example
                    self.classes = ["Music", "2pop"]
                    
                print(f"Loaded classes: {self.classes}")
            except Exception as e:
                print(f"Warning: Failed to load model: {str(e)}")
                self.model = None
    
    def _detect_pops(self, audio, sr, threshold_factor=3.5):
        """
        Detect 2-pops in the beginning of an audio file.
        Focuses on finding sine-wave-like beeps followed by silence.
        Uses the same algorithm for all files regardless of filename.
        
        Args:
            audio (np.ndarray): Audio data as numpy array
            sr (int): Sample rate of the audio
            threshold_factor (float): Factor to multiply RMS for pop detection threshold
            
        Returns:
            list: Indices of detected pops, empty list if not found
        """
        # Store the sample rate for reporting
        self.sample_rate = sr
        
        # Extract filename from input audio for debugging (does not affect processing)
        input_file = getattr(self, 'current_file', 'unknown')
        filename = os.path.basename(input_file) if input_file != 'unknown' else 'unknown'
        
        # Note files with '2pop' in name purely for debugging output
        if '2pop' in filename.lower():
            print(f"File has '2pop' in filename: {filename} (for debugging only)")
        
        # Convert to mono if stereo
        if len(audio.shape) > 1:
            audio_mono = np.mean(audio, axis=1)
        else:
            audio_mono = audio
            
        # Focus only on the first 4 seconds for pops
        analysis_window = min(len(audio_mono), int(4 * sr))
        analysis_audio = audio_mono[:analysis_window]
        
        # Calculate the energy envelope (for transient detection)
        window_size = int(0.005 * sr)  # 5ms window
        energy = np.array([
            np.sum(np.abs(analysis_audio[i:i+window_size])**2) 
            for i in range(0, len(analysis_audio) - window_size, window_size)
        ])
        
        # Smooth energy curve
        energy_smooth = gaussian_filter1d(energy, sigma=1)
        
        # Calculate threshold based on signal statistics
        rms = np.sqrt(np.mean(energy_smooth**2))
        background_level = np.percentile(energy_smooth, 25)  # Use 25th percentile as background
        threshold = background_level + (rms - background_level) * threshold_factor
        
        # Find peaks above threshold
        peaks = []
        peak_amplitudes = []
        for i in range(1, len(energy_smooth) - 1):
            if (energy_smooth[i] > energy_smooth[i-1] and 
                energy_smooth[i] > energy_smooth[i+1] and 
                energy_smooth[i] > threshold):
                peak_sample = i * window_size
                peaks.append(peak_sample)
                peak_amplitudes.append(energy_smooth[i])
        
        # Print peaks found for debugging
        if len(peaks) > 0:
            print(f"File: {filename} - Found {len(peaks)} potential peaks")
            for i, peak in enumerate(peaks[:5]):  # Show first 5 peaks
                print(f"  Peak {i+1}: {peak/sr:.3f}s")
        else:
            print(f"File: {filename} - No significant peaks detected")
            return []
            
        # Need at least 2 peaks to have a 2-pop pair
        if len(peaks) < 2:
            print(f"File: {filename} - Not enough peaks for 2-pop analysis")
            return []
            
        # Look for pairs with appropriate spacing (0.7-1.3 seconds)
        potential_pairs = []
        for i in range(len(peaks) - 1):
            for j in range(i + 1, len(peaks)):
                distance = peaks[j] - peaks[i]
                if 0.7 * sr <= distance <= 1.3 * sr:
                    # Score is based on peak amplitudes and proper spacing
                    peak_score = peak_amplitudes[i] + peak_amplitudes[j]
                    potential_pairs.append({
                        'indices': [peaks[i], peaks[j]],
                        'score': peak_score,
                        'distance': distance
                    })
                    
        if not potential_pairs:
            print(f"File: {filename} - No peaks with appropriate spacing detected")
            return []
            
        # Sort by score (highest first)
        potential_pairs.sort(key=lambda x: x['score'], reverse=True)
            
        # Check for silence after the second pop (crucial for identifying true 2-pops)
        valid_pairs = []
        for pair in potential_pairs:
            second_pop = pair['indices'][1]
            
            # Define the region to check for silence (2+ seconds after second pop)
            silence_start = second_pop + int(0.2 * sr)  # Start checking 0.2s after pop
            silence_end = min(second_pop + int(2.5 * sr), len(analysis_audio))  # Check up to 2.5s after
            
            # If we don't have enough audio to check, skip this pair
            if silence_end - silence_start < sr:
                continue
                
            # Calculate average energy in the region
            silence_region = analysis_audio[silence_start:silence_end]
            silence_energy = np.mean(np.abs(silence_region)**2)
            
            # Calculate threshold for what we consider "silence"
            # (Using background level from earlier as reference)
            silence_threshold = background_level * 3
            
            # Check for low energy in the first second after the pop
            # and higher energy after that (music starts)
            first_second = min(silence_start + sr, silence_end)
            silence_energy_first = np.mean(np.abs(analysis_audio[silence_start:first_second])**2)
            
            if silence_energy_first < silence_threshold:
                # This pair has silence after the second pop - good sign of a 2-pop
                # Now try to detect music onset
                music_start = 0
                for k in range(silence_start, silence_end - window_size, window_size):
                    segment_energy = np.mean(np.abs(analysis_audio[k:k+window_size])**2)
                    if segment_energy > silence_threshold * 2:  # Music is typically much louder
                        music_start = k
                        break
                
                # If music starts at least 1.5 seconds after the second pop, this is likely a valid 2-pop
                if music_start == 0 or (music_start - second_pop) >= int(1.5 * sr):
                    valid_pairs.append({
                        'indices': pair['indices'],
                        'score': pair['score'],
                        'music_onset': music_start if music_start > 0 else second_pop + int(2 * sr)
                    })
        
        if valid_pairs:
            # Sort valid pairs by score and take the best one
            valid_pairs.sort(key=lambda x: x['score'], reverse=True)
            best_pair = valid_pairs[0]
            
            pop_indices = best_pair['indices']
            print(f"File: {filename}")
            print(f"  Found 2-pops at {pop_indices[0]/sr:.2f}s and {pop_indices[1]/sr:.2f}s")
            print(f"  Distance between pops: {(pop_indices[1]-pop_indices[0])/sr:.2f}s")
            print(f"  Expected music onset around: {best_pair['music_onset']/sr:.2f}s")
            
            return pop_indices
        else:
            # If we have potential pairs but none passed the silence check
            print(f"File: {filename} - Found peak pairs but none were followed by appropriate silence")
            return []
    
    def _detect_music_onset(self, audio, sr, start_index=0, sensitivity=2.0):
        """
        Detect the onset of music in an audio file.
        
        Args:
            audio (np.ndarray): Audio data as numpy array
            sr (int): Sample rate of the audio
            start_index (int): Index to start looking from
            sensitivity (float): Sensitivity factor for onset detection
            
        Returns:
            int: Index of detected music onset
        """
        if self.advanced_onset:
            return self._detect_music_onset_advanced(audio, sr, start_index, sensitivity)
        else:
            return self._detect_music_onset_basic(audio, sr, start_index, sensitivity)
    
    def _detect_music_onset_basic(self, audio, sr, start_index=0, sensitivity=2.0):
        """
        Basic method to detect the onset of music in an audio file.
        
        Args:
            audio (np.ndarray): Audio data as numpy array
            sr (int): Sample rate of the audio
            start_index (int): Index to start looking from
            sensitivity (float): Sensitivity factor for onset detection
            
        Returns:
            int: Index of detected music onset
        """
        # Convert to mono if stereo
        if len(audio.shape) > 1:
            audio_mono = np.mean(audio, axis=1)
        else:
            audio_mono = audio
            
        # Only analyze audio after the specified start index
        analysis_audio = audio_mono[start_index:]
        
        # Compute onset envelope
        onset_env = librosa.onset.onset_strength(
            y=analysis_audio, 
            sr=sr,
            hop_length=512,
            aggregate=np.median
        )
        
        # Compute the background level
        background = np.mean(onset_env[:10])  # Use the beginning as reference
        
        # Set threshold based on background and sensitivity
        threshold = background * sensitivity
        
        # Find the first onset that exceeds the threshold
        onset_frames = np.where(onset_env > threshold)[0]
        
        if len(onset_frames) > 0:
            # Convert frame index to sample index
            onset_sample = onset_frames[0] * 512
            return start_index + onset_sample
        else:
            # If no clear onset is found, return a default (2 seconds after start_index)
            return start_index + int(2 * sr)
            
    def _detect_music_onset_advanced(self, audio, sr, start_index=0, sensitivity=2.0):
        """
        Advanced method to detect the onset of music in an audio file.
        Uses multiple features and adaptive thresholding for more reliable detection.
        
        Args:
            audio (np.ndarray): Audio data as numpy array
            sr (int): Sample rate of the audio
            start_index (int): Index to start looking from
            sensitivity (float): Sensitivity factor for onset detection
            
        Returns:
            int: Index of detected music onset
        """
        # Convert to mono if stereo
        if len(audio.shape) > 1:
            audio_mono = np.mean(audio, axis=1)
        else:
            audio_mono = audio
            
        # Only analyze audio after the specified start index
        analysis_audio = audio_mono[start_index:]
        
        # Parameters
        hop_length = 512
        
        # Compute multiple onset features
        
        # 1. Standard onset strength (energy flux)
        onset_env_energy = librosa.onset.onset_strength(
            y=analysis_audio, 
            sr=sr,
            hop_length=hop_length,
        )
        
        # 2. Another onset feature - spectral difference
        onset_env_spectral = librosa.onset.onset_strength(
            y=analysis_audio, 
            sr=sr,
            hop_length=hop_length,
        )
        
        # 3. Spectral contrast
        try:
            contrast = librosa.feature.spectral_contrast(y=analysis_audio, sr=sr, hop_length=hop_length)
            onset_env_contrast = np.mean(contrast, axis=0)
        except:
            # Fallback if spectral contrast fails
            onset_env_contrast = np.zeros_like(onset_env_energy)
        
        # 4. RMS energy
        rms = librosa.feature.rms(y=analysis_audio, hop_length=hop_length)[0]
        
        # 5. Zero crossing rate change
        zcr = librosa.feature.zero_crossing_rate(y=analysis_audio, hop_length=hop_length)[0]
        zcr_diff = np.diff(zcr, prepend=zcr[0])
        zcr_diff = np.abs(zcr_diff)
        
        # Normalize all features to [0, 1] range
        def normalize(x):
            min_val = np.min(x)
            max_val = np.max(x)
            if max_val > min_val:
                return (x - min_val) / (max_val - min_val)
            else:
                return np.zeros_like(x)
                
        onset_env_energy_norm = normalize(onset_env_energy)
        onset_env_spectral_norm = normalize(onset_env_spectral)
        onset_env_contrast_norm = normalize(onset_env_contrast)
        rms_norm = normalize(rms)
        zcr_diff_norm = normalize(zcr_diff)
        
        # Combine features with weighting
        combined_onset = (
            0.3 * onset_env_energy_norm + 
            0.3 * onset_env_spectral_norm +
            0.1 * onset_env_contrast_norm +
            0.2 * rms_norm +
            0.1 * zcr_diff_norm
        )
        
        # Apply median filter to smooth the curve
        combined_onset_smooth = librosa.util.peak_pick(
            combined_onset, 
            pre_max=3,
            post_max=3,
            pre_avg=10,
            post_avg=10,
            delta=sensitivity * 0.05,
            wait=10
        )
        
        # Compute adaptive threshold
        window_size = 5
        adaptive_threshold = np.zeros_like(combined_onset)
        
        for i in range(len(combined_onset)):
            start_idx = max(0, i - window_size)
            end_idx = min(len(combined_onset), i + window_size + 1)
            adaptive_threshold[i] = np.mean(combined_onset[start_idx:end_idx]) * sensitivity
        
        # Find first significant onset
        # Looking for both local peaks and sustained energy
        onset_candidates = []
        
        for i in range(1, len(combined_onset)-1):
            # Check if it's a peak
            if combined_onset[i] > combined_onset[i-1] and combined_onset[i] > combined_onset[i+1]:
                # Check if it exceeds the adaptive threshold
                if combined_onset[i] > adaptive_threshold[i]:
                    # Check if there's sustained energy after this point
                    future_window = min(20, len(combined_onset) - i)
                    if np.mean(combined_onset[i:i+future_window]) > np.mean(combined_onset[max(0,i-future_window):i]) * 1.2:
                        onset_candidates.append(i)
        
        # If peaks were found with the complex method
        if onset_candidates:
            # Convert frame index to sample index
            onset_sample = onset_candidates[0] * hop_length
            return start_index + onset_sample
            
        # Fallback to peak picking if no candidates found
        peaks = librosa.util.peak_pick(
            combined_onset, 
            pre_max=3,
            post_max=3,
            pre_avg=5,
            post_avg=5,
            delta=sensitivity * 0.03,
            wait=10
        )
        
        if len(peaks) > 0:
            onset_sample = peaks[0] * hop_length
            return start_index + onset_sample
        
        # If still no peaks, use a simpler threshold method
        above_threshold = np.where(combined_onset > np.mean(combined_onset) * sensitivity)[0]
        
        if len(above_threshold) > 0:
            onset_sample = above_threshold[0] * hop_length
            return start_index + onset_sample
        
        # Last resort: if no clear onset is found, return a default
        return start_index + int(2 * sr)
    
    def _measure_lufs(self, audio, sr):
        """
        Measure the integrated LUFS of an audio file.
        
        Args:
            audio (np.ndarray): Audio data as numpy array
            sr (int): Sample rate of the audio
            
        Returns:
            float: Measured LUFS value
        """
        # Update meter if sample rate is different
        if sr != self.meter.rate:
            self.meter = pyln.Meter(sr)
            
        # Convert to float32 if not already
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
            
        # Measure LUFS
        return self.meter.integrated_loudness(audio)
    
    def _normalize_lufs(self, audio, current_lufs, target_lufs=None):
        """
        Normalize audio to target LUFS.
        
        Args:
            audio (np.ndarray): Audio data as numpy array
            current_lufs (float): Current LUFS measurement
            target_lufs (float, optional): Target LUFS value, uses instance default if None
            
        Returns:
            np.ndarray: Normalized audio
        """
        if target_lufs is None:
            target_lufs = self.target_lufs
            
        # Calculate the gain needed
        gain_db = target_lufs - current_lufs
        gain_linear = 10 ** (gain_db / 20.0)
        
        # Apply gain
        return audio * gain_linear
    
    def _create_fade(self, audio, fade_samples=None):
        """
        Create a short fade-in at the beginning of audio.
        
        Args:
            audio (np.ndarray): Audio data as numpy array
            fade_samples (int, optional): Number of samples for fade, uses instance default if None
            
        Returns:
            np.ndarray: Audio with fade-in applied
        """
        if fade_samples is None:
            fade_samples = self.fade_samples
            
        # Create fade curve (linear fade)
        fade_curve = np.linspace(0, 1, fade_samples)
        
        # Apply fade to audio
        result = audio.copy()
        
        # Handle both mono and stereo
        if len(result.shape) > 1:
            # Stereo
            for channel in range(result.shape[1]):
                result[:fade_samples, channel] *= fade_curve
        else:
            # Mono
            result[:fade_samples] *= fade_curve
            
        return result
    
    def process_file(self, input_file, output_file=None, normalize=True):
        """
        Process a single audio file to remove 2-pops and normalize.
        
        Args:
            input_file (str): Path to input audio file
            output_file (str, optional): Path for output audio file, defaults to overwriting input
            normalize (bool): Whether to normalize the audio to target LUFS
            
        Returns:
            dict: Processing results including LUFS measurements
        """
        if output_file is None:
            output_file = input_file
            
        # Store current file path for debugging
        self.current_file = input_file
            
        # Load audio file
        try:
            audio, sr = sf.read(input_file)
        except Exception as e:
            return {"error": f"Failed to read file {input_file}: {str(e)}"}
            
        # Store sample rate for later use
        self.sample_rate = sr
            
        # Store original LUFS
        original_lufs = self._measure_lufs(audio, sr)
        
        # Extract filename for printing
        filename = os.path.basename(input_file)
        
        # Detect 2-pops
        pop_indices = self._detect_pops(audio, sr)
        
        # If 2-pops are detected, process the file
        if pop_indices and len(pop_indices) == 2:
            # Use the second pop as reference point
            second_pop_index = pop_indices[1]
            
            # Find music onset after the second pop
            music_onset_index = self._detect_music_onset(audio, sr, start_index=second_pop_index)
            
            # Print detection results
            print(f"File: {filename}")
            print(f"  2-pops detected at: {pop_indices[0]/sr:.2f}s and {pop_indices[1]/sr:.2f}s")
            print(f"  Music onset detected at: {music_onset_index/sr:.2f}s")
            
            # Verify that we have an appropriate silence gap between pops and music
            # (At least 1.5 seconds after the second pop)
            min_silence_samples = int(1.5 * sr)
            if music_onset_index - second_pop_index < min_silence_samples:
                print(f"  Warning: Very short gap between 2nd pop and music: {(music_onset_index-second_pop_index)/sr:.2f}s")
                print(f"  This may not be a true 2-pop sequence")
                # Still process the file as if it had 2-pops, since we're being permissive
            
            # Calculate silence duration in samples
            silence_samples = int(self.silence_duration * sr)
            
            # Create new audio without pops, with silence, and with fade
            edited_audio = np.zeros_like(audio)
            
            # Add silence before music onset
            music_start_with_silence = music_onset_index - silence_samples
            
            # If music_start_with_silence is negative, we need to adjust
            if music_start_with_silence < 0:
                # Just use a minimal offset from the second pop
                music_start_with_silence = second_pop_index + int(0.1 * sr)
                
            # Copy all audio from music start (with silence) to end
            edited_audio[:len(audio) - music_start_with_silence] = audio[music_start_with_silence:]
            
            # Apply fade at the beginning of the music
            edited_audio = self._create_fade(edited_audio, self.fade_samples)
            
        else:
            # No 2-pops detected, keep original audio
            edited_audio = audio
            
            # Print for debugging
            print(f"File: {filename} - No 2-pops detected")
            
        # Normalize if requested
        if normalize:
            edited_lufs = self._measure_lufs(edited_audio, sr)
            normalized_audio = self._normalize_lufs(edited_audio, edited_lufs)
            final_audio = normalized_audio
            final_lufs = self.target_lufs
        else:
            final_audio = edited_audio
            final_lufs = self._measure_lufs(edited_audio, sr)
            
        # Write output file
        sf.write(output_file, final_audio, sr)
        
        return {
            "file": input_file,
            "pops_detected": len(pop_indices) == 2,
            "pop_positions": pop_indices if len(pop_indices) == 2 else None,
            "original_lufs": original_lufs,
            "final_lufs": final_lufs
        }
    
    def _process_file_phase1(self, input_file, output_file, progress=None):
        """
        Process a single file in phase 1 (pop detection and removal).
        Helper function for multithreaded processing.
        
        Args:
            input_file (str): Path to input file
            output_file (str): Path to output file
            progress (MultiThreadedProgress, optional): Progress tracker
            
        Returns:
            dict: Processing result for this file
        """
        # Store the input file path in the class instance for debugging
        self.current_file = str(input_file)
        
        # Process file
        result = self.process_file(str(input_file), str(output_file), normalize=False)
        
        # Update progress if available
        if progress:
            progress.update()
            
        return result
    
    def _process_file_phase2(self, input_file, output_file, normalization_target, progress=None):
        """
        Process a single file in phase 2 (normalization).
        Helper function for multithreaded processing.
        
        Args:
            input_file (str): Path to input file
            output_file (str): Path to output file
            normalization_target (float): Target LUFS
            progress (MultiThreadedProgress, optional): Progress tracker
            
        Returns:
            dict: File and LUFS information
        """
        try:
            # Read the processed file
            audio, sr = sf.read(str(output_file))
            
            # Measure current LUFS
            current_lufs = self._measure_lufs(audio, sr)
            
            # Normalize
            normalized_audio = self._normalize_lufs(audio, current_lufs, normalization_target)
            
            # Write normalized audio
            sf.write(str(output_file), normalized_audio, sr)
            
            if progress:
                progress.update()
                
            return {
                'file': str(input_file),
                'final_lufs': normalization_target
            }
        
        except Exception as e:
            if progress:
                progress.update()
                
            return {
                'file': str(input_file),
                'error': str(e)
            }
    
    def process_folder(self, input_folder, output_folder=None, target_lufs=None, match_quietest=False):
        """
        Process a folder of audio files with multithreading support.
        
        Args:
            input_folder (str): Path to input folder
            output_folder (str, optional): Path for output folder, defaults to input folder
            target_lufs (float, optional): Target LUFS value, uses instance default if None
            match_quietest (bool): Whether to match all files to the quietest file
            
        Returns:
            list: Processing results for each file
        """
        if output_folder is None:
            output_folder = input_folder
        else:
            os.makedirs(output_folder, exist_ok=True)
            
        # Get all audio files in the folder
        extensions = ['.wav', '.aiff', '.mp3', '.m4a']
        input_files = []
        
        for ext in extensions:
            input_files.extend(Path(input_folder).glob(f'*{ext}'))
            input_files.extend(Path(input_folder).glob(f'*{ext.upper()}'))
            
        if not input_files:
            return {"error": f"No supported audio files found in {input_folder}"}
        
        # Define output files
        output_files = [Path(output_folder) / input_file.name for input_file in input_files]
        
        # Phase 1: Process files without normalization using thread pool
        results = []
        file_lufs = {}
        
        print(f"Phase 1: Processing {len(input_files)} files...")
        
        # Process files sequentially for better debugging
        if os.environ.get('DEBUG') == '1':
            print("DEBUG mode: Processing files sequentially")
            progress = MultiThreadedProgress(len(input_files), desc="Phase 1: Processing")
            
            for input_file, output_file in zip(input_files, output_files):
                # Process each file individually
                result = self._process_file_phase1(str(input_file), str(output_file), progress)
                results.append(result)
                
                if 'error' not in result and 'final_lufs' in result:
                    file_lufs[str(input_file)] = result['final_lufs']
                    
            progress.close()
        else:
            # Initialize progress tracker
            progress = MultiThreadedProgress(len(input_files), desc="Phase 1: Processing")
            
            # Process files in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_threads) as executor:
                # Submit all tasks
                future_to_file = {
                    executor.submit(
                        self._process_file_phase1, 
                        str(input_file), 
                        str(output_file),
                        progress
                    ): str(input_file)
                    for input_file, output_file in zip(input_files, output_files)
                }
                
                # Collect results as they complete
                for future in concurrent.futures.as_completed(future_to_file):
                    input_file = future_to_file[future]
                    try:
                        result = future.result()
                        results.append(result)
                        
                        if 'error' not in result and 'final_lufs' in result:
                            file_lufs[input_file] = result['final_lufs']
                            
                    except Exception as e:
                        results.append({
                            'file': input_file,
                            'error': str(e)
                        })
            
            progress.close()
        
        # If matching to quietest file, find the quietest LUFS
        if match_quietest and file_lufs:
            quietest_lufs = min(file_lufs.values())
            print(f"Quietest file LUFS: {quietest_lufs:.1f} LUFS")
            normalization_target = quietest_lufs
        else:
            normalization_target = target_lufs if target_lufs is not None else self.target_lufs
            print(f"Normalizing to: {normalization_target:.1f} LUFS")
        
        # Phase 2: Normalize all files to the target
        print("Phase 2: Normalizing files...")
        
        # Initialize progress tracker for phase 2
        progress = MultiThreadedProgress(len(input_files), desc="Phase 2: Normalizing")
        
        # Process files in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            # Submit all tasks
            future_to_file = {
                executor.submit(
                    self._process_file_phase2, 
                    str(input_file), 
                    str(output_file),
                    normalization_target,
                    progress
                ): str(input_file)
                for input_file, output_file in zip(input_files, output_files)
            }
            
            # Collect normalization results
            normalization_results = []
            for future in concurrent.futures.as_completed(future_to_file):
                try:
                    norm_result = future.result()
                    normalization_results.append(norm_result)
                except Exception as e:
                    input_file = future_to_file[future]
                    normalization_results.append({
                        'file': input_file,
                        'error': str(e)
                    })
        
        progress.close()
        
        # Update the final LUFS values in the results
        for norm_result in normalization_results:
            if 'error' not in norm_result and 'final_lufs' in norm_result:
                for result in results:
                    if result.get('file') == norm_result['file']:
                        result['final_lufs'] = norm_result['final_lufs']
        
        return results

def main():
    """Command line interface for the audio processor."""
    parser = argparse.ArgumentParser(description='Automatic 2-pop remover and audio normalizer')
    
    # Input/output options
    parser.add_argument('input', help='Input audio file or folder')
    parser.add_argument('--output', '-o', help='Output audio file or folder (default: overwrite input)')
    
    # Processing options
    parser.add_argument('--target-lufs', '-l', type=float, default=-23.0, 
                      help='Target LUFS for normalization (default: -23.0)')
    parser.add_argument('--silence', '-s', type=float, default=0.5,
                      help='Duration of silence before music in seconds (default: 0.5)')
    parser.add_argument('--fade', '-f', type=int, default=100,
                      help='Number of samples for fade-in (default: 100)')
    parser.add_argument('--match-quietest', '-m', action='store_true',
                      help='Match all files to the quietest file (folder mode only)')
    parser.add_argument('--model', type=str, help='Path to TensorFlow Lite model for 2-pop detection')
    parser.add_argument('--threads', '-t', type=int, default=None,
                      help='Number of threads for parallel processing (default: CPU count - 1)')
    parser.add_argument('--basic-onset', action='store_true',
                      help='Use basic onset detection instead of advanced algorithm')
    parser.add_argument('--debug', '-d', action='store_true',
                      help='Enable detailed debugging output and sequential processing')
    parser.add_argument('--threshold', type=float, default=3.5,
                      help='Threshold factor for traditional 2-pop detection (default: 3.5)')
    
    args = parser.parse_args()
    
    # Set debug mode if requested
    if args.debug:
        os.environ['DEBUG'] = '1'
        print("Debug mode enabled - processing files sequentially with detailed output")
    
    # Initialize processor
    processor = AudioProcessor(
        target_lufs=args.target_lufs,
        silence_duration=args.silence,
        fade_samples=args.fade,
        model_path=args.model,
        num_threads=args.threads,
        advanced_onset=not args.basic_onset
    )
    
    # Set threshold factor if specified
    if args.threshold != 3.5:  # If different from default
        processor.threshold_factor = args.threshold
        print(f"Using custom threshold factor: {args.threshold}")
    
    # Process single file or folder
    input_path = Path(args.input)
    
    if input_path.is_file():
        # Single file processing
        result = processor.process_file(str(input_path), args.output)
        if 'error' in result:
            print(f"Error: {result['error']}")
        else:
            print(f"\nProcessing Summary for {os.path.basename(result['file'])}")
            print(f"2-Pops detected: {'Yes' if result['pops_detected'] else 'No'}")
            if result['pops_detected'] and result.get('pop_positions'):
                print(f"  Positions: {result['pop_positions'][0]/processor.sample_rate:.2f}s and {result['pop_positions'][1]/processor.sample_rate:.2f}s")
            print(f"Original LUFS: {result['original_lufs']:.1f}")
            print(f"Final LUFS: {result['final_lufs']:.1f}")
            
    elif input_path.is_dir():
        # Folder processing
        results = processor.process_folder(
            str(input_path), 
            args.output, 
            match_quietest=args.match_quietest
        )
        
        if isinstance(results, dict) and 'error' in results:
            print(f"Error: {results['error']}")
        else:
            print("\nProcessing Summary:")
            print(f"Total files: {len(results)}")
            
            # Accurately count files with 2-pops detected
            pops_detected = sum(1 for r in results if r.get('pops_detected', False))
            print(f"Files with 2-pops detected: {pops_detected}")
            
            # Show which files had 2-pops detected
            if pops_detected > 0:
                print("\nFiles with 2-pops detected:")
                for r in results:
                    if r.get('pops_detected', False):
                        print(f"  - {os.path.basename(r['file'])}")
            
            # Show which files that should have had 2-pops (based on name) but didn't
            missed_files = []
            for r in results:
                if not r.get('pops_detected', False) and '2pop' in os.path.basename(r['file']).lower():
                    missed_files.append(os.path.basename(r['file']))
            
            if missed_files:
                print("\nFiles that might have 2-pops but none detected:")
                for file in missed_files:
                    print(f"  - {file}")
            
            if not args.match_quietest:
                print(f"\nAll files normalized to: {args.target_lufs:.1f} LUFS")
            else:
                # Find minimum LUFS
                min_lufs = min((r['final_lufs'] for r in results if 'final_lufs' in r), default=None)
                if min_lufs:
                    print(f"\nAll files normalized to match quietest: {min_lufs:.1f} LUFS")
    else:
        print(f"Error: Input path {input_path} not found")
        
if __name__ == "__main__":
    main()