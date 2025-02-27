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

# For Liner.AI model loading (if installed)
try:
    import tensorflow as tf
    import json
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
            model_path (str, optional): Path to Liner.AI TFLite model for 2-pop detection
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
        
        # Try to load ML model if path is provided
        if model_path and os.path.exists(model_path):
            self._load_model(model_path)
            
    def _load_model(self, model_path):
        """
        Load the Liner.AI TFLite model for 2-pop detection.
        
        Args:
            model_path (str): Path to the saved TFLite model
        """
        try:
            # Check if TensorFlow is available
            if not TF_AVAILABLE:
                print("Warning: TensorFlow not available. ML-based detection disabled.")
                return False
                
            print(f"Loading 2-pop detection TFLite model from {model_path}...")
            
            # Load the TFLite model as an interpreter
            self.model = tf.lite.Interpreter(model_path=model_path)
            
            # Get input and output details
            self.input_details = self.model.get_input_details()
            self.output_details = self.model.get_output_details()
            
            # Load class mapping
            try:
                # Try to load classes from classes.json if available
                with open('classes.json', 'r') as f:
                    classes_data = json.load(f)
                    # Convert to list if it's a dict
                    if isinstance(classes_data, dict):
                        # Sort by value to get correct order
                        self.classes = [k for k, v in sorted(classes_data.items(), key=lambda item: item[1])]
                    else:
                        self.classes = classes_data
            except:
                # Default classes based on provided example
                print("Warning: Could not load classes.json, using default classes")
                self.classes = ["Music", "2pop"]
            
            print(f"Loaded classes: {self.classes}")
            
            # Test a dummy prediction to ensure everything works
            dummy_input = np.zeros(16000, dtype=np.float32)  # 1 second at 16 kHz
            
            # Resize tensor and allocate
            self.model.resize_tensor_input(self.input_details[0]['index'], (1, len(dummy_input)))
            self.model.allocate_tensors()
            
            # Set the input tensor
            self.model.set_tensor(self.input_details[0]['index'], dummy_input[None].astype('float32'))
            
            # Run inference
            self.model.invoke()
            
            # Test getting output
            _ = self.model.get_tensor(self.output_details[0]['index'])
            
            print("TFLite model loaded successfully!")
            return True
            
        except Exception as e:
            print(f"Warning: Failed to load model: {str(e)}")
            self.model = None
            return False
    
    def _detect_pops_ml(self, audio, sr):
        """
        Detect 2-pops using the TensorFlow Lite model.
        
        Args:
            audio (np.ndarray): Audio data as numpy array
            sr (int): Sample rate of the audio
            
        Returns:
            list: Indices of detected pops, empty list if not found
        """
        if self.model is None:
            return []
            
        try:
            # Convert to mono if stereo
            if len(audio.shape) > 1:
                audio_mono = np.mean(audio, axis=1)
            else:
                audio_mono = audio
                
            # Only analyze the first 5 seconds for pops
            analysis_window = min(len(audio_mono), int(5 * sr))
            analysis_audio = audio_mono[:analysis_window]
            
            # Resample to 16kHz if needed (the model expects 16kHz)
            TARGET_SR = 16000
            if sr != TARGET_SR:
                analysis_audio = librosa.resample(
                    analysis_audio, 
                    orig_sr=sr, 
                    target_sr=TARGET_SR
                )
                
            # Create segments to analyze with 1-second windows (16000 samples at 16kHz)
            window_size = TARGET_SR  # 1 second window
            hop_size = window_size // 2  # 50% overlap
            
            pop_indices = []
            
            # Process overlapping windows
            for i in range(0, len(analysis_audio) - window_size + 1, hop_size):
                segment = analysis_audio[i:i+window_size]
                
                # Ensure the segment is exactly window_size
                if len(segment) != window_size:
                    # Pad with zeros if needed
                    segment = np.pad(segment, (0, window_size - len(segment)))
                
                # Resize input tensor for this segment
                self.model.resize_tensor_input(
                    self.input_details[0]['index'], 
                    (1, len(segment))
                )
                self.model.allocate_tensors()
                
                # Set the input tensor
                self.model.set_tensor(
                    self.input_details[0]['index'], 
                    segment[None].astype('float32')
                )
                
                # Run inference
                self.model.invoke()
                
                # Get output
                class_scores = self.model.get_tensor(self.output_details[0]['index'])
                
                # Class 1 corresponds to "2pop" based on classes.json
                pop_class_idx = self.classes.index("2pop")
                confidence = class_scores[0][pop_class_idx]
                
                # If confidence is high enough, record this position
                pop_threshold = 0.7  # Adjust based on model performance
                if confidence > pop_threshold:
                    # Convert back to original sample rate
                    original_idx = int(i * (sr / TARGET_SR))
                    pop_indices.append(original_idx)
                    
            # Filter to find pairs that are about 1 second apart (typical for 2-pops)
            if len(pop_indices) >= 2:
                filtered_pairs = []
                for i in range(len(pop_indices) - 1):
                    for j in range(i + 1, len(pop_indices)):
                        distance = pop_indices[j] - pop_indices[i]
                        if 0.8 * sr <= distance <= 1.2 * sr:
                            filtered_pairs.append([pop_indices[i], pop_indices[j]])
                            
                # Return the pair with highest confidence
                if filtered_pairs:
                    return filtered_pairs[0]
            
            return []
            
        except Exception as e:
            print(f"ML model inference error: {str(e)}")
            print(f"Details: {e.__class__.__name__}")
            import traceback
            traceback.print_exc()
            return []
    
    def _detect_pops(self, audio, sr, threshold_factor=5.0, min_distance_samples=0.5*44100):
        """
        Detect 2-pops in the beginning of an audio file.
        Tries ML-based detection first if model is available, then falls back to traditional method.
        
        Args:
            audio (np.ndarray): Audio data as numpy array
            sr (int): Sample rate of the audio
            threshold_factor (float): Factor to multiply RMS for pop detection threshold
            min_distance_samples (int): Minimum distance between pops
            
        Returns:
            list: Indices of detected pops, empty list if not found
        """
        # Try ML-based detection first if model is available
        if self.model is not None:
            ml_results = self._detect_pops_ml(audio, sr)
            if ml_results and len(ml_results) == 2:
                return ml_results
                
        # Fall back to traditional method
        # Convert to mono if stereo
        if len(audio.shape) > 1:
            audio_mono = np.mean(audio, axis=1)
        else:
            audio_mono = audio
            
        # Only analyze the first 5 seconds for pops
        analysis_window = min(len(audio_mono), int(5 * sr))
        analysis_audio = audio_mono[:analysis_window]
        
        # Calculate the energy envelope
        window_size = int(0.01 * sr)  # 10ms window
        energy = np.array([
            np.sum(np.abs(analysis_audio[i:i+window_size])**2) 
            for i in range(0, len(analysis_audio) - window_size, window_size)
        ])
        
        # Smooth energy curve
        energy_smooth = gaussian_filter1d(energy, sigma=1)
        
        # Calculate threshold based on signal statistics
        rms = np.sqrt(np.mean(energy_smooth**2))
        threshold = rms * threshold_factor
        
        # Find peaks above threshold
        peaks = []
        for i in range(1, len(energy_smooth) - 1):
            if (energy_smooth[i] > energy_smooth[i-1] and 
                energy_smooth[i] > energy_smooth[i+1] and 
                energy_smooth[i] > threshold):
                peaks.append(i * window_size)
        
        # Filter peaks to find those that could be 2-pops (usually ~1 second apart)
        if len(peaks) < 2:
            return []
            
        pop_indices = []
        for i in range(len(peaks) - 1):
            for j in range(i + 1, len(peaks)):
                # Check if peaks are between 0.8 and 1.2 seconds apart (typical for 2-pops)
                distance = peaks[j] - peaks[i]
                if 0.8 * sr <= distance <= 1.2 * sr:
                    pop_indices = [peaks[i], peaks[j]]
                    # Return first valid pair found
                    return pop_indices
                    
        return pop_indices
    
    def _detect_music_onset(self, audio, sr, start_index=0, sensitivity=2.0):
        """
        Detect the onset of music in an audio file.
        Uses advanced detection if enabled, otherwise falls back to basic method.
        
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
        
        # 2. Spectral flux
        onset_env_spectral = librosa.onset.onset_strength(
            y=analysis_audio, 
            sr=sr,
            hop_length=hop_length,
            feature=librosa.feature.spectral_flux
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
            
        # Load audio file
        try:
            audio, sr = sf.read(input_file)
        except Exception as e:
            return {"error": f"Failed to read file {input_file}: {str(e)}"}
            
        # Store original LUFS
        original_lufs = self._measure_lufs(audio, sr)
        
        # Detect 2-pops
        pop_indices = self._detect_pops(audio, sr)
        
        # If 2-pops are detected, process the file
        if pop_indices and len(pop_indices) == 2:
            # Use the second pop as reference point
            second_pop_index = pop_indices[1]
            
            # Find music onset after the second pop
            music_onset_index = self._detect_music_onset(audio, sr, start_index=second_pop_index)
            
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
        result = self.process_file(str(input_file), str(output_file), normalize=False)
        
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
    parser.add_argument('--model', type=str, help='Path to Liner.AI model for 2-pop detection')
    parser.add_argument('--threads', '-t', type=int, default=None,
                      help='Number of threads for parallel processing (default: CPU count - 1)')
    parser.add_argument('--basic-onset', action='store_true',
                      help='Use basic onset detection instead of advanced algorithm')
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = AudioProcessor(
        target_lufs=args.target_lufs,
        silence_duration=args.silence,
        fade_samples=args.fade,
        model_path=args.model,
        num_threads=args.threads,
        advanced_onset=not args.basic_onset
    )
    
    # Process single file or folder
    input_path = Path(args.input)
    
    if input_path.is_file():
        # Single file processing
        result = processor.process_file(str(input_path), args.output)
        if 'error' in result:
            print(f"Error: {result['error']}")
        else:
            print(f"Processed: {result['file']}")
            print(f"2-Pops detected: {'Yes' if result['pops_detected'] else 'No'}")
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
            
            pops_detected = sum(1 for r in results if r.get('pops_detected', False))
            print(f"Files with 2-pops detected: {pops_detected}")
            
            if not args.match_quietest:
                print(f"All files normalized to: {args.target_lufs:.1f} LUFS")
            else:
                # Find minimum LUFS
                min_lufs = min((r['final_lufs'] for r in results if 'final_lufs' in r), default=None)
                if min_lufs:
                    print(f"All files normalized to match quietest: {min_lufs:.1f} LUFS")
    else:
        print(f"Error: Input path {input_path} not found")
        
if __name__ == "__main__":
    main()
