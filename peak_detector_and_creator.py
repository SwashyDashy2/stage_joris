import wave
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, butter, lfilter
import os

# ==== CONFIGURATION ====
INPUT_WAV_FILE = r'15 minute test\recording_15min.wav'  # Replace with your WAV file path
OUTPUT_DIR = r'./peaks_15mins'             # Directory to save output peak clips

# Analysis parameters (same as in your original code)
RATE_DESIRED = 44100         # sample rate (we assume the input file is at this rate)
CHUNK = 1024
CHUNK_DURATION = CHUNK / RATE_DESIRED
LOW_CUT = 100.0
HIGH_CUT = 2000.0
FILTER_ORDER = 4
PROMINENCE = 8
SMOOTHING_WINDOW = 5

# Snippet parameters: total snippet duration = 8 seconds (4 seconds before and 4 seconds after the peak)
PRE_BUFFER_SECONDS = 4
POST_BUFFER_SECONDS = 4

# ==== FILTER FUNCTIONS ====
def butter_bandpass(lowcut, highcut, fs, order=FILTER_ORDER):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut, highcut, fs, order=FILTER_ORDER):
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    return lfilter(b, a, data)

# ==== SMOOTHING FUNCTION ====
def smooth_data(data, window_size=SMOOTHING_WINDOW):
    # Convolve with a uniform window in 'same' mode to preserve array shape.
    return np.convolve(data, np.ones(window_size) / window_size, mode='same')

# ==== LEQ Calculation ====
def calculate_leq_chunk(audio_data, sample_rate):
    squared_amplitude = np.square(audio_data)
    mean_squared_amplitude = np.mean(squared_amplitude)
    if mean_squared_amplitude <= 0:
        return -np.inf
    p_ref = 2e-5  # reference pressure
    leq = 10 * np.log10(mean_squared_amplitude / (p_ref ** 2) + 1e-10)
    return leq

# ==== PROCESS THE WAV FILE: Compute LEQ over time ====
def calculate_leq_over_time(wav_file, chunk_duration=CHUNK_DURATION):
    with wave.open(wav_file, 'rb') as wf:
        num_channels = wf.getnchannels()
        sample_rate = wf.getframerate()
        num_frames = wf.getnframes()
        audio_frames = wf.readframes(num_frames)
        # Read the raw audio data (int16)
        audio_data = np.frombuffer(audio_frames, dtype=np.int16)
        # Mix to mono if necessary
        if num_channels > 1:
            audio_data = audio_data.reshape(-1, num_channels)
            audio_data = np.sum(audio_data, axis=1)
        # Normalize audio to the range [-1, 1]
        audio_data = audio_data.astype(np.float32) / np.iinfo(np.int16).max
        
        # Filter the entire audio signal
        filtered_audio = bandpass_filter(audio_data, LOW_CUT, HIGH_CUT, sample_rate)
        
        # Split filtered audio into chunks and compute LEQ
        chunk_size = int(sample_rate * chunk_duration)
        leq_values = []
        time_stamps = []
        for start in range(0, len(filtered_audio), chunk_size):
            chunk_data = filtered_audio[start:start + chunk_size]
            if len(chunk_data) == chunk_size:
                leq = calculate_leq_chunk(chunk_data, sample_rate)
                leq_values.append(leq)
                time_stamps.append(start / sample_rate)
    return leq_values, time_stamps, audio_data, sample_rate

# ==== PEAK DETECTION ====
def find_significant_peaks(leq_values, time_stamps, prominence=PROMINENCE):
    # Smooth the LEQ values first.
    smoothed_leq = smooth_data(leq_values, window_size=SMOOTHING_WINDOW)
    peaks_indices, _ = find_peaks(smoothed_leq, prominence=prominence)
    max_leq = max(smoothed_leq)
    # Only consider peaks that are at least 50% of the maximum smoothed LEQ
    filtered_peaks = [(time_stamps[i], smoothed_leq[i]) 
                      for i in peaks_indices if smoothed_leq[i] >= 0.5 * max_leq]
    return filtered_peaks

# ==== SAVE 8-SECOND AUDIO SNIPPETS AROUND EACH PEAK ====
def save_peak_snippets(audio_data, sample_rate, peaks, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    total_samples = len(audio_data)
    for t_peak, _ in peaks:
        # Compute the start and end indices
        start_index = int((t_peak - PRE_BUFFER_SECONDS) * sample_rate)
        end_index = int((t_peak + POST_BUFFER_SECONDS) * sample_rate)
        # Ensure indices are within valid bounds
        if start_index < 0:
            start_index = 0
        if end_index > total_samples:
            end_index = total_samples
        # Extract the snippet (audio_data is normalized; convert back to int16)
        snippet = (audio_data[start_index:end_index] * np.iinfo(np.int16).max).astype(np.int16)
        # Define output filename
        filename = f"Peak_{t_peak:.2f}.wav"
        filepath = os.path.join(output_dir, filename)
        # Write the snippet to a WAV file
        with wave.open(filepath, 'wb') as wf:
            wf.setnchannels(1)  # Mono
            wf.setsampwidth(2)  # 2 bytes per sample for int16
            wf.setframerate(sample_rate)
            wf.writeframes(snippet.tobytes())
        print(f"Saved snippet: {filepath}")

# ==== MAIN EXECUTION ====
if __name__ == "__main__":
    leq_values, time_stamps, audio_data, fs = calculate_leq_over_time(INPUT_WAV_FILE)
    peaks = find_significant_peaks(leq_values, time_stamps)
    print(f"Detected {len(peaks)} peak(s):")
    for t, val in peaks:
        print(f" - Peak at {t:.2f}s, LEQ: {val:.2f} dB")
    save_peak_snippets(audio_data, fs, peaks, OUTPUT_DIR)
