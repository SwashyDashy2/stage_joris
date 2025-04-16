import os
import numpy as np
import wave
import struct
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sounddevice as sd
import threading
import time
from scipy.signal import find_peaks, butter, lfilter
import pandas as pd  # Needed for Excel export

# ==== CONFIG ====

# Single file analysis (when folder analysis is disabled)
wav_file = r'15 minute test\peaks_15mins\Peak_131.05 Car.wav'  # WAV file to analyze

# Folder analysis options (only used when ANALYZE_FOLDER = True)
ANALYZE_FOLDER = True                      # Set to True to analyze an entire folder
FOLDER_PATH = r'15 minute test\peaks_15mins' # Change to your folder path containing WAV files
EXCEL_OUTPUT_ENABLED = False                # Set to True to export results to an Excel file
EXCEL_OUTPUT_NAME = "10 Seconds_Analysis.xlsx"  # Name of the Excel file to save results

CHUNK_DURATION = 0.1  # Duration (in seconds) of each chunk for analysis
PROMINENCE = 6        # Minimum prominence for peak detection

# Band-pass filter configuration (adjust as needed)
LOW_CUT = 100.0     # Hz, low cutoff frequency to remove wind noise
HIGH_CUT = 2000.0   # Hz, high cutoff frequency to preserve speech and other desired sounds
FILTER_ORDER = 2    # Order of the filter

# === TOGGLE VISUALIZATION ===
VISUALIZE_AFTER_ANALYSIS = True  # Set to True to display the graph

# ---- Snippet extraction parameters ----
# We extract an 8-second snippet: 4 seconds before the peak and 4 seconds after.
PRE_BUFFER_SECONDS = 4
POST_BUFFER_SECONDS = 4

# ---- Folder output for peak snippets ----
# Peak snippets will be saved to this folder (created if it doesn't exist)
PEAK_SNIPPET_OUTPUT_DIR = r'peak_snippets'

# ---- Band-Pass Filter Functions ----
def butter_bandpass(lowcut, highcut, fs, order=FILTER_ORDER):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut, highcut, fs, order=FILTER_ORDER):
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    return lfilter(b, a, data)

# ---- LEQ Calculation Functions ----
def calculate_leq_chunk(audio_data, sample_rate, chunk_size):
    """Calculate the LEQ (dB) for a given audio chunk."""
    squared_amplitude = np.square(audio_data)
    p_ref = 2e-5  # Reference pressure (20 µPa)
    leq = 10 * np.log10(np.mean(squared_amplitude) / (p_ref ** 2) + 1e-10)
    return leq

def calculate_leq_over_time(file_path, combine_channels=True, chunk_duration=CHUNK_DURATION):
    """
    Load the WAV file, optionally combine channels, apply a band-pass filter,
    and calculate LEQ values over time.
    Returns lists of LEQ values and timestamps, the filtered audio, and the sample rate.
    """
    with wave.open(file_path, 'rb') as wf:
        num_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        num_frames = wf.getnframes()

        # Read all frames and unpack into an array
        audio_frames = wf.readframes(num_frames)
        audio_data = np.frombuffer(audio_frames, dtype=np.int16)

        # Reshape data into (num_samples, num_channels)
        audio_data = audio_data.reshape(-1, num_channels)

        # Combine channels if desired (here we sum the channels)
        if combine_channels:
            audio_data = np.sum(audio_data, axis=1)
            # Alternatively, use average: audio_data = np.mean(audio_data, axis=1)

        # Normalize to float32 in range [-1, 1]
        audio_data = audio_data.astype(np.float32) / np.iinfo(np.int16).max

        # Apply band-pass filter
        audio_data = bandpass_filter(audio_data, LOW_CUT, HIGH_CUT, sample_rate)

        # Chunk processing
        chunk_size = int(sample_rate * chunk_duration)
        leq_values = []
        time_stamps = []
        for start in range(0, len(audio_data), chunk_size):
            end = start + chunk_size
            chunk_data = audio_data[start:end]
            if len(chunk_data) == chunk_size:
                leq = calculate_leq_chunk(chunk_data, sample_rate, chunk_size)
                leq_values.append(leq)
                time_stamps.append(start / sample_rate)

    return leq_values, time_stamps, audio_data, sample_rate

def find_significant_peaks(leq_values, time_stamps, prominence=PROMINENCE):
    """
    Detect peaks in the LEQ values using a prominence threshold.
    Only peaks that reach at least half the maximum LEQ are retained.
    """
    peaks_indices, _ = find_peaks(leq_values, prominence=prominence)
    max_leq = max(leq_values)
    filtered_peaks = [(time_stamps[i], leq_values[i]) for i in peaks_indices if leq_values[i] >= 0.5 * max_leq]
    return filtered_peaks

# ---- Save Peak Snippets ----
def save_peak_snippets(audio_data, sample_rate, peaks, output_dir):
    """
    For each peak (given as a tuple (time, value)),
    extract an 8-second snippet (4 sec before and 4 sec after the peak)
    and save it as a WAV file with the filename "Peak_xx.xx.wav".
    """
    os.makedirs(output_dir, exist_ok=True)
    total_samples = len(audio_data)
    for t_peak, _ in peaks:
        start_index = int((t_peak - PRE_BUFFER_SECONDS) * sample_rate)
        end_index = int((t_peak + POST_BUFFER_SECONDS) * sample_rate)
        if start_index < 0:
            start_index = 0
        if end_index > total_samples:
            end_index = total_samples
        snippet = (audio_data[start_index:end_index] * np.iinfo(np.int16).max).astype(np.int16)
        filename = f"Peak_{t_peak:.2f}.wav"
        filepath = os.path.join(output_dir, filename)
        with wave.open(filepath, 'wb') as wf:
            wf.setnchannels(1)  # mono
            wf.setsampwidth(2)  # 2 bytes (16-bit)
            wf.setframerate(sample_rate)
            wf.writeframes(snippet.tobytes())
        print(f"Saved snippet: {filepath}")

def analyze_and_visualize(file_path):
    """
    Analyze the WAV file, print trigger information,
    display an LEQ graph with detected peaks and audio playback
    (if visualization is enabled), and return the detected peaks,
    the processed audio data, and sample rate.
    """
    leq_values, time_stamps, audio_data, sample_rate = calculate_leq_over_time(file_path, combine_channels=True)
    peaks = find_significant_peaks(leq_values, time_stamps, prominence=PROMINENCE)
    num_peaks = len(peaks)
    
    if peaks:
        if num_peaks > 1:
            print(f"{os.path.basename(file_path)}: Multiple peaks detected ({num_peaks} peaks).")
        else:
            print(f"{os.path.basename(file_path)}: A single peak was detected.")
    else:
        print(f"{os.path.basename(file_path)}: No significant peaks were detected.")
    
    if VISUALIZE_AFTER_ANALYSIS:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(time_stamps, leq_values, label='LEQ over Time', color='blue')
        if peaks:
            ax.scatter(*zip(*peaks), color='red', label='Significant Peaks', zorder=3)
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('LEQ (dB)')
        ax.set_title(f'LEQ Values: {os.path.basename(file_path)}')
        ax.grid(True)
        ax.legend()

        # --- Playback & Animation ---
        vertical_line = ax.axvline(x=0, color='red', linewidth=1)
        total_duration = len(audio_data) / sample_rate
        start_time = None
        delay_time = 0.5

        def update(frame):
            nonlocal start_time
            if start_time is None:
                start_time = time.time()
            elapsed = time.time() - start_time - delay_time
            elapsed = max(0, min(elapsed, total_duration))
            vertical_line.set_xdata([elapsed])
            return vertical_line,

        #def play_audio():
         #   sd.play(audio_data, sample_rate)
          #  sd.wait()

        #audio_thread = threading.Thread(target=play_audio, daemon=True)
        #audio_thread.start()
        ani = animation.FuncAnimation(fig, update, interval=50, blit=True, cache_frame_data=False)
        plt.tight_layout()
        plt.show()
    
    return peaks, audio_data, sample_rate

if __name__ == '__main__':
    if ANALYZE_FOLDER:
        # Analyze all WAV files in the specified folder.
        results = []
        for file in os.listdir(FOLDER_PATH):
            if file.lower().endswith('.wav'):
                full_path = os.path.join(FOLDER_PATH, file)
                peaks, audio_data, sample_rate = analyze_and_visualize(full_path)
                results.append({"File": file, "Peaks": len(peaks)})
                # Save peak audio snippets for this file in a subfolder.
                output_dir = os.path.join(FOLDER_PATH, "peak_snippets")
                save_peak_snippets(audio_data, sample_rate, peaks, output_dir)
        if results:
            df = pd.DataFrame(results)
            print(df)
            if EXCEL_OUTPUT_ENABLED:
                excel_output_path = os.path.join(FOLDER_PATH, EXCEL_OUTPUT_NAME)
                df.to_excel(excel_output_path, index=False)
                print(f"Excel file saved to {excel_output_path}")
    else:
        # Single file analysis mode
        peaks, audio_data, sample_rate = analyze_and_visualize(wav_file)
        output_dir = os.path.join(os.path.dirname(wav_file), "peak_snippets")
        save_peak_snippets(audio_data, sample_rate, peaks, output_dir)
