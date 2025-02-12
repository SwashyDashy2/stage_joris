import numpy as np
import wave
import struct
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sounddevice as sd
import threading
import time
from scipy.signal import find_peaks  # Make sure to install SciPy (pip install scipy)

def calculate_leq_chunk(audio_data, sample_rate, chunk_size):
    squared_amplitude = np.square(audio_data)
    leq = 10 * np.log10(np.mean(squared_amplitude))
    return leq

def calculate_leq_over_time(wav_file, chunk_duration=0.1):
    with wave.open(wav_file, 'rb') as wf:
        num_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        num_frames = wf.getnframes()

        # Read all frames and unpack the data into an array
        audio_frames = wf.readframes(num_frames)
        format_string = "<" + str(num_frames * num_channels) + "h"
        audio_data = np.array(struct.unpack(format_string, audio_frames))

        # If the file is stereo (or more channels), use only the first channel
        if num_channels > 1:
            audio_data = audio_data[::num_channels]

        # Calculate chunk size (number of samples per chunk)
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

def find_significant_peaks(leq_values, time_stamps, prominence=3):
    """
    Detect peaks in the LEQ values based on a given prominence.
    Additionally, any peak with an LEQ value less than 50% of the maximum LEQ value is excluded.
    """
    peaks_indices, properties = find_peaks(leq_values, prominence=prominence)
    max_leq = max(leq_values)
    # Filter peaks: only include those with an LEQ value at least 50% of the maximum
    filtered_peaks = [(time_stamps[i], leq_values[i]) for i in peaks_indices if leq_values[i] >= 0.5 * max_leq]
    return filtered_peaks

# --- Main Execution ---
wav_file = r'wavs\example_1.wav'
leq_values, time_stamps, audio_data, sample_rate = calculate_leq_over_time(wav_file, chunk_duration=0.1)

# Convert audio_data to float32 normalized to [-1, 1] for sounddevice playback
audio_data = audio_data.astype(np.float32) / np.iinfo(np.int16).max

# --- Plot the LEQ graph ---
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(time_stamps, leq_values, label='LEQ over Time', color='blue')

# Detect significant peaks with our updated criteria
peaks = find_significant_peaks(leq_values, time_stamps, prominence=3)
if peaks:
    ax.scatter(*zip(*peaks), color='red', label='Significant Peaks', zorder=3)

ax.set_xlabel('Time (seconds)')
ax.set_ylabel('LEQ (dB)')
ax.set_title('LEQ Values with Playback Progress')
ax.grid(True)
ax.legend()

# --- Add a vertical red line that will move during playback ---
ymin, ymax = ax.get_ylim()
vertical_line = ax.axvline(x=0, color='red', linewidth=1)

# --- Determine the total duration of the audio ---
total_duration = len(audio_data) / sample_rate

# --- Animation update function with delay ---
start_time = None
delay_time = 0.5  # 0.5 second delay before the line starts moving

def update(frame):
    global start_time
    if start_time is None:
        start_time = time.time()

    # Calculate elapsed time considering the delay
    elapsed = time.time() - start_time - delay_time
    if elapsed < 0:
        elapsed = 0
    if elapsed > total_duration:
        elapsed = total_duration
    vertical_line.set_xdata([elapsed])  # Wrap elapsed in a list
    return vertical_line,

# --- Audio Playback Function ---
def play_audio():
    sd.play(audio_data, sample_rate)
    sd.wait()

# Start audio playback in a separate thread so it plays concurrently with the animation.
audio_thread = threading.Thread(target=play_audio, daemon=True)
audio_thread.start()

# --- Set up and start the animation ---
ani = animation.FuncAnimation(fig, update, interval=50, blit=True, cache_frame_data=False)

plt.tight_layout()
plt.show()
