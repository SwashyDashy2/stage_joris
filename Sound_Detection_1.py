import pyaudio
import wave
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import time
import struct
import math
import threading
import sounddevice as sd
import matplotlib.animation as animation
from scipy.signal import find_peaks, butter, lfilter

# ==== CONFIG ====
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
PRE_BUFFER_SECONDS = 4
POST_BUFFER_SECONDS = 4
THRESHOLD = 2000
DEVICE_INDEX = 1  # Set your input device index here

LOW_CUT = 100.0
HIGH_CUT = 2000.0
FILTER_ORDER = 4
CHUNK_DURATION = CHUNK / RATE
PROMINENCE = 8

# Smoothing window (number of consecutive LEQ measurements)
SMOOTHING_WINDOW = 5

VISUALIZE_AFTER_RECORDING = True

# Trigger cooldown in seconds: new trigger events are not allowed more than once per second.
TRIGGER_COOLDOWN = 1.0
last_trigger_time = 0  # Global variable to hold the time of the last trigger

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
    # Use 'same' mode to preserve the shape of the data array.
    return np.convolve(data, np.ones(window_size) / window_size, mode='same')

# ==== PRE-BUFFER ====
pre_buffer_maxlen = int(RATE / CHUNK * PRE_BUFFER_SECONDS)
pre_buffer = deque(maxlen=pre_buffer_maxlen)
db_buffer = deque(maxlen=pre_buffer_maxlen)

# ==== AUDIO STREAM ====
p = pyaudio.PyAudio()
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                input_device_index=DEVICE_INDEX,
                frames_per_buffer=CHUNK)

print("Listening for threshold events...")

# Instead of a single recording, we use a list of active recordings.
# Each recording is a dict with its own 'frames', 'start_time', and optionally other data.
active_recordings = []

# ==== ANALYSIS FUNCTIONS ====
def calculate_leq_chunk(audio_data, sample_rate):
    squared_amplitude = np.square(audio_data)
    mean_squared_amplitude = np.mean(squared_amplitude)
    if mean_squared_amplitude <= 0:
        return -np.inf
    p_ref = 2e-5
    leq = 10 * np.log10(mean_squared_amplitude / (p_ref ** 2) + 1e-10)
    return leq

def calculate_leq_over_time(wav_file, chunk_duration=CHUNK_DURATION):
    with wave.open(wav_file, 'rb') as wf:
        num_channels = wf.getnchannels()
        sample_rate = wf.getframerate()
        num_frames = wf.getnframes()
        audio_frames = wf.readframes(num_frames)
        audio_data = np.frombuffer(audio_frames, dtype=np.int16)
        audio_data = audio_data.reshape(-1, num_channels)
        audio_data = np.sum(audio_data, axis=1)
        audio_data = audio_data.astype(np.float32) / np.iinfo(np.int16).max
        filtered_audio = bandpass_filter(audio_data, LOW_CUT, HIGH_CUT, sample_rate)

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

def find_significant_peaks(leq_values, time_stamps, prominence=PROMINENCE):
    # Smooth the LEQ values first so that short spikes are attenuated
    smoothed_leq = smooth_data(leq_values)
    peaks_indices, _ = find_peaks(smoothed_leq, prominence=prominence)
    max_leq = max(smoothed_leq)
    filtered_peaks = [(time_stamps[i], smoothed_leq[i])
                      for i in peaks_indices if smoothed_leq[i] >= 0.5 * max_leq]
    return filtered_peaks

def animate_plot(filtered_audio, fs, time_stamps, leq_values):
    fig, ax = plt.subplots(figsize=(10, 6))
    # Plot smoothed LEQ values for a cleaner look
    smoothed_leq = smooth_data(leq_values)
    ax.plot(time_stamps, smoothed_leq, label='Smoothed LEQ over Time', color='blue')
    peaks = find_significant_peaks(leq_values, time_stamps, prominence=5)
    if peaks:
        ax.scatter(*zip(*peaks), color='red', label='Significant Peaks', zorder=3)
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('LEQ (dB)')
    ax.set_title('LEQ Values with Detected Peaks')
    ax.grid(True)
    ax.legend()

    vertical_line = ax.axvline(x=0, color='red', linewidth=1)
    total_duration = len(filtered_audio) / fs
    delay_time = 0.5
    _start_time = [None]

    def update(frame):
        if _start_time[0] is None:
            _start_time[0] = time.time()
        elapsed = time.time() - _start_time[0] - delay_time
        elapsed = max(0, min(elapsed, total_duration))
        vertical_line.set_xdata([elapsed])
        return vertical_line,

    def play_audio():
        sd.play(filtered_audio, fs)
        sd.wait()

    threading.Thread(target=play_audio, daemon=True).start()
    ani = animation.FuncAnimation(fig, update, interval=50, blit=True, cache_frame_data=False)
    plt.tight_layout()
    plt.show()

# ==== MAIN LOOP ====
try:
    while True:
        data = stream.read(CHUNK, exception_on_overflow=False)
        pre_buffer.append(data)
        chunk_signal = np.frombuffer(data, dtype=np.int16)

        # Convert raw chunk to float and filter it for detection
        chunk_signal_float = chunk_signal.astype(np.float32) / np.iinfo(np.int16).max
        filtered_signal = bandpass_filter(chunk_signal_float, LOW_CUT, HIGH_CUT, RATE)
        chunk_db = calculate_leq_chunk(filtered_signal, RATE)
        db_buffer.append(chunk_db)

        trigger_by_amplitude = np.max(np.abs(chunk_signal)) > THRESHOLD

        trigger_by_peak = False
        if len(db_buffer) == db_buffer.maxlen:
            db_array = np.array(db_buffer)
            peaks_indices, _ = find_peaks(db_array, prominence=PROMINENCE)
            if peaks_indices.size > 0 and peaks_indices[-1] == len(db_array) - 1:
                trigger_by_peak = True

        current_time = time.time()
        # Check the cooldown period (1 second) before starting a new recording event.
        if (trigger_by_amplitude or trigger_by_peak) and (current_time - last_trigger_time >= TRIGGER_COOLDOWN):
            print("Trigger event detected! Starting a new recording event.")
            new_recording = {
                'frames': list(pre_buffer),  # Include pre-buffer data
                'start_time': current_time,
                'peak_count': 1  # If you want to count peaks per recording separately
            }
            active_recordings.append(new_recording)
            last_trigger_time = current_time

        # Append the current data chunk to every active recording event.
        finished_recordings = []
        for rec in active_recordings:
            rec['frames'].append(data)
            if current_time - rec['start_time'] >= POST_BUFFER_SECONDS:
                finished_recordings.append(rec)
        # Finalize finished recordings
        for rec in finished_recordings:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"vehicle_{timestamp}"
            if rec['peak_count'] > 1:
                filename += f"_multiPeak({rec['peak_count']})"
            filename += ".wav"

            with wave.open(filename, 'wb') as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(p.get_sample_size(FORMAT))
                wf.setframerate(RATE)
                wf.writeframes(b''.join(rec['frames']))

            print(f"Saved recording to {filename}")
            active_recordings.remove(rec)

except KeyboardInterrupt:
    print("Exiting...")

finally:
    stream.stop_stream()
    stream.close()
    p.terminate()
