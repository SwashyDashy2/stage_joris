import wave
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, butter, lfilter
import sounddevice as sd
import threading
import time
import matplotlib.animation as animation

# ==== CONFIGURATION ====
INPUT = r'vehicle_20250409_135326.wav'  # Your audio file path
LOW_CUT = 100.0
HIGH_CUT = 2000.0
FILTER_ORDER = 4
CHUNK = 1024
CHUNK_DURATION = CHUNK / 44100
PROMINENCE = 8
SMOOTHING_WINDOW = 5
VISUALIZE = True

# ==== FILTER FUNCTIONS ====
def butter_bandpass(lowcut, highcut, fs, order=FILTER_ORDER):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    return butter(order, [low, high], btype='band')

def bandpass_filter(data, lowcut, highcut, fs, order=FILTER_ORDER):
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    return lfilter(b, a, data)

# ==== SMOOTHING FUNCTION ====
def smooth_data(data, window_size=SMOOTHING_WINDOW):
    return np.convolve(data, np.ones(window_size)/window_size, mode='same')

# ==== LEQ Calculation ====
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

        # Mix to mono if stereo
        if num_channels > 1:
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
    return leq_values, time_stamps, filtered_audio, sample_rate

# ==== Peak Detection ====
def find_significant_peaks(leq_values, time_stamps, prominence=PROMINENCE):
    smoothed_leq = smooth_data(leq_values)
    peaks_indices, _ = find_peaks(smoothed_leq, prominence=prominence)
    max_leq = max(smoothed_leq)
    filtered_peaks = [(time_stamps[i], smoothed_leq[i])
                      for i in peaks_indices if smoothed_leq[i] >= 0.5 * max_leq]
    return filtered_peaks

# ==== Plot with Audio Playback ====
def animate_plot(filtered_audio, fs, time_stamps, leq_values):
    fig, ax = plt.subplots(figsize=(10, 6))
    smoothed_leq = smooth_data(leq_values)
    ax.plot(time_stamps, smoothed_leq, label='Smoothed LEQ', color='blue')
    peaks = find_significant_peaks(leq_values, time_stamps, prominence=PROMINENCE)
    if peaks:
        ax.scatter(*zip(*peaks), color='red', label='Significant Peaks', zorder=3)

    ax.set_xlabel('Time (s)')
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

# ==== MAIN EXECUTION ====
if __name__ == "__main__":
    leq_vals, time_stamps, audio_data, fs = calculate_leq_over_time(INPUT)
    peaks = find_significant_peaks(leq_vals, time_stamps)
    print(f"Detected {len(peaks)} trigger(s) at:")
    for t, db in peaks:
        print(f" - Time: {t:.2f}s, LEQ: {db:.2f} dB")

    if VISUALIZE:
        animate_plot(audio_data, fs, time_stamps, leq_vals)
