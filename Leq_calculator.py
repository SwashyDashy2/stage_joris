"""
Dit programma analyseert een audiobestand en bepaalt het geluidsniveau met een interval van 0,1 seconden.

Functionaliteiten:
- Het geluidsniveau wordt uitgedrukt in decibels (dB) ten opzichte van een referentiewaarde van 20 µPa.
- Detecteert pieken in het signaal, bijvoorbeeld om het aantal keer dat een hond blaft te tellen.
- Visualiseert het geluidsniveau in een grafiek over tijd.
- Speelt het audiofragment synchroon af met de grafiek.
- Een rode verticale lijn in de grafiek geeft aan hoever het afspelen van het fragment gevorderd is.
"""

import numpy as np
import wave
import struct
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sounddevice as sd
import threading
import time
from scipy.signal import find_peaks

wav_file = r'wavs\sirene\police-siren-21498.mp3'  #Wav bestand wat geanalyseerd wordt.

def calculate_leq_chunk(audio_data, sample_rate, chunk_size):
    squared_amplitude = np.square(audio_data)
    p_ref = 2e-5  # Referentiedruk voor geluid in lucht (20 µPa)
    leq = 10 * np.log10(np.mean(squared_amplitude) / (p_ref ** 2) + 1e-10)
    return leq

def calculate_leq_over_time(wav_file, combine_channels=True, chunk_duration=0.1):
    with wave.open(wav_file, 'rb') as wf:
        num_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        num_frames = wf.getnframes()

        # Read all frames and unpack into an array
        audio_frames = wf.readframes(num_frames)
        audio_data = np.frombuffer(audio_frames, dtype=np.int16)

        # Reshape the data into (num_samples, num_channels)
        audio_data = audio_data.reshape(-1, num_channels)

        # Combine all channels (sum or average)
        if combine_channels:
            audio_data = np.sum(audio_data, axis=1)  # Summing all channels
            # Alternatively, you can average the channels instead:
            # audio_data = np.mean(audio_data, axis=1)

        # Normalize to float [-1,1]
        audio_data = audio_data.astype(np.float32) / np.iinfo(np.int16).max

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

def find_significant_peaks(leq_values, time_stamps, prominence=3):
    peaks_indices, _ = find_peaks(leq_values, prominence=prominence)
    max_leq = max(leq_values)
    filtered_peaks = [(time_stamps[i], leq_values[i]) for i in peaks_indices if leq_values[i] >= 0.5 * max_leq]
    return filtered_peaks


leq_values, time_stamps, audio_data, sample_rate = calculate_leq_over_time(wav_file, combine_channels=True, chunk_duration=0.1)

# Plot the LEQ graph
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(time_stamps, leq_values, label='LEQ over Time', color='blue')

# Detect peaks
peaks = find_significant_peaks(leq_values, time_stamps, prominence=5)
if peaks:
    ax.scatter(*zip(*peaks), color='red', label='Significant Peaks', zorder=3)

ax.set_xlabel('Time (seconds)')
ax.set_ylabel('LEQ (dB)')
ax.set_title(f'LEQ Values (Combined Channels)')
ax.grid(True)
ax.legend()

# --- Playback & Animation ---
ymin, ymax = ax.get_ylim()
vertical_line = ax.axvline(x=0, color='red', linewidth=1)

total_duration = len(audio_data) / sample_rate
start_time = None
delay_time = 0.5  

def update(frame):
    global start_time
    if start_time is None:
        start_time = time.time()
    elapsed = time.time() - start_time - delay_time
    elapsed = max(0, min(elapsed, total_duration))
    vertical_line.set_xdata([elapsed])
    return vertical_line,

def play_audio():
    sd.play(audio_data, sample_rate)
    sd.wait()

audio_thread = threading.Thread(target=play_audio, daemon=True)
audio_thread.start()

ani = animation.FuncAnimation(fig, update, interval=50, blit=True, cache_frame_data=False)
plt.tight_layout()
plt.show()
