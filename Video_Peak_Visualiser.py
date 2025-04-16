import os
import numpy as np
from scipy.signal import find_peaks
from pydub import AudioSegment
import ffmpeg
import matplotlib.pyplot as plt
import sounddevice as sd
import time

# ==== CONFIGURATION ====
VIDEO_PATH = r"6_minuten_testfootage.mkv"  # Full path to your MKV file
PROMINENCE = 8          # Existing prominence parameter for peak detection
CHUNK_DURATION = 0.1     # Duration (in seconds) of each chunk for LEQ calculation
AUDIO_TEMP_FILE = "temp_audio.wav"  # Temporary file for audio extraction

# New parameters for additional filtering:
DESIRED_DB_THRESHOLD = 64.0  # Only consider peaks above 64 dB
MIN_WIDTH_SEC = 0.7          # Only consider peaks that are at least 1 second wide

# ==== Extract audio from video (using ffmpeg) ====
ffmpeg.input(VIDEO_PATH).output(
    AUDIO_TEMP_FILE, 
    acodec="pcm_s16le", ac=1, ar="44100", vn=None
).overwrite_output().run(quiet=True)

# ==== Load the audio using pydub ====
audio_seg = AudioSegment.from_wav(AUDIO_TEMP_FILE)
sample_rate = audio_seg.frame_rate

# Convert audio samples to a NumPy array (mono)
mono_samples = np.array(audio_seg.get_array_of_samples())

# ==== Calculate LEQ (dB) over CHUNK_DURATION chunks ====
chunk_size = int(sample_rate * CHUNK_DURATION)
leq_values = []
for i in range(0, len(mono_samples) - chunk_size, chunk_size):
    chunk = mono_samples[i:i + chunk_size]
    mean_power = np.mean(chunk.astype(np.float64) ** 2)
    leq = 10 * np.log10(mean_power + 1e-10)  # Avoid log(0)
    leq_values.append(leq)
leq_values = np.array(leq_values)
time_stamps = np.arange(len(leq_values)) * CHUNK_DURATION

# ==== Detect peaks in the LEQ signal using SciPy's find_peaks ====
# Compute the minimum width in number of samples:
min_width_samples = int(MIN_WIDTH_SEC / CHUNK_DURATION)

peaks_indices, props = find_peaks(
    leq_values,
    height=DESIRED_DB_THRESHOLD,  # Only peaks above desired dB threshold
    width=min_width_samples,      # Only peaks wide enough (in number of samples)
    prominence=PROMINENCE         # Existing prominence requirement
)
peak_times = time_stamps[peaks_indices]
peak_values = leq_values[peaks_indices]

print("Detected peak times (sec):", peak_times)

# ==== Plot the LEQ (dB) signal and overlay detected peaks ====
plt.figure(figsize=(12, 6))
plt.plot(time_stamps, leq_values, label="LEQ (dB)")
plt.plot(peak_times, peak_values, "ro", label="Detected Peaks")
plt.xlabel("Time (s)")
plt.ylabel("dB Level")
plt.title("LEQ over Time and Detected Peaks")
plt.ylim(40, 80)  # Limit y-axis between 40 and 80 dB
plt.legend()
plt.grid(True)

# Add a text box for a timer at the top-right of the plot
timer_text = plt.text(0.95, 0.95, "Time: 0.00 sec", ha="right", va="top",
                       transform=plt.gca().transAxes, fontsize=12, color='red')

# ==== Play audio and update timer in main thread ====
audio_duration = len(mono_samples) / sample_rate
start_time = time.time()
sd.play(mono_samples, samplerate=sample_rate)

# Update the timer until the audio is finished
while True:
    elapsed = time.time() - start_time
    timer_text.set_text(f"Time: {elapsed:.2f} sec")
    plt.draw()
    plt.pause(0.05)  # Allows GUI to update
    if elapsed >= audio_duration:
        break

sd.wait()  # Ensure audio playback is finished

# Keep the plot open after audio finishes
plt.show()

# ==== Clean up temporary audio file ====
if os.path.exists(AUDIO_TEMP_FILE):
    os.remove(AUDIO_TEMP_FILE)

print("✅ Finished!")
