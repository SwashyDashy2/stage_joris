import os
import numpy as np
from scipy.signal import find_peaks
from pydub import AudioSegment
import ffmpeg
import math

# ==== CONFIGURATION ====
VIDEO_PATH = r"C:\Code_mp\stage_joris\6_minuten_testfootage.mkv"  # Full path to your MKV file
OUTPUT_FOLDER = "test_peaks"
PROMINENCE = 8          # Base prominence for peak detection
CLIP_DURATION = 8        # seconds; exact clip length we want
PRE_PEAK_SECONDS = 4     # seconds before the peak; clip will be [peak - 4, peak + 4]
POST_PEAK_SECONDS = 4    # seconds after the peak
AUDIO_TEMP_FILE = "temp_audio.wav"  # Temporary file for audio extraction.

# New filtering parameters:
DESIRED_DB_THRESHOLD = 64.0  # Only detect peaks with LEQ >= 64 dB
MIN_WIDTH_SEC = 0.7          # Only detect peaks that last at least 1 second

# ==== Ensure output directory exists ====
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ==== Get video duration using ffmpeg.probe ====
probe = ffmpeg.probe(VIDEO_PATH)
video_duration = float(probe['format']['duration'])

# ==== Extract audio from video (using ffmpeg) ====
# Extract as a mono, 44100 Hz WAV file.
ffmpeg.input(VIDEO_PATH).output(
    AUDIO_TEMP_FILE,
    acodec="pcm_s16le", ac=1, ar="44100", vn=None
).overwrite_output().run(quiet=True)

# ==== Load the audio using pydub ====
audio_seg = AudioSegment.from_wav(AUDIO_TEMP_FILE)
sample_rate = audio_seg.frame_rate

# Convert audio samples to a NumPy array (mono)
mono_samples = np.array(audio_seg.get_array_of_samples())

# ==== Calculate power & LEQ over 100ms chunks ====
chunk_duration = 0.1  # seconds
chunk_size = int(sample_rate * chunk_duration)
leq_values = []
for i in range(0, len(mono_samples) - chunk_size, chunk_size):
    # Compute mean power in the chunk and convert to dB
    mean_power = np.mean(mono_samples[i:i + chunk_size].astype(np.float64) ** 2)
    leq = 10 * np.log10(mean_power + 1e-10)  # avoid log(0)
    leq_values.append(leq)

leq_values = np.array(leq_values)
# Create time stamps for each chunk
time_stamps = np.arange(len(leq_values)) * chunk_duration

# ==== Detect peaks in the LEQ signal with additional filtering ====
# Convert the minimum width from seconds to a number of chunks:
min_width_samples = int(MIN_WIDTH_SEC / chunk_duration)

peaks_indices, props = find_peaks(
    leq_values,
    height=DESIRED_DB_THRESHOLD,  # Only consider peaks >= DESIRED_DB_THRESHOLD
    width=min_width_samples,      # Only consider peaks that are at least MIN_WIDTH_SEC long
    prominence=PROMINENCE         # Existing prominence requirement
)
peak_times = time_stamps[peaks_indices]
peak_values = leq_values[peaks_indices]

print("Detected peak times (sec):", peak_times)

# ==== Filter out peaks that cannot produce an 8-second clip =====
# Only allow peaks that have at least 4 seconds before and after them.
valid_peaks = [t for t in peak_times if t >= PRE_PEAK_SECONDS and t <= (video_duration - POST_PEAK_SECONDS)]
print("Valid peaks for 8s clips (sec):", valid_peaks)

# ==== Group peaks that would yield overlapping 8s clips ====
# Each candidate clip is defined as [peak - 4, peak + 4]. 
# If a subsequent peak falls within the same 8-second window, group them together.
clips = []  # Each element: (clip_start, clip_end, list_of_peaks)
if valid_peaks:
    # Start the first group with the first valid peak
    current_group = [valid_peaks[0]]
    for t in valid_peaks[1:]:
        # If t is within an 8-second window starting at (first_peak - 4)...
        if t - current_group[0] < CLIP_DURATION:
            current_group.append(t)
        else:
            # Finalize the previous group
            clip_start = current_group[0] - PRE_PEAK_SECONDS
            clip_end = clip_start + CLIP_DURATION  # exactly 8 seconds
            clips.append((clip_start, clip_end, current_group))
            # Start a new group
            current_group = [t]
    # Finalize last group
    clip_start = current_group[0] - PRE_PEAK_SECONDS
    clip_end = clip_start + CLIP_DURATION
    clips.append((clip_start, clip_end, current_group))

print("Clip groups:")
for clip in clips:
    print(f"Clip from {clip[0]:.2f} to {clip[1]:.2f} sec, peaks: {clip[2]}")

# ==== Extract each clip with FFmpeg ====
peak_clip_counter = 1
multiple_peak_counter = 1

for clip_start, clip_end, group in clips:
    # Ensure the clip is exactly 8 seconds (should always be true)
    duration = clip_end - clip_start
    if abs(duration - CLIP_DURATION) > 0.001:
        continue

    if len(group) > 1:
        out_filename = f"multiple_peaks_{multiple_peak_counter}.mp4"
        multiple_peak_counter += 1
    else:
        out_filename = f"peak_{peak_clip_counter}.mp4"
        peak_clip_counter += 1

    output_filepath = os.path.join(OUTPUT_FOLDER, out_filename)
    print(f"Extracting 8s clip from {clip_start:.2f} sec to {clip_end:.2f} sec as {out_filename} (peaks: {len(group)})")
    
    try:
        (
            ffmpeg
            .input(VIDEO_PATH, ss=clip_start, t=CLIP_DURATION)
            .output(
                output_filepath,
                vcodec="libx264",
                acodec="aac",
                reset_timestamps=1
            )
            .overwrite_output()
            .run(quiet=True)
        )
    except ffmpeg.Error as e:
        print("Error extracting clip:", e)
    
# ==== Clean up temporary audio file ====
if os.path.exists(AUDIO_TEMP_FILE):
    os.remove(AUDIO_TEMP_FILE)

print(f"✅ Finished! {len(clips)} clips saved to '{OUTPUT_FOLDER}'")
