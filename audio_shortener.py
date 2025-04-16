import os
import wave
import math
import numpy as np
import shutil
from scipy.signal import find_peaks

# =============== CONFIGURATIONS ===============
BASE_SOURCE_FOLDER = r"wavs\Cars Length Test\10 seconds"
TARGET_DURATIONS = [9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]  # iterative target durations
CHUNK_DURATION = 0.1   # seconds for LEQ calculation
PEAK_PROMINENCE = 5    # prominence threshold for peak detection
SAMPLE_WIDTH = 2       # 16-bit = 2 bytes per sample
P_REF = 2e-5           # reference pressure for dB calc in air
BEGIN_END_MARGIN = 1.0 # seconds to check for peaks near start/end

def read_wave_file(filepath):
    with wave.open(filepath, 'rb') as wf:
        num_channels = wf.getnchannels()
        sample_rate = wf.getframerate()
        num_frames = wf.getnframes()
        raw_data = wf.readframes(num_frames)
        audio_data = np.frombuffer(raw_data, dtype=np.int16)
        audio_data = audio_data.reshape(-1, num_channels)
        audio_data = np.mean(audio_data, axis=1)  # Convert to mono
        audio_data = audio_data.astype(np.float32) / np.iinfo(np.int16).max
    return audio_data, sample_rate

def write_wave_file(filepath, audio_data, sample_rate):
    audio_int16 = (audio_data * np.iinfo(np.int16).max).astype(np.int16)
    with wave.open(filepath, 'wb') as wf:
        wf.setnchannels(1)  # mono
        wf.setsampwidth(SAMPLE_WIDTH)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())

def shorten_audio(audio_data, sample_rate, keep_duration):
    total_length_sec = len(audio_data) / sample_rate
    remove_sec = total_length_sec - keep_duration

    margin_frames = int(BEGIN_END_MARGIN * sample_rate)
    total_frames = len(audio_data)
    start_region = audio_data[:margin_frames]
    end_region = audio_data[-margin_frames:]

    start_rms = np.sqrt(np.mean(start_region**2)) if len(start_region) > 0 else 0
    end_rms = np.sqrt(np.mean(end_region**2)) if len(end_region) > 0 else 0

    threshold_peak = 0.05
    has_peak_start = (start_rms >= threshold_peak)
    has_peak_end = (end_rms >= threshold_peak)

    remove_frames = int(remove_sec * sample_rate)

    if not has_peak_start and has_peak_end:
        new_audio = audio_data[remove_frames:]
    elif has_peak_start and not has_peak_end:
        new_audio = audio_data[:total_frames - remove_frames]
    elif not has_peak_start and not has_peak_end:
        new_audio = audio_data[remove_frames:]
    else:
        if abs(start_rms - end_rms) < 1e-5:
            half_frames = remove_frames // 2
            remainder = remove_frames - 2 * half_frames
            new_audio = audio_data[half_frames: total_frames - (half_frames + remainder)]
        elif start_rms < end_rms:
            new_audio = audio_data[remove_frames:]
        else:
            new_audio = audio_data[:total_frames - remove_frames]

    return new_audio

def process_folder_for_duration(source_folder, dest_folder, target_duration):
    os.makedirs(dest_folder, exist_ok=True)
    for fname in os.listdir(source_folder):
        if fname.lower().endswith(".wav"):
            src_path = os.path.join(source_folder, fname)
            audio_data, sr = read_wave_file(src_path)
            shortened = shorten_audio(audio_data, sr, keep_duration=target_duration)
            dst_path = os.path.join(dest_folder, fname)
            write_wave_file(dst_path, shortened, sr)
            print(f"Saved {target_duration:.1f}s file: {dst_path}")

def iterative_shorten(base_folder, target_durations):
    current_folder = base_folder
    for target in target_durations:
        # Create a new folder name based on target duration
        new_folder = os.path.join(os.path.dirname(base_folder), f"{int(target)} seconds")
        print(f"Processing files from {current_folder} to create {target:.1f}s segments in folder: {new_folder}")
        process_folder_for_duration(current_folder, new_folder, target)
        # Use the newly created folder as the source for the next iteration
        current_folder = new_folder

if __name__ == "__main__":
    iterative_shorten(BASE_SOURCE_FOLDER, TARGET_DURATIONS)
