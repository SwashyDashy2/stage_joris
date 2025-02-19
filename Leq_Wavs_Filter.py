import os
import pandas as pd
import wave
import struct
import numpy as np

# Global configuration variables
chunk_length = 4         # Lengte van de chunk in seconden
min_db = 76.3              # Minimum dB-waarde waaraan een chunk (piekkarakter) moet voldoen

def calculate_leq_chunk(audio_data):
    squared_amplitude = np.square(audio_data)
    mean_squared = np.mean(squared_amplitude)
    if mean_squared == 0:
        return -np.inf
    leq = 10 * np.log10(mean_squared)
    return leq

def calculate_leq_over_time(wav_file, analysis_chunk_duration=0.1):
    with wave.open(wav_file, 'rb') as wf:
        num_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        num_frames = wf.getnframes()

        audio_frames = wf.readframes(num_frames)
        format_string = "<" + str(num_frames * num_channels) + "h"
        audio_data = np.array(struct.unpack(format_string, audio_frames))

        if num_channels > 1:
            audio_data = audio_data[::num_channels]

    chunk_size = int(sample_rate * analysis_chunk_duration)

    leq_values = []
    time_stamps = []
    for start in range(0, len(audio_data), chunk_size):
        end = start + chunk_size
        chunk = audio_data[start:end]
        if len(chunk) == chunk_size:
            leq = calculate_leq_chunk(chunk)
            leq_values.append(leq)
            time_stamps.append(start / sample_rate)
    return leq_values, time_stamps, audio_data, sample_rate, sample_width

def save_wav(filename, audio_data, sample_rate, sample_width):
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        audio_frames = struct.pack("<" + str(len(audio_data)) + "h", *audio_data)
        wf.writeframes(audio_frames)

def categorize_leq(leq_value):
    category = (int(leq_value) // 10) * 10
    return f"{category}-{category + 10} dB"

def process_wav_files(directory, output_excel):
    extraction_info = []

    for filename in os.listdir(directory):
        if filename.lower().endswith(".wav"):
            filepath = os.path.join(directory, filename)
            print(f"Verwerken (pieken bepalen): {filename}")
            try:
                leq_values, time_stamps, audio_data, sample_rate, sample_width = calculate_leq_over_time(filepath)
                if not leq_values:
                    print(f"Geen data in {filename}")
                    continue
                extraction_info.append({
                    "filename": filename,
                    "filepath": filepath,
                    "leq_values": leq_values,
                    "time_stamps": time_stamps,
                    "audio_data": audio_data,
                    "sample_rate": sample_rate,
                    "sample_width": sample_width
                })
            except Exception as e:
                print(f"Fout bij verwerken {filename}: {e}")

    chunks_folder = os.path.join(directory, "Peak_dB_Values")
    os.makedirs(chunks_folder, exist_ok=True)

    chunk_results = []
    for info in extraction_info:
        filename = info["filename"]
        leq_values = info["leq_values"]
        time_stamps = info["time_stamps"]
        audio_data = info["audio_data"]
        sample_rate = info["sample_rate"]
        sample_width = info["sample_width"]

        peaks = []
        in_peak = False
        current_peak_index = None
        for i, leq in enumerate(leq_values):
            if leq >= min_db:
                if not in_peak:
                    in_peak = True
                    current_peak_index = i
                else:
                    if leq > leq_values[current_peak_index]:
                        current_peak_index = i
            else:
                if in_peak:
                    peaks.append(current_peak_index)
                    in_peak = False
                    current_peak_index = None
        if in_peak:
            peaks.append(current_peak_index)

        if not peaks:
            print(f"Geen pieken >= {min_db} dB in {filename}; geen chunk aangemaakt.")
            continue

        for peak_idx in peaks:
            peak_leq = leq_values[peak_idx]
            peak_time = time_stamps[peak_idx]
            peak_sample_index = int(peak_time * sample_rate)
            half_chunk_samples = int((chunk_length / 2) * sample_rate)

            if peak_sample_index < half_chunk_samples or (len(audio_data) - peak_sample_index) < half_chunk_samples:
                print(f"Niet genoeg audio rond de piek in {filename} op {peak_time:.2f}s")
                continue

            chunk_start = peak_sample_index - half_chunk_samples
            chunk_end = peak_sample_index + half_chunk_samples
            chunk_data = audio_data[chunk_start:chunk_end]

            chunk_filename = f"{os.path.splitext(filename)[0]}_chunk_{peak_time:.2f}s.wav"
            chunk_filepath = os.path.join(chunks_folder, chunk_filename)
            save_wav(chunk_filepath, chunk_data, sample_rate, sample_width)
            print(f"Chunk opgeslagen voor {filename} als {chunk_filename} (piektijd: {peak_time:.2f}s, {peak_leq:.2f} dB)")

            chunk_results.append([filename, peak_leq, categorize_leq(peak_leq), chunk_filename, peak_time])

    if chunk_results:
        df_chunks = pd.DataFrame(chunk_results, columns=["Source File", "Peak Leq (dB)", "Category", "Chunk File", "Timestamp (s)"])
        chunk_output_excel = "chunk_" + output_excel
        df_chunks.to_excel(chunk_output_excel, index=False)
        print(f"Chunk resultaten opgeslagen in {chunk_output_excel}")
    else:
        print("Geen chunks aangemaakt.")

if __name__ == "__main__":
    input_directory = r"wavs\Highway"
    output_file = "leq_results_highway.xlsx"
    process_wav_files(input_directory, output_file)
