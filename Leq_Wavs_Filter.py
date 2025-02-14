import os
import pandas as pd
import wave
import struct
import numpy as np

def calculate_leq_chunk(audio_data, sample_rate, chunk_size):
    """
    Bereken de Leq voor een data-chunk.
    """
    squared_amplitude = np.square(audio_data)
    # Zorg dat we geen log10(0) proberen te berekenen:
    mean_squared = np.mean(squared_amplitude)
    if mean_squared == 0:
        return -np.inf
    leq = 10 * np.log10(mean_squared)
    return leq

def calculate_leq_over_time(wav_file, chunk_duration=0.1):
    """
    Berekent de Leq-waarden over de tijd voor het gegeven .wav-bestand.
    De audio wordt in stukjes (chunks) van chunk_duration (standaard 0.1 sec) opgedeeld.
    """
    with wave.open(wav_file, 'rb') as wf:
        num_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        num_frames = wf.getnframes()

        # Lees alle frames en pak de data uit
        audio_frames = wf.readframes(num_frames)
        format_string = "<" + str(num_frames * num_channels) + "h"
        audio_data = np.array(struct.unpack(format_string, audio_frames))

        # Als het bestand stereo (of meer kanalen) is, gebruik alleen het eerste kanaal
        if num_channels > 1:
            audio_data = audio_data[::num_channels]

        # Bepaal de chunkgrootte (aantal samples per chunk)
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

def categorize_leq(leq_value):
    """
    Bepaalt in welke dB-categorie de Leq-waarde valt.
    Voorbeeld: een waarde tussen 0 en 10 dB wordt gecategoriseerd als '0-10 dB'.
    """
    category = (int(leq_value) // 10) * 10
    return f"{category}-{category + 10} dB"

def process_wav_files(directory, output_excel):
    """
    Leest alle .wav-bestanden in de opgegeven map, berekent de maximale Leq-waarde
    per bestand en slaat de resultaten (inclusief categorie) op in een Excel-bestand.
    """
    results = []
    
    for filename in os.listdir(directory):
        if filename.lower().endswith(".wav"):
            filepath = os.path.join(directory, filename)
            print(f"Verwerken: {filename}")
            
            try:
                leq_values, _, _, _ = calculate_leq_over_time(filepath)
                if leq_values:
                    max_leq = max(leq_values)
                    category = categorize_leq(max_leq)
                else:
                    max_leq = None
                    category = "Geen data"
                results.append([filename, max_leq, category])
            except Exception as e:
                print(f"Fout bij verwerken {filename}: {e}")
                results.append([filename, "Fout", "Fout"])
    
    # Data opslaan in een Excel-bestand
    df = pd.DataFrame(results, columns=["Bestand", "Max Leq (dB)", "Categorie"])
    df.to_excel(output_excel, index=False)
    print(f"Resultaten opgeslagen in {output_excel}")

if __name__ == "__main__":
    # Pas deze map aan naar waar jouw .wav-bestanden staan
    input_directory = r"wavs\Geiten_conv"
    output_file = "leq_results_geiten.xlsx"
    process_wav_files(input_directory, output_file)
