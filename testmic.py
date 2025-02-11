import soundfile as sf
import os

# Geef de map aan waar de WAV-bestanden staan
folder_path = os.path.join('wavs', 'Barcelona_2022', 'segments', 'L_larger_90')

# Loop door alle bestanden in de map
for filename in os.listdir(folder_path):
    if filename.endswith(".wav"):  # Alleen .wav-bestanden
        file_path = os.path.join(folder_path, filename)
        
        try:
            # Lees de audio met soundfile
            data, sample_rate = sf.read(file_path)

            # Converteer naar 16-bit PCM (standaard Signed Integer PCM)
            output_path = os.path.join(folder_path, f"converted_{filename}")
            sf.write(output_path, data, sample_rate, subtype='PCM_16')

            print(f"Succesvol geconverteerd: {filename} naar 16-bit PCM")
        
        except Exception as e:
            print(f"Fout bij het verwerken van {filename}: {e}")
