import soundfile as sf
import os

# Geef de map aan waar de bestanden staan
folder_path = os.path.join('wavs', 'Calibration')

# Geef de map aan waar de WAV-bestanden moeten worden opgeslagen
output_folder = os.path.join('wavs', 'Calibration_conv')

# Maak de outputmap aan als deze nog niet bestaat
os.makedirs(output_folder, exist_ok=True)

# Loop door alle bestanden in de map
for filename in os.listdir(folder_path):
    if filename.endswith(".wav"):  # Pas aan naar de extensie van de bestanden die je wilt converteren
        file_path = os.path.join(folder_path, filename)
        
        try:
            # Lees de audio met soundfile (werkt ook met .flac-bestanden)
            data, sample_rate = sf.read(file_path)

            # Zet de audio op in .wav-formaat en sla op in de nieuwe map
            output_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}.wav")
            sf.write(output_path, data, sample_rate)

            print(f"Succesvol geconverteerd: {filename} naar .wav")
        
        except Exception as e:
            print(f"Fout bij het verwerken van {filename}: {e}")
