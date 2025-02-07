from pydub import AudioSegment
import os

# Geef de map aan waar de WAV-bestanden staan
folder_path = os.path.join('wavs', 'Barcelona_2022', 'segments', 'L_larger_90')

# Loop door alle bestanden in de map
for filename in os.listdir(folder_path):
    if filename.endswith(".wav"):  # Alleen .wav-bestanden
        file_path = os.path.join(folder_path, filename)
        
        try:
            # Laad het .wav bestand in Pydub
            audio = AudioSegment.from_wav(file_path)

            # Zet de audio om naar 16-bit PCM (standaardkwaliteit)
            audio = audio.set_sample_width(2)  # 16-bit PCM is 2 bytes (16 bits)
            audio = audio.set_frame_rate(44100)  # Set de frame rate (sample rate)
            
            # Exporteer de geconverteerde file
            output_path = os.path.join(folder_path, f"converted_{filename}")
            audio.export(output_path, format="wav")
            print(f"Succesvol geconverteerd: {filename}")
        except Exception as e:
            print(f"Fout bij het converteren van {filename}: {e}")
