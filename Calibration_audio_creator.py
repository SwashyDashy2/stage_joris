import numpy as np
import scipy.io.wavfile as wav

# Parameters
fs = 44100  # Sample rate (Hz)
duration = 6  # Total duration in seconds
freq = 1000  # Testtoon (Hz)

# Definieer dB-niveaus per seconde (0 dBFS = max volume)
dB_levels = [0, -6, -12, -20, -40, -60]  # Verandert elke seconde

# Genereer de golf
t = np.linspace(0, duration, fs * duration, endpoint=False)
waveform = np.zeros_like(t)

for i, dB in enumerate(dB_levels):
    start = i * fs  # Startpunt in samples
    end = (i + 1) * fs  # Eindpunt in samples
    amplitude = 10**(dB / 20)  # dB omzetten naar amplitude
    waveform[start:end] = amplitude * np.sin(2 * np.pi * freq * t[start:end])

# Opslaan als WAV
wav.write("test_dB_levels.wav", fs, (waveform * 32767).astype(np.int16))

print("WAV-bestand met bekende dB-niveaus gegenereerd: test_dB_levels.wav")
