import os
import re

# Folder containing the .wav files
folder = r"sounds\ambulance"  # Change this to your folder path

# Get a list of files that match the pattern "sound_*.wav"
files = [f for f in os.listdir(folder) if f.lower().endswith(".wav") and f.startswith("sound_")]

# Function to extract the numeric part from a filename like "sound_123.wav"
def get_number(filename):
    match = re.search(r'sound_(\d+)\.wav', filename, re.IGNORECASE)
    return int(match.group(1)) if match else 0

# Sort files based on their numeric value
files.sort(key=get_number)

# Rename files:
# - First 200 files → ambulance_1.wav, ambulance_2.wav, ..., ambulance_200.wav
# - Next 50 files → firetruck_1.wav, firetruck_2.wav, ..., firetruck_50.wav
for i, filename in enumerate(files, start=1):
    old_path = os.path.join(folder, filename)
    if i <= 200:
        new_name = f"ambulance_{i}.wav"
    elif i <= 250:
        new_name = f"firetruck_{i - 200}.wav"
    else:
        # For files beyond 250, we do nothing.
        continue

    new_path = os.path.join(folder, new_name)
    os.rename(old_path, new_path)
    print(f"Renamed {filename} to {new_name}")
