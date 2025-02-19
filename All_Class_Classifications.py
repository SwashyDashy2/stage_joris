"""
Dit programma is ontwikkeld om geitengeluiden te detecteren in een map met audiobestanden die zijn opgenomen nabij een geitenboerderij. De gedetecteerde geluiden worden opgeslagen in een outputbestand voor verdere analyse.

Functionaliteiten:
- Analyseert meerdere audiobestanden in een opgegeven map en detecteert geitengeluiden.
- Slaat gedetecteerde geluidsevents op in een Excel-bestand, inclusief:
  - De naam van het audiobestand waarin het geitengeluid is gedetecteerd.
  - Het tijdstip waarop het geitengeluid is waargenomen.
  - De rang van het geluid binnen de classificatie (eerste, tweede of derde meest waarschijnlijke classificatie).
- Geluiden die buiten de top 3 classificaties vallen, worden niet opgeslagen.

Belangrijke parameters:
- `N_CLASSES`: Moet groter zijn dan 3 om voldoende classificaties te waarborgen.
- `ROOT_NAME`: Specificeert het classificatiefilter (standaard ingesteld op "Goat").
- `OUTPUT_EXCEL`: Bepaalt de locatie waar het gegenereerde Excel-bestand wordt opgeslagen.

Dit programma vereist als input een map met verschillende audiobestanden. Hoewel de bredere toepassing van deze detectie niet volledig gedefinieerd is, heeft het bewezen effectief te zijn bij het identificeren van geitengeluiden binnen grote datasets.
"""

import os
import json
import numpy as np
import pandas as pd
from scipy.io import wavfile
from mediapipe.tasks import python
from mediapipe.tasks.python.components import containers
from mediapipe.tasks.python import audio

# --- Configuration Parameters ---
N_CLASSES = 3
MIN_SCORE = 0.01
ROOT_NAME = "Goat"  # Root category to search in JSON
INTERVAL_MS = 975   # Analysis interval in milliseconds
MODEL_PATH = 'yamnet_classifier.tflite'
JSON_PATH = "Depth_mapping_Mediapipe.json"
AUDIO_DIR = r'C:\Code_mp\stage_joris\wavs\Geiten_conv'
OUTPUT_EXCEL = "goat_events2.xlsx"

# --- Load JSON and get valid names ---
def get_valid_names(data, root_name):
    def find_entry(name):
        return next((entry for entry in data if entry['name'] == name), None)
    
    def gather_names(entry):
        names = [entry['name']]
        for child_id in entry.get('child_ids', []):
            child_entry = next((item for item in data if item['id'] == child_id), None)
            if child_entry:
                names.extend(gather_names(child_entry))
        return names
    
    root_entry = find_entry(root_name)
    return gather_names(root_entry) if root_entry else []

with open(JSON_PATH, "r") as file:
    json_data = json.load(file)

valid_names = get_valid_names(json_data, ROOT_NAME)
if not valid_names:
    raise ValueError(f"Root name '{ROOT_NAME}' not found in JSON data.")

# --- Function to classify one audio file and record goat events ---
def classify_audio_file(audio_file_path, model_path, n_classes, interval_ms, min_score, valid_names):
    """
    Processes an audio file and returns a list of events.
    Each event is a tuple: (file_name, time_in_seconds, position)
    where 'position' is the rank (1, 2, or 3) of the classification.
    """
    events = []
    # Load audio and normalize to int16
    sample_rate, audio_data = wavfile.read(audio_file_path)
    audio_data = (audio_data / np.max(np.abs(audio_data)) * np.iinfo(np.int16).max).astype(np.int16)
    
    # Load the model and run the classifier
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = audio.AudioClassifierOptions(base_options=base_options, max_results=n_classes)
    with audio.AudioClassifier.create_from_options(options) as classifier:
        audio_clip = containers.AudioData.create_from_array(
            audio_data.astype(float) / np.iinfo(np.int16).max, sample_rate
        )
        classification_results = classifier.classify(audio_clip)
    
    duration = len(audio_data) / sample_rate  # duration in seconds
    
    # Process each interval in the audio file
    for timestamp in range(0, int(duration * 1000), interval_ms):
        idx = timestamp // interval_ms
        if idx < len(classification_results):
            # Get the top three classifications for this interval
            classifications = classification_results[idx].classifications[0].categories
            for pos, cat in enumerate(classifications[:3], start=1):
                if cat.category_name in valid_names and cat.score >= min_score:
                    events.append((os.path.basename(audio_file_path), timestamp / 1000, pos))
    return events

# --- Loop over all WAV files in the folder ---
audio_files = [os.path.join(AUDIO_DIR, f) for f in os.listdir(AUDIO_DIR) if f.lower().endswith('.wav')]
total_files = len(audio_files)
all_events = []

for i, file_path in enumerate(audio_files):
    events = classify_audio_file(file_path, MODEL_PATH, N_CLASSES, INTERVAL_MS, MIN_SCORE, valid_names)
    all_events.extend(events)
    # Print progress in the format "current/total"
    print(f"{i+1}/{total_files}")

# --- Write events to an Excel file ---
df = pd.DataFrame(all_events, columns=["File Name", "Time (s)", "Position"])
df.to_excel(OUTPUT_EXCEL, index=False)
