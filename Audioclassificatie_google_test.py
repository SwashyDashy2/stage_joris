import numpy as np
import os
import json
import matplotlib.pyplot as plt
from scipy.io import wavfile
from IPython.display import Audio, display
from mediapipe.tasks import python
from mediapipe.tasks.python.components import containers
from mediapipe.tasks.python import audio
from collections import Counter, defaultdict

# --- Configuration Parameters ---
N_CLASSES = 10
MIN_SCORE = 0.1
MIN_CLASSIFICATIONS = 10
MIN_PROBABILITY_GRAPH = 0.15
ROOT_NAME = "Vehicle"
FILTER_APPLIED = False
TOP_THREE = False  # Whether to display only top-1 or top-3 in the scatter plot/histogram
SCAT_MIN_CLASS = 4
AUDIO_FILE_PATH = r'wavs\hond.wav'
MODEL_PATH = 'yamnet_classifier.tflite'
JSON_PATH = "Depth_mapping_Mediapipe.json"
INTERVAL_MS = 975  # Analysis interval in milliseconds

# Validate Audio File
if not os.path.isfile(AUDIO_FILE_PATH):
    raise FileNotFoundError(f"Audio file not found: {AUDIO_FILE_PATH}")

# Load and Normalize Audio
sample_rate, audio_data = wavfile.read(AUDIO_FILE_PATH)
audio_data = (audio_data / np.max(np.abs(audio_data)) * np.iinfo(np.int16).max).astype(np.int16)
display(Audio(audio_data[:1], rate=sample_rate, autoplay=False))

# Load Model and Classify Audio
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = audio.AudioClassifierOptions(base_options=base_options, max_results=N_CLASSES)

with audio.AudioClassifier.create_from_options(options) as classifier:
    audio_clip = containers.AudioData.create_from_array(
        audio_data.astype(float) / np.iinfo(np.int16).max, sample_rate
    )
    classification_results = classifier.classify(audio_clip)

duration = len(audio_data) / sample_rate

# Process Classification Results
time_stamps, sources, second_sources, third_sources = [], [], [], []
all_top10 = []   # NEW: To store the top 10 classes (each as (class, probability)) at every timestep
class_counts = Counter()

for timestamp in range(0, int(duration * 1000), INTERVAL_MS):
    idx = timestamp // INTERVAL_MS
    if idx < len(classification_results):
        classifications = classification_results[idx].classifications[0].categories
        # Build a list with the top 10 predictions and their probabilities
        top10 = [(cat.category_name, cat.score) for cat in classifications[:N_CLASSES]]
        all_top10.append(top10)
        
        # Print the top 10 classes and probabilities for the current timestamp
        print(f"\nTimestamp {timestamp / 1000:.3f} seconds:")
        for rank, (cls, prob) in enumerate(top10, 1):
            print(f"  {rank}. {cls}: {prob:.2f}")

        if classifications[0].score >= MIN_PROBABILITY_GRAPH:
            time_stamps.append(timestamp / 1000)
            sources.append(classifications[0].category_name)
            second_sources.append(classifications[1].category_name if len(classifications) > 1 else None)
            third_sources.append(classifications[2].category_name if len(classifications) > 2 else None)
        for category in classifications[:N_CLASSES]:
            if category.score >= MIN_SCORE:
                class_counts[category.category_name] += 1

filtered_class_counts = {k: v for k, v in class_counts.items() if v >= MIN_CLASSIFICATIONS}

# Load and Validate JSON Data
def get_valid_names(data, root_name):
    def find_entry(name):
        return next((entry for entry in data if entry['name'] == name), None)
    
    def gather_names(entry):
        names = [entry['name']]
        for child_id in entry['child_ids']:
            child_entry = next((item for item in data if item['id'] == child_id), None)
            if child_entry:
                names.extend(gather_names(child_entry))
        return names
    
    root_entry = find_entry(root_name)
    return gather_names(root_entry) if root_entry else []

with open(JSON_PATH, "r") as file:
    data = json.load(file)

if not any(entry['name'] == ROOT_NAME for entry in data):
    raise ValueError(f"Root name '{ROOT_NAME}' not found in data.")

valid_names = get_valid_names(data, ROOT_NAME)

# Filter Data Based on Valid Names
filtered_sources, filtered_time_stamps, filtered_second_sources, filtered_third_sources = [], [], [], []

for ms, ss, ts, timestamp in zip(sources, second_sources, third_sources, time_stamps):
    if ms in valid_names:
        filtered_sources.append(ms)
        filtered_second_sources.append(ss)
        filtered_third_sources.append(ts)
        filtered_time_stamps.append(timestamp)

# Apply Filters (if FILTER_APPLIED is False, use original lists)
scatter_sources = filtered_sources if FILTER_APPLIED else sources
scatter_time_stamps = filtered_time_stamps if FILTER_APPLIED else time_stamps
scatter_second_sources = filtered_second_sources if FILTER_APPLIED else second_sources
scatter_third_sources = filtered_third_sources if FILTER_APPLIED else third_sources

# Filter Scatter Data based on SCAT_MIN_CLASS threshold
list_of_sources = scatter_sources + scatter_second_sources + scatter_third_sources
scatter_class_counts = Counter(list_of_sources)
allowed_classes = [key for key, value in scatter_class_counts.items() if value >= SCAT_MIN_CLASS]

scat_filtered_sources, scat_filtered_time_stamps, scat_filtered_second_sources, scat_filtered_third_sources = [], [], [], []
for ms, ss, ts, timestamp in zip(scatter_sources, scatter_second_sources, scatter_third_sources, scatter_time_stamps):
    if ms in allowed_classes and ss in allowed_classes and ts in allowed_classes:
        scat_filtered_sources.append(ms)
        scat_filtered_second_sources.append(ss)
        scat_filtered_third_sources.append(ts)
        scat_filtered_time_stamps.append(timestamp)

# Generate Plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
ax1.bar(filtered_class_counts.keys(), filtered_class_counts.values())
ax1.set_xlabel('Class')
ax1.set_ylabel('Identifications')
ax1.set_title('Class Identifications')
ax1.tick_params(axis='x', rotation=45)

ax2.scatter(scat_filtered_time_stamps, scat_filtered_sources, marker='o', label='Most Likely')
if TOP_THREE:
    ax2.scatter(scat_filtered_time_stamps, scat_filtered_second_sources, marker='x', label='Second Most Likely')
    ax2.scatter(scat_filtered_time_stamps, scat_filtered_third_sources, marker='^', label='Third Most Likely')

ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Source')
ax2.set_title('Sound Sources Over Time')
ax2.legend()
plt.tight_layout()
plt.show()
