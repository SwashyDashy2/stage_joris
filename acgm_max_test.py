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
AUDIO_FILE_PATH = r'wavs\Length Tests\Harley Davidson drive by sound.wav'
# --- Histogram Parameters ---
MIN_SCORE = 0.1
MIN_CLASSIFICATIONS = 0

# --- Scatterplot Parameters ---
MIN_PROBABILITY_GRAPH = 0.1
TOP_THREE = True  # Whether to display only top-1 or top-3 in the scatter plot
SCAT_MIN_CLASS = 3

# --- Audio Classifier Parameters ---
N_CLASSES = 11
FILTER_APPLIED = False
ROOT_NAME = "Animal"  # Livestock, farm animals, working animals

# --- Initialize Paths ---
MODEL_PATH = 'yamnet_classifier.tflite'
JSON_PATH = "Depth_mapping_Mediapipe.json"
INTERVAL_MS = 975 

# Validate Audio File
if not os.path.isfile(AUDIO_FILE_PATH):
    raise FileNotFoundError(f"Audio file not found: {AUDIO_FILE_PATH}")

# Display a short preview of the audio (optional)
sample_rate, audio_data = wavfile.read(AUDIO_FILE_PATH)
audio_data = (audio_data / np.max(np.abs(audio_data)) * np.iinfo(np.int16).max).astype(np.int16)
display(Audio(audio_data[:1], rate=sample_rate, autoplay=False))

def classify_audio(audio_file_path, model_path, n_classes, interval_ms,
                   min_probability_graph, min_score, top_three):
    """
    Loads the audio and the classifier model, classifies the audio, and processes the results.
    
    Returns a dictionary with:
      - sample_rate, audio_data, classification_results, duration
      - time_stamps, sources, second_sources, third_sources, all_top10
      - class_counts (filtered by MIN_CLASSIFICATIONS)
    """
    # Load and normalize audio
    sample_rate, audio_data = wavfile.read(audio_file_path)
    # Normalize audio (scale to int16 range)
    audio_data = (audio_data / np.max(np.abs(audio_data)) * np.iinfo(np.int16).max).astype(np.int16)
    
    # Load model and classify audio
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = audio.AudioClassifierOptions(base_options=base_options, max_results=n_classes)
    with audio.AudioClassifier.create_from_options(options) as classifier:
        audio_clip = containers.AudioData.create_from_array(
            audio_data.astype(float) / np.iinfo(np.int16).max, sample_rate
        )
        classification_results = classifier.classify(audio_clip)
    
    # Compute duration in seconds
    duration = len(audio_data) / sample_rate

    # Process Classification Results
    time_stamps, sources, second_sources, third_sources = [], [], [], []
    all_top10 = []   # To store the top 10 predictions (each as (class, probability)) at every timestep
    class_counts = Counter()
    
    for timestamp in range(0, int(duration * 1000), interval_ms):
        idx = timestamp // interval_ms
        if idx < len(classification_results):
            classifications = classification_results[idx].classifications[0].categories
            # Build a list with the top 10 predictions and their probabilities
            top10 = [(cat.category_name, cat.score) for cat in classifications[:n_classes]]
            all_top10.append(top10)
            
            # Print the top 10 classes and probabilities for the current timestamp
            print(f"\nTimestamp {timestamp / 1000:.3f} seconds:")
            for rank, (cls, prob) in enumerate(top10, 1):
                print(f"  {rank}. {cls}: {prob:.2f}")

            if classifications[0].score >= min_probability_graph:
                time_stamps.append(timestamp / 1000)
                sources.append(classifications[0].category_name)
                second_sources.append(classifications[1].category_name if len(classifications) > 1 else None)
                third_sources.append(classifications[2].category_name if len(classifications) > 2 else None)
            for category in classifications[:n_classes]:
                if category.score >= min_score:
                    class_counts[category.category_name] += 1

    filtered_class_counts = {k: v for k, v in class_counts.items() if v >= MIN_CLASSIFICATIONS}
    
    return {
        'sample_rate': sample_rate,
        'audio_data': audio_data,
        'classification_results': classification_results,
        'duration': duration,
        'time_stamps': time_stamps,
        'sources': sources,
        'second_sources': second_sources,
        'third_sources': third_sources,
        'all_top10': all_top10,
        'class_counts': filtered_class_counts
    }

# Use the classification function
results = classify_audio(AUDIO_FILE_PATH, MODEL_PATH, N_CLASSES, INTERVAL_MS,
                           MIN_PROBABILITY_GRAPH, MIN_SCORE, TOP_THREE)

# ---------------------------------------------------------------------------
# Calculation of maximum probability per class and top 5 display with depth info
# ---------------------------------------------------------------------------
max_probabilities = defaultdict(float)
for timestamp_results in results['all_top10']:
    for cls, prob in timestamp_results:
        # Update the maximum probability encountered for this class
        if prob > max_probabilities[cls]:
            max_probabilities[cls] = prob

# Get the top 5 classifications with the highest maximum probability
top5 = sorted(max_probabilities.items(), key=lambda item: item[1], reverse=True)[:5]

# --- Load and Validate JSON Data and Build Depth Mapping ---
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

# Build a mapping from class name to depth (if available)
depth_map = {entry['name']: entry.get('depth', "N/A") for entry in data}

print("\nTop 5 classifications by maximum probability (with Depth info):")
for cls, max_prob in top5:
    depth = depth_map.get(cls, "N/A")
    print(f"{cls}: {max_prob:.2f} (Depth = {depth})")

# Optional: Plot the top 5 maximum probabilities (annotated with depth)
fig, ax = plt.subplots(figsize=(8, 5))
categories, max_probs = zip(*top5)
ax.bar(categories, max_probs)
ax.set_title("Top 5 Classifications by Maximum Probability")
ax.set_ylabel("Maximum Probability")
plt.xticks(rotation=45)
for i, category in enumerate(categories):
    depth = depth_map.get(category, "N/A")
    ax.text(i, max_probs[i] + 0.01, f"Depth = {depth}", ha='center', va='bottom')
plt.show()

# ---------------------------------------------------------------------------
# Conclusion: Determine the class from the deepest (highest depth) group
# among the top 5 with the highest maximum probability.
# ---------------------------------------------------------------------------
top5_with_depth = []
for cls, max_prob in top5:
    depth = depth_map.get(cls, None)
    try:
        depth_numeric = float(depth)
    except (TypeError, ValueError):
        depth_numeric = None
    if depth_numeric is not None:
        top5_with_depth.append((cls, max_prob, depth_numeric))

if top5_with_depth:
    # Get the maximum depth among the top 5 (i.e., the deepest level)
    max_depth = max(x[2] for x in top5_with_depth)
    # Filter for classes with this maximum depth
    candidates = [x for x in top5_with_depth if x[2] == max_depth]
    # Select the candidate with the highest maximum probability among these
    best_candidate = max(candidates, key=lambda x: x[1])
    best_cls, best_max_prob, best_depth = best_candidate
    print(f"\nConclusion: The sound is likely a {best_cls}. "
          f"Since it is of the deepest level ({int(best_depth)}) and among those, "
          f"it has the highest maximum probability ({best_max_prob:.2f}).")
else:
    print("\nNo depth information available in top 5 classifications.")

# ---------------------------------------------------------------------------
# Filter Data Based on Valid Names (for scatter plot)
# ---------------------------------------------------------------------------
filtered_sources, filtered_time_stamps = [], []
filtered_second_sources, filtered_third_sources = [], []

for ms, ss, ts, timestamp in zip(results['sources'], results['second_sources'], results['third_sources'], results['time_stamps']):
    if ms in valid_names:
        filtered_sources.append(ms)
        filtered_second_sources.append(ss)
        filtered_third_sources.append(ts)
        filtered_time_stamps.append(timestamp)

scatter_sources = filtered_sources if FILTER_APPLIED else results['sources']
scatter_time_stamps = filtered_time_stamps if FILTER_APPLIED else results['time_stamps']
scatter_second_sources = filtered_second_sources if FILTER_APPLIED else results['second_sources']
scatter_third_sources = filtered_third_sources if FILTER_APPLIED else results['third_sources']

# ---------------------------------------------------------------------------
# Generate Plots for Overall Classifications and Scatter Data
# ---------------------------------------------------------------------------
if FILTER_APPLIED:
    histogram_class_counts = {k: v for k, v in results['class_counts'].items() if k in valid_names}
else:
    histogram_class_counts = results['class_counts']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

ax1.bar(histogram_class_counts.keys(), histogram_class_counts.values())
ax1.set_xlabel('Class')
ax1.set_ylabel('Identifications')
ax1.set_title('Class Identifications')
ax1.tick_params(axis='x', rotation=45)

ax2.scatter(scatter_time_stamps, scatter_sources, marker='o', label='Most Likely')
if TOP_THREE:
    ax2.scatter(scatter_time_stamps, scatter_second_sources, marker='x', label='Second Most Likely')
    ax2.scatter(scatter_time_stamps, scatter_third_sources, marker='^', label='Third Most Likely')

ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Source')
ax2.set_title('Sound Sources Over Time')
ax2.legend()

plt.tight_layout()
plt.show()
