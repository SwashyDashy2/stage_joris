import numpy as np
from scipy.io import wavfile
from IPython.display import Audio, display
from mediapipe.tasks import python
from mediapipe.tasks.python.components import containers
from mediapipe.tasks.python import audio
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import json
import os

# Number of classes to display
n_classes = 10
min_score = 0.10
min_classifications = 0
minp_graph = 0.05  # New parameter for minimum probability to include in the line graph
category = []
root_name = "Animal"  # Class to filter for the graph
filter_applied = True  # Set to False to disable filtering
top_three = True # Set to False to disable displaying the second and third most likely sources

#
audio_file_name = r'wavs\Barcelona_2022\full_audio\NRSD_20220404_170050.wav'
if not os.path.isfile(audio_file_name):
    raise FileNotFoundError(f"Audio file not found: {audio_file_name}")
# Read the audio file to get the data and sample rate
sample_rate, audio_data = wavfile.read(audio_file_name)

# Normalize the audio data to fit within the required range
audio_data = audio_data / np.max(np.abs(audio_data))

# Ensure audio data is in the correct format (16-bit PCM)
audio_data = (audio_data * np.iinfo(np.int16).max).astype(np.int16)

# Display the audio file with the correct rate
display(Audio(audio_data[:1], rate=sample_rate, autoplay=False))

# Customize and associate model for Classifier
base_options = python.BaseOptions(model_asset_path='yamnet_classifier.tflite')
options = audio.AudioClassifierOptions(
    base_options=base_options, max_results=n_classes)

# Create classifier, segment audio clips, and classify
with audio.AudioClassifier.create_from_options(options) as classifier:
    sample_rate, wav_data = wavfile.read(audio_file_name)
    audio_clip = containers.AudioData.create_from_array(
        wav_data.astype(float) / np.iinfo(np.int16).max, sample_rate)
    classification_result_list = classifier.classify(audio_clip)

# Calculate the duration of the audio clip in seconds
duration = len(audio_data) / sample_rate

# Interval in milliseconds
interval_ms = 975

# Collect predictions at specified intervals
timestamp_predictions = []
class_counts = Counter()
time_stamps = []
most_likely_sources = []
second_most_likely_sources = []  # New list for second most likely sources
third_most_likely_sources = []  # New list for third most likely sources

# Iterate through clips to display classifications at 975 ms intervals
for timestamp in range(0, int(duration * 1000), interval_ms):
    idx = timestamp // interval_ms
    if idx < len(classification_result_list):
        classification_result = classification_result_list[idx]
        top_category = classification_result.classifications[0].categories[0]
        #print(f'Timestamp {timestamp}:')
        for i in range(min(n_classes, len(classification_result.classifications[0].categories))):
            category = classification_result.classifications[0].categories[i]
            if category.score >= min_score:
                #print(f'  {i+1}. {category.category_name} ({category.score:.2f})')
                timestamp_predictions.append(category.category_name)
                class_counts[category.category_name] += 1
        # Track the most likely, second most likely, and third most likely sources for the line graph
        if top_category.score >= minp_graph:
            time_stamps.append(timestamp / 1000)  # Convert to seconds
            most_likely_sources.append(top_category.category_name)
            # Add second and third most likely sources
            if len(classification_result.classifications[0].categories) > 1:
                second_category = classification_result.classifications[0].categories[1]
                second_most_likely_sources.append(second_category.category_name)
            else:
                second_most_likely_sources.append(None)
            if len(classification_result.classifications[0].categories) > 2:
                third_category = classification_result.classifications[0].categories[2]
                third_most_likely_sources.append(third_category.category_name)
            else:
                third_most_likely_sources.append(None)
    else:
        print(f'Timestamp {timestamp}: No classification result available')

# Filter classes based on min_classifications
filtered_class_counts = {k: v for k, v in class_counts.items() if v >= min_classifications}

# Print the three classes with the highest total probability
class_probabilities = defaultdict(float)
for classification_result in classification_result_list:
    for category in classification_result.classifications[0].categories:
        if category.score >= min_score:
            class_probabilities[category.category_name] += category.score

sorted_classes = sorted(class_probabilities.items(), key=lambda x: x[1], reverse=True)

#print("This sound is likely:")
#for i, (category, probability) in enumerate(sorted_classes[:3], 1):
    #print(f"{i}. {category} (Total Probability: {probability:.2f})")


def get_names_from_child_ids(data, root_name):
    # Find the entry with the given root_name
    def find_entry(name, data):
        for entry in data:
            if entry['name'] == name:
                return entry
        return None

    # Recursive function to gather names
    def gather_names(entry, data):
        names = [entry['name']]
        for child_id in entry['child_ids']:
            # Find the child entry by ID
            child_entry = next((item for item in data if item['id'] == child_id), None)
            if child_entry:
                names.extend(gather_names(child_entry, data))
        return names

    # Start the search with the root_name
    root_entry = find_entry(root_name, data)
    if root_entry:
        return gather_names(root_entry, data)
    else:
        return []

# Load data from the JSON file
with open("Depth_mapping_Mediapipe.json", "r") as file:
    data = json.load(file)

# Check if the root_name is a used name in the data file under the "name" tag
if any(entry['name'] == root_name for entry in data):
    # Example usage
    valid_names = get_names_from_child_ids(data, root_name)
else:
    raise ValueError(f"The root name '{root_name}' is not found in the data.")

# Compare and filter out invalid sources and their corresponding time stamps
filtered_sources = []
filtered_time_stamps = []
filtered_second_sources = []  # New list for filtered second sources
filtered_third_sources = []  # New list for filtered third sources

for ms, ss, ts, timestamp in zip(most_likely_sources, second_most_likely_sources, third_most_likely_sources, time_stamps):
    if ms in valid_names:
        filtered_sources.append(ms)
        filtered_second_sources.append(ss)
        filtered_third_sources.append(ts)
        filtered_time_stamps.append(timestamp)

# Now filtered_sources, filtered_second_sources, filtered_third_sources, and filtered_time_stamps only contain valid entries
print("Filtered Sources:", filtered_sources)
print("Filtered Second Sources:", filtered_second_sources)
print("Filtered Third Sources:", filtered_third_sources)
print("Filtered Time Stamps:", filtered_time_stamps)

# Prepare the data for the scatter plot based on filter_applied
if filter_applied:
    # Use the filtered values for the scatter plot
    scatter_time_stamps = filtered_time_stamps
    scatter_sources = filtered_sources
    scatter_second_sources = filtered_second_sources
    scatter_third_sources = filtered_third_sources
    marker_color = 'green'  # Green marker if filter is applied
else:
    # Use the non-filtered values for the scatter plot
    scatter_time_stamps = time_stamps
    scatter_sources = most_likely_sources
    scatter_second_sources = second_most_likely_sources
    scatter_third_sources = third_most_likely_sources
    marker_color = 'red'  # Red marker if no filter is applied

# Prepare the data for the scatter plot based on top_three
if top_three:
    # Use the top 3 sources for the scatter plot
    scatter_time_stamps = time_stamps
    scatter_sources = most_likely_sources
    scatter_second_sources = second_most_likely_sources
    scatter_third_sources = third_most_likely_sources
else:
    # Use only the most likely source for the scatter plot
    scatter_time_stamps = time_stamps
    scatter_sources = most_likely_sources
    scatter_second_sources = [None] * len(time_stamps)  # Empty for second sources
    scatter_third_sources = [None] * len(time_stamps)  # Empty for third sources

# Create the figure and axis for both the histogram and scatter plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Histogram
ax1.bar(filtered_class_counts.keys(), filtered_class_counts.values())
ax1.set_xlabel('Class')
ax1.set_ylabel('Number of Identifications')
ax1.set_title('Histogram of Number of Identifications for Predicted Classes')
ax1.tick_params(axis='x', rotation=45, labelsize=8)
ax1.tick_params(axis='y', labelsize=8)

# Scatter plot
ax2.scatter(scatter_time_stamps, scatter_sources, marker='o', label='Most Likely')
if top_three:
    ax2.scatter(scatter_time_stamps, scatter_second_sources, marker='x', label='Second Most Likely')
    ax2.scatter(scatter_time_stamps, scatter_third_sources, marker='^', label='Third Most Likely')

# Add squares for the legend (without plotting them on the scatter plot)
filter_marker_color = 'green' if filter_applied else 'red'
top_three_marker_color = 'green' if top_three else 'red'

# Add the filter square to the legend
ax2.plot([], [], marker='s', markersize=10, color=filter_marker_color, label="Filter Applied")
# Add the top three square to the legend
ax2.plot([], [], marker='s', markersize=10, color=top_three_marker_color, label="Top Three Sources")

ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Source')
ax2.set_title('Sound Sources Over Time')
ax2.tick_params(axis='x', labelsize=8)
ax2.tick_params(axis='y', labelsize=8)

# Place the legend outside the plot
ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))

# Tight layout to ensure the plots fit well in the same figure
plt.tight_layout()
plt.show()
