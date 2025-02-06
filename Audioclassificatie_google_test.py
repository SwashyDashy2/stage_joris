import numpy as np
from scipy.io import wavfile
from IPython.display import Audio, display
from mediapipe.tasks import python
from mediapipe.tasks.python.components import containers
from mediapipe.tasks.python import audio
from collections import Counter, defaultdict
import matplotlib.pyplot as plt

# Number of classes to display
n_classes = 10
min_score = 0.10
min_classifications = 0

import os

audio_file_name = r'wavs\Barcelona_2022\segments\L_larger_90\passby_155.wav'
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

# Iterate through clips to display classifications at 975 ms intervals
for timestamp in range(0, int(duration * 1000), interval_ms):
    idx = timestamp // interval_ms
    if idx < len(classification_result_list):
        classification_result = classification_result_list[idx]
        top_category = classification_result.classifications[0].categories[0]
        print(f'Timestamp {timestamp}:')
        for i in range(min(n_classes, len(classification_result.classifications[0].categories))):
            category = classification_result.classifications[0].categories[i]
            if category.score >= min_score:
                print(f'  {i+1}. {category.category_name} ({category.score:.2f})')
                timestamp_predictions.append(category.category_name)
                class_counts[category.category_name] += 1
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
print("This sound is likely:")
for i, (category, probability) in enumerate(sorted_classes[:3], 1):
    print(f"{i}. {category} (Total Probability: {probability:.2f})")

# Plot the histogram
plt.bar(filtered_class_counts.keys(), filtered_class_counts.values())
plt.xlabel('Class')
plt.ylabel('Number of Identifications')
plt.title('Histogram of Number of Identifications for Predicted Classes')
plt.xticks(rotation=45, fontsize=8)
plt.yticks(fontsize=8)
plt.show()
