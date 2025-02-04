import numpy as np
from scipy.io import wavfile
from IPython.display import Audio, display
from mediapipe.tasks import python
from mediapipe.tasks.python.components import containers
from mediapipe.tasks.python import audio
from collections import Counter
import matplotlib.pyplot as plt

# Number of classes to display
n_classes = 5

audio_file_name = r'wavs\hond2.wav'
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

# Iterate through clips to display classifications at 975 ms intervals
for timestamp in range(0, int(duration * 1000), interval_ms):
    idx = timestamp // interval_ms
    if idx < len(classification_result_list):
        classification_result = classification_result_list[idx]
        top_category = classification_result.classifications[0].categories[0]
        print(f'Timestamp {timestamp}:')
        for i in range(min(n_classes, len(classification_result.classifications[0].categories))):
            category = classification_result.classifications[0].categories[i]
            if category.score >= 0.2:
                print(f'  {i+1}. {category.category_name} ({category.score:.2f})')
                timestamp_predictions.append(category.category_name)
    else:
        print(f'Timestamp {timestamp}: No classification result available')

# Count the occurrences of each class at the specified timestamps
prediction_counts = Counter(timestamp_predictions)

# Plot the histogram
plt.bar(prediction_counts.keys(), prediction_counts.values())
plt.xlabel('Class')
plt.ylabel('Frequency')
plt.title('Histogram of Predicted Classes at Specified Timestamps')
plt.xticks(rotation=45, fontsize=8)
plt.yticks(fontsize=8)
plt.show()
