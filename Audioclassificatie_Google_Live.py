import numpy as np
import os
import random
import string
import sounddevice as sd
import scipy.io.wavfile as wavfile
from mediapipe.tasks import python
from mediapipe.tasks.python import audio
from mediapipe.tasks.python.components import containers
from collections import Counter, defaultdict
import time
import threading

# Number of classes to display
n_classes = 10
min_score = 0.0
min_classifications = 10
duration_ms = 975  # Duration of each segment to record in ms
sample_rate = 16000  # Set sample rate
mic_index = 1  # Microphone

# Flag to control the loop
is_running = True

# Function to generate a random filename
def generate_random_filename():
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=10)) + '.wav'

# Initialize the audio classifier
base_options = python.BaseOptions(model_asset_path='yamnet_classifier.tflite')
options = audio.AudioClassifierOptions(base_options=base_options, max_results=n_classes)

# Create the classifier
classifier = audio.AudioClassifier.create_from_options(options)

# Function to record audio
def record_audio(duration_ms):
    duration_sec = duration_ms / 1000  # Convert to seconds
    print("Recording...")
    audio_data = sd.rec(int(duration_sec * sample_rate), samplerate=sample_rate, channels=1, dtype='int16', device=mic_index)
    sd.wait()  # Wait until the recording is finished
    
    return audio_data.flatten()

# Function to classify audio
def classify_audio(audio_data):
    # Save the audio to a random file
    audio_filename = os.path.join('wavs', generate_random_filename())
    wavfile.write(audio_filename, sample_rate, audio_data)
    print(f"Saved file as {audio_filename}")

    # Prepare audio data for classification
    audio_clip = containers.AudioData.create_from_array(audio_data.astype(float) / np.iinfo(np.int16).max, sample_rate)
    classification_result_list = classifier.classify(audio_clip)

    # Process the classification results
    class_counts = Counter()
    timestamp_predictions = []

    for classification_result in classification_result_list:
        print("Classification result:")
        for category in classification_result.classifications[0].categories:
            if category.score >= min_score:
                print(f'  {category.category_name} ({category.score:.2f})')
                timestamp_predictions.append(category.category_name)
                class_counts[category.category_name] += 1

    # Filter classes based on minimum classifications
    filtered_class_counts = {k: v for k, v in class_counts.items() if v >= min_classifications}

    # Print the three classes with the highest total probability
    class_probabilities = defaultdict(float)
    for classification_result in classification_result_list:
        for category in classification_result.classifications[0].categories:
            if category.score >= min_score:
                class_probabilities[category.category_name] += category.score

    sorted_classes = sorted(class_probabilities.items(), key=lambda x: x[1], reverse=True)

    # Delete the audio file after classification
    os.remove(audio_filename)
    print(f"Deleted file {audio_filename}")

# Continuous audio recording and classification
def continuous_classification():
    global is_running
    while is_running:
        # Record audio for 975 ms
        audio_data = record_audio(duration_ms)
        
        # Classify the recorded audio
        classify_audio(audio_data)

        # Delay before recording the next segment
        time.sleep(0.1)  # Adjust if necessary

# Function to stop the classification loop
def stop_classification():
    global is_running
    input("Press Enter to stop the classification...\n")
    is_running = False

# Start classification in a separate thread
classification_thread = threading.Thread(target=continuous_classification)
classification_thread.start()

# Wait for user to stop the classification
stop_classification()

# Wait for the classification thread to finish
classification_thread.join()

print("Classification stopped.")
