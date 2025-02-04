from transformers import AutoProcessor, BeatsForAudioClassification
import torch
import torchaudio

# Step 1: Load the BEATs model and processor
model_name = "facebook/beat-large-960h"
model = BeatsForAudioClassification.from_pretrained(model_name)
processor = AutoProcessor.from_pretrained(model_name)

# Step 2: Load an audio file
audio_file_path = "hond2.wav"  # Replace with your audio file path
audio_input, _ = torchaudio.load(audio_file_path)

# Step 3: Preprocess the audio
inputs = processor(audio_input, sampling_rate=16000, return_tensors="pt")

# Step 4: Run the model and get predictions
with torch.no_grad():
    logits = model(**inputs).logits

# Step 5: Get the predicted class index
predicted_class_idx = logits.argmax(-1).item()

# Step 6: Print the predicted class index
print(f"Predicted class index: {predicted_class_idx}")
