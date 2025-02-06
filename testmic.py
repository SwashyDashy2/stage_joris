import sounddevice as sd
import numpy as np

sample_rate = 44100  # Match the classification model's expected rate
duration = 3  # Record for 3 seconds
mic_index = 1  # Microphone (change as needed)
speaker_index = 8  # Laptop speakers (change as needed)

# List all audio devices
print("Available audio devices:")
print(sd.query_devices())

# Check and print the names of the selected microphone and speakers
mic_name = sd.query_devices(mic_index)['name']
speaker_name = sd.query_devices(speaker_index)['name']
print(f"Using microphone: {mic_name}")
print(f"Using speakers: {speaker_name}")

# Check if the sample rate is valid for both input and output devices
input_device_info = sd.query_devices(mic_index)
output_device_info = sd.query_devices(speaker_index)

# If the devices do not support 16000 Hz, try a common value like 44100 or 48000 Hz
if input_device_info['default_samplerate'] != sample_rate:
    print(f"Warning: The input device does not support {sample_rate} Hz. Using {input_device_info['default_samplerate']} Hz instead.")
    sample_rate = int(input_device_info['default_samplerate'])

if output_device_info['default_samplerate'] != sample_rate:
    print(f"Warning: The output device does not support {sample_rate} Hz. Using {output_device_info['default_samplerate']} Hz instead.")
    sample_rate = int(output_device_info['default_samplerate'])

# Record audio
print("Recording... Speak now!")
recorded_audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16', device=mic_index)
sd.wait()  # Wait for recording to finish
print("Recording complete!")

# Play back the recorded audio
print("Playing back the recording...")
sd.play(recorded_audio, samplerate=sample_rate, device=speaker_index)
sd.wait()
print("Playback complete!")