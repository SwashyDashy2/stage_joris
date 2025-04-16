import pyaudio
import wave
import time

# ==== CONFIG ====
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100  # CD-quality
CHUNK = 1024
DEVICE_INDEX = 1
RECORD_SECONDS = 10 * 60  # 10 minutes
OUTPUT_FILENAME = "recording_10min.wav"

# ==== INITIALIZE ====
p = pyaudio.PyAudio()

print("Starting recording for 10 minutes...")

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                input_device_index=DEVICE_INDEX,
                frames_per_buffer=CHUNK)

frames = []

# ==== RECORD ====
for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK, exception_on_overflow=False)
    frames.append(data)

# ==== STOP & SAVE ====
print("Recording finished, saving file...")

stream.stop_stream()
stream.close()
p.terminate()

with wave.open(OUTPUT_FILENAME, 'wb') as wf:
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))

print(f"Saved as: {OUTPUT_FILENAME}")
