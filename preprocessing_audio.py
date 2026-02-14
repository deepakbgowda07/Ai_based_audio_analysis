import librosa
import soundfile as sf
import noisereduce as nr
import numpy as np
import os

base_folder = os.path.join("audio_dataset", "Txt_Wav")

INPUT_FILE = os.path.join(base_folder, "Noisy_speech[26_min].wav")
OUTPUT_FILE = os.path.join(base_folder, "Noisy_speech[26_min]_preprocessed.wav")

TARGET_SR = 16000  # Whisper native sample rate

# -----------------------
# Load Audio
# -----------------------

print("Loading audio...")
audio, sr = librosa.load(INPUT_FILE, sr=None)

# -----------------------
# Convert to Mono
# -----------------------

if len(audio.shape) > 1:
    audio = librosa.to_mono(audio)

# -----------------------
# Resample to 16kHz
# -----------------------

if sr != TARGET_SR:
    audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
    sr = TARGET_SR

# -----------------------
# Normalize Volume
# -----------------------

audio = audio / np.max(np.abs(audio))

# -----------------------
# Light Noise Reduction
# -----------------------

print("Applying light noise reduction...")
audio = nr.reduce_noise(
    y=audio,
    sr=sr,
    prop_decrease=0.8  # don't use 1.0 (too aggressive)
)

# -----------------------
# Save Cleaned Audio
# -----------------------

sf.write(OUTPUT_FILE, audio, sr)

print(f"Preprocessed audio saved as {OUTPUT_FILE}")
