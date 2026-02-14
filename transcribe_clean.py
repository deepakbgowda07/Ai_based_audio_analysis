import os
import re
from faster_whisper import WhisperModel

# -------------------------------
# CONFIGURATION
# -------------------------------

base_folder = os.path.join("audio_dataset", "Txt_Wav")

AUDIO_FILE = os.path.join(base_folder, "LongReport[60_min]_preprocessed.wav")
OUTPUT_FILE = os.path.join(base_folder, "output_LongReport[60_min]_preprocessed.txt")

MODEL_SIZE = "medium.en"

FILLER_WORDS = [
    "uh", "um", "you know", "like", "i mean",
    "sort of", "kind of", "basically", "actually"
]

# -------------------------------
# Helper Functions
# -------------------------------

def format_timestamp(seconds):
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hrs:02}:{mins:02}:{secs:02}"

def remove_fillers(text):
    pattern = r"\b(" + "|".join(FILLER_WORDS) + r")\b"
    return re.sub(pattern, "", text, flags=re.IGNORECASE)

def clean_text(text):
    text = remove_fillers(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def split_into_sentences(text):
    return re.split(r'(?<=[.!?]) +', text)

# -------------------------------
# Load Model
# -------------------------------

model = WhisperModel(
    MODEL_SIZE,
    device="cuda",
    compute_type="float16"
)

# -------------------------------
# Transcription
# -------------------------------

segments, info = model.transcribe(
    AUDIO_FILE,
    beam_size=5,
    language="en",
    vad_filter=True
)

# -------------------------------
# Save Output
# -------------------------------

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for segment in segments:
        cleaned = clean_text(segment.text)
        sentences = split_into_sentences(cleaned)

        for sentence in sentences:
            if sentence.strip():
                timestamp = format_timestamp(segment.start)
                f.write(f"[{timestamp}] {sentence.strip()}\n\n")

print(f"Transcription complete. Saved to {OUTPUT_FILE}")
