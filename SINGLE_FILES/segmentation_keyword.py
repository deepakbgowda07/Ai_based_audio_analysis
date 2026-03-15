import json
import logging

from keybert import KeyBERT
from transformers import pipeline
from textblob import TextBlob

logging.basicConfig(level=logging.INFO)

# ==============================
# File paths
# ==============================

INPUT_FILE = "audio_dataset/segmented/output_clean_1.json"
OUTPUT_FILE = "audio_dataset/segmented/final_episode_output.json"

# ==============================
# Load segments
# ==============================

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    raw_segments = json.load(f)

logging.info(f"Loaded {len(raw_segments)} topic segments")

# ==============================
# Initialize models
# ==============================

kw_model = KeyBERT()

logging.info("Loading summarization model...")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# ==============================
# Helper functions
# ==============================

def time_to_seconds(time_str):
    h, m, s = map(int, time_str.split(":"))
    return h * 3600 + m * 60 + s


def extract_keywords(text, top_n=5):

    keywords = kw_model.extract_keywords(
        text,
        keyphrase_ngram_range=(1, 2),
        stop_words="english",
        top_n=top_n
    )

    return [k[0] for k in keywords]


def generate_summary(text):

    text = text[:1024]

    result = summarizer(
        text,
        max_length=40,
        min_length=15,
        do_sample=False
    )

    return result[0]["summary_text"]


def get_sentiment(text):

    blob = TextBlob(text)

    return round(blob.sentiment.polarity, 3)

# ==============================
# Process segments
# ==============================

processed_segments = []

for idx, segment in enumerate(raw_segments):

    start_sec = time_to_seconds(segment["start_time"])
    end_sec = time_to_seconds(segment["end_time"])
    text = segment["content"]

    processed_segments.append({

        "segment_id": f"seg_{idx+1}",
        "start_time": start_sec,
        "end_time": end_sec,
        "text": text,
        "keywords": extract_keywords(text),
        "summary": generate_summary(text),
        "sentiment_score": get_sentiment(text)

    })

# ==============================
# Episode duration
# ==============================

total_duration = max(
    time_to_seconds(seg["end_time"])
    for seg in raw_segments
)

# ==============================
# Final JSON
# ==============================

final_output = {

    "episode_id": "AI_Hustle_Ep1",
    "duration": total_duration,
    "segments": processed_segments

}

# ==============================
# Save JSON
# ==============================

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(final_output, f, indent=4)

logging.info("Final structured JSON saved.")
logging.info("Episode processing complete.")