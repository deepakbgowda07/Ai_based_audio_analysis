import re
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os
import textwrap

print("Current working directory:", os.getcwd())

# ==========================
# 1️⃣ Load TXT File
# ==========================

INPUT_FILE = "audio_dataset/Txt_Wav/output_Noisy_speech[26_min]_preprocessed.txt"
OUTPUT_FILE = "audio_dataset/Txt_Wav/segmented_output_Noisy_speech[26_min]_preprocessed.txt"

segments = []
pattern = r"\[(\d{2}:\d{2}:\d{2})\]\s*(.*)"

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for line in f:
        match = re.match(pattern, line.strip())
        if match:
            segments.append({
                "timestamp": match.group(1),
                "text": match.group(2)
            })

if len(segments) < 5:
    print("Not enough segments for topic segmentation.")
    exit()

print(f"Loaded {len(segments)} segments.")

# ==========================
# 2️⃣ Context-Aware Embeddings (Sliding Window)
# ==========================

print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

window_size = 3

window_texts = []
for i in range(len(segments)):
    window = " ".join(
        segments[j]["text"]
        for j in range(i, min(i + window_size, len(segments)))
    )
    window_texts.append(window)

print("Generating embeddings...")
embeddings = model.encode(window_texts, show_progress_bar=True)

# ==========================
# 3️⃣ Compute Similarities
# ==========================

similarities = []
for i in range(len(embeddings) - 1):
    sim = cosine_similarity(
        [embeddings[i]],
        [embeddings[i + 1]]
    )[0][0]
    similarities.append(sim)

similarities = np.array(similarities)

# ==========================
# 4️⃣ Smooth Similarities (Moving Average)
# ==========================

def moving_average(data, k=3):
    return np.convolve(data, np.ones(k)/k, mode='same')

smoothed_similarities = moving_average(similarities, k=3)

# ==========================
# 5️⃣ Detect Topic Boundaries
# ==========================

threshold = np.percentile(smoothed_similarities, 20)

raw_boundaries = [
    i + 1 for i, sim in enumerate(smoothed_similarities)
    if sim < threshold
]

# ==========================
# 6️⃣ Enforce Minimum Topic Length
# ==========================

min_topic_size = 3
boundaries = []
last_boundary = 0

for b in raw_boundaries:
    if b - last_boundary >= min_topic_size:
        boundaries.append(b)
        last_boundary = b

print("\nDetected topic boundaries:", boundaries)

# ==========================
# 7️⃣ Group Into Topics
# ==========================

topics = []
current_topic = []
boundary_set = set(boundaries)

for i, seg in enumerate(segments):
    if i in boundary_set and current_topic:
        topics.append(current_topic)
        current_topic = []
    current_topic.append(seg)

if current_topic:
    topics.append(current_topic)

# ==========================
# 8️⃣ Save Styled Topics
# ==========================

with open(OUTPUT_FILE, "w", encoding="utf-8") as out:

    out.write("=" * 60 + "\n")
    out.write(" " * 18 + "TOPIC SEGMENTATION REPORT\n")
    out.write("=" * 60 + "\n\n")

    for idx, topic in enumerate(topics):

        start_time = topic[0]["timestamp"]
        end_time = topic[-1]["timestamp"]
        segment_count = len(topic)

        out.write(f"🔹 TOPIC {idx + 1}\n")
        out.write("-" * 60 + "\n")
        out.write(f"Time Range   : {start_time}  →  {end_time}\n")
        out.write(f"Segment Count: {segment_count}\n\n")
        out.write("Transcript:\n")

        for seg in topic:
            wrapped = textwrap.fill(seg["text"], width=80)
            out.write(f"• {wrapped}\n")

        out.write("\n" + "-" * 60 + "\n\n")

print(f"\nImproved topic report saved to: {OUTPUT_FILE}")
