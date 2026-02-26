import re
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os
import textwrap
import json

print("Current working directory:", os.getcwd())

# ==========================
# 1️⃣ File Paths
# ==========================

INPUT_FILE = "audio_dataset/Txt_Wav/output_clean_2.txt"
OUTPUT_TXT = "audio_dataset/Txt_Wav/segmented_output.txt"
OUTPUT_JSON = "audio_dataset/Txt_Wav/topics_structured.json"

# ==========================
# 2️⃣ Load Transcript
# ==========================

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
# 3️⃣ Context-Aware Embeddings (Sliding Window)
# ==========================

print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

window_size = 3

window_texts = [
    " ".join(
        segments[j]["text"]
        for j in range(i, min(i + window_size, len(segments)))
    )
    for i in range(len(segments))
]

print("Generating embeddings...")
embeddings = model.encode(window_texts, show_progress_bar=True)

# ==========================
# 4️⃣ Compute Similarities
# ==========================

similarities = [
    cosine_similarity([embeddings[i]], [embeddings[i + 1]])[0][0]
    for i in range(len(embeddings) - 1)
]

similarities = np.array(similarities)

# ==========================
# 5️⃣ Smooth Similarities
# ==========================

def moving_average(data, k=3):
    return np.convolve(data, np.ones(k) / k, mode='same')

smoothed_similarities = moving_average(similarities, k=3)

# ==========================
# 6️⃣ Detect Topic Boundaries
# ==========================

threshold = np.percentile(smoothed_similarities, 20)

raw_boundaries = [
    i + 1 for i, sim in enumerate(smoothed_similarities)
    if sim < threshold
]

# ==========================
# 7️⃣ Enforce Minimum Topic Length
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
# 8️⃣ Group Into Topics
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

print(f"Total topics created: {len(topics)}")

# ==========================
# 9️⃣ Save Styled TXT Report
# ==========================

with open(OUTPUT_TXT, "w", encoding="utf-8") as out:

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

print(f"Styled topic report saved to: {OUTPUT_TXT}")

# ==========================
# 🔟 Save Structured JSON (For FAISS Indexing)
# ==========================

structured_topics = [
    {
        "start_time": topic[0]["timestamp"],
        "end_time": topic[-1]["timestamp"],
        "content": " ".join([seg["text"] for seg in topic])
    }
    for topic in topics
]

with open(OUTPUT_JSON, "w", encoding="utf-8") as jf:
    json.dump(structured_topics, jf, indent=4)

print(f"Structured topics saved to: {OUTPUT_JSON}")

print("\nTopic segmentation complete.")