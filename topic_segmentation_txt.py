import re
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import os
print("Current working directory:", os.getcwd())


# ==========================
# 1️⃣ Load TXT File
# ==========================

INPUT_FILE = "audio_dataset/Txt_Wav/output_clean_audio_1.txt"

OUTPUT_FILE = "audio_dataset/Txt_Wav/segmented_topics.txt"




segments = []

pattern = r"\[(\d{2}:\d{2}:\d{2})\]\s*(.*)"

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for line in f:
        match = re.match(pattern, line.strip())
        if match:
            timestamp = match.group(1)
            text = match.group(2)
            segments.append({
                "timestamp": timestamp,
                "text": text
            })

if len(segments) < 3:
    print("Not enough segments for topic segmentation.")
    exit()

print(f"Loaded {len(segments)} segments.")


# ==========================
# 2️⃣ Load Embedding Model
# ==========================

print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

texts = [seg["text"] for seg in segments]

print("Generating embeddings...")
embeddings = model.encode(texts, show_progress_bar=True)


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


# ==========================
# 4️⃣ Detect Topic Boundaries
# ==========================

threshold = np.percentile(similarities, 25)  #Lower percentile = fewer topics.

boundaries = []

for i, sim in enumerate(similarities):
    if sim < threshold:
        boundaries.append(i + 1)

print("\nDetected topic boundaries at segments:", boundaries)


# ==========================
# 5️⃣ Group Into Topics
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
# 6️⃣ Print Topics
# ==========================

# ==========================
# 6️⃣ Save Topics To File
# ==========================

# ==========================
# 6️⃣ Save Styled Topics To File
# ==========================

import textwrap

with open(OUTPUT_FILE, "w", encoding="utf-8") as out:

    out.write("=" * 60 + "\n")
    out.write(" " * 18 + "TOPIC SEGMENTATION REPORT\n")
    out.write("=" * 60 + "\n\n")

    for idx, topic in enumerate(topics):

        start_time = topic[0]["timestamp"]
        end_time = topic[-1]["timestamp"]
        segment_count = len(topic)

        out.write("🔹 TOPIC " + str(idx + 1) + "\n")
        out.write("-" * 60 + "\n")
        out.write(f"Time Range   : {start_time}  →  {end_time}\n")
        out.write(f"Segment Count: {segment_count}\n\n")
        out.write("Transcript:\n")

        for seg in topic:
            wrapped_text = textwrap.fill(seg["text"], width=80)
            out.write(f"• {wrapped_text}\n")

        out.write("\n" + "-" * 60 + "\n\n")

print(f"\nStyled topic report saved to: {OUTPUT_FILE}")



