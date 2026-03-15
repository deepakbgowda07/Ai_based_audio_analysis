import json
import re
import os

# ==========================
# 1️⃣ Build Safe Paths
# ==========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_TXT = os.path.join(
    BASE_DIR,
    "audio_dataset",
    "Txt_Wav",
    "segmented_LongReport.txt"
)

OUTPUT_JSON = os.path.join(
    BASE_DIR,
    "audio_dataset",
    "segmented",
    "segmented_LongReport.json"
)

# ==========================
# 2️⃣ Read TXT File
# ==========================

topics = []

with open(INPUT_TXT, "r", encoding="utf-8") as f:
    lines = f.readlines()

current_topic = None
content_lines = []

for line in lines:

    # Detect topic start
    if line.startswith("🔹 TOPIC"):
        if current_topic:
            current_topic["content"] = " ".join(content_lines).strip()
            topics.append(current_topic)
            content_lines = []

        current_topic = {}

    # Extract time range
    elif "Time Range" in line:
        match = re.search(r"(\d{2}:\d{2}:\d{2})\s+→\s+(\d{2}:\d{2}:\d{2})", line)
        if match:
            current_topic["start_time"] = match.group(1)
            current_topic["end_time"] = match.group(2)

    # Extract transcript bullet points
    elif line.startswith("•"):
        clean_line = line.replace("•", "").strip()
        content_lines.append(clean_line)

# Add last topic
if current_topic:
    current_topic["content"] = " ".join(content_lines).strip()
    topics.append(current_topic)

# ==========================
# 3️⃣ Save JSON
# ==========================

with open(OUTPUT_JSON, "w", encoding="utf-8") as jf:
    json.dump(topics, jf, indent=4)

print(f"Converted {len(topics)} topics to JSON.")
print(f"Saved to: {OUTPUT_JSON}")
