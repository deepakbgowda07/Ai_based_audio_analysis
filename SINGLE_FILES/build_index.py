import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# ==========================
# 1️⃣ Safe Paths
# ==========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

JSON_PATH = os.path.join(
    BASE_DIR,
    "audio_dataset",
    "segmented",
    "segmented_output_Noisy_speech[26_min]_preprocessed.json"
)

INDEX_DIR = os.path.join(BASE_DIR, "index")
os.makedirs(INDEX_DIR, exist_ok=True)

# Extract base name from JSON file
base_name = os.path.splitext(os.path.basename(JSON_PATH))[0]

FAISS_INDEX_PATH = os.path.join(INDEX_DIR, f"{base_name}.index")
TEXT_STORE_PATH = os.path.join(INDEX_DIR, f"{base_name}_texts.npy")

# ==========================
# 2️⃣ Load Topics
# ==========================

with open(JSON_PATH, "r", encoding="utf-8") as f:
    topics = json.load(f)

topic_texts = [t["content"] for t in topics]

print(f"Loaded {len(topic_texts)} topics.")

# ==========================
# 3️⃣ Generate Embeddings
# ==========================

print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Generating topic embeddings...")
embeddings = model.encode(topic_texts, show_progress_bar=True)
embeddings = np.array(embeddings).astype("float32")

# ==========================
# 4️⃣ Build FAISS Index
# ==========================

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# ==========================
# 5️⃣ Save Index
# ==========================

faiss.write_index(index, FAISS_INDEX_PATH)
np.save(TEXT_STORE_PATH, topic_texts)

print("FAISS index built successfully.")
print(f"Index saved to: {FAISS_INDEX_PATH}")
print(f"Topic texts saved to: {TEXT_STORE_PATH}")
