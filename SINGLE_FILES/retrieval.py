import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# ==========================
# 1️⃣ Safe Paths
# ==========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INDEX_PATH = os.path.join(BASE_DIR, "index", "output_clean_1.index")
TEXT_PATH = os.path.join(BASE_DIR, "index", "output_clean_1_texts.npy")

# ==========================
# 2️⃣ Load Index + Text
# ==========================

index = faiss.read_index(INDEX_PATH)
topic_texts = np.load(TEXT_PATH, allow_pickle=True)

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Retrieval system ready.")

# ==========================
# 3️⃣ Retrieval Function
# ==========================

def retrieve(query, k=3):
    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")

    distances, indices = index.search(query_embedding, k)

    results = []
    for idx in indices[0]:
        results.append(topic_texts[idx])

    return results
