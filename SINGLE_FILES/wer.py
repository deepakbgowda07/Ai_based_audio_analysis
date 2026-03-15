from jiwer import wer, process_words
import re

def normalize(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)   # remove punctuation
    text = re.sub(r"\s+", " ", text)      # normalize spaces
    return text.strip()

# Load files
with open("reference.txt", "r", encoding="utf-8") as f:
    reference = f.read()

with open("output.txt", "r", encoding="utf-8") as f:
    hypothesis = f.read()

# Normalize
reference_norm = normalize(reference)
hypothesis_norm = normalize(hypothesis)

# Compute WER
error = wer(reference_norm, hypothesis_norm)

print(f"\nWER: {error:.4f} ({error*100:.2f}%)\n")

# Detailed breakdown
details = process_words(reference_norm, hypothesis_norm)

print("Substitutions :", details.substitutions)
print("Deletions     :", details.deletions)
print("Insertions    :", details.insertions)
print("Correct words :", details.hits)
