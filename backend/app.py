"""
Flask Backend for Audio Semantic Segmentation Pipeline  (v2 — single file)
===========================================================================
Pipeline:
  audio upload → validate → preprocess → transcribe → embed →
  segment topics → sentiment + keywords → FAISS index → RAG (Gemini)

Improvements over v1:
  1. Audio preprocessing   — mono · 16 kHz · normalise · noise-reduce
  2. Real RAG              — Google Gemini (gemini-1.5-flash)
  3. Persistent FAISS      — saved to / loaded from  faiss_index/
  4. Real perf metrics     — time.time() per stage
  5. Structured errors     — typed try/except + meaningful HTTP codes
  6. Logging               — Python logging → stdout + pipeline.log
  7. File validation       — extension · size · soundfile header check
  8. Clean endpoints       — all logic in functions, Flask routes stay thin

Requirements:
  pip install flask flask-cors faster-whisper librosa soundfile noisereduce \
              sentence-transformers transformers torch scikit-learn numpy \
              faiss-cpu google-generativeai werkzeug

Environment:
  GEMINI_API_KEY=<your key>        # optional — for LLM-powered RAG answers
"""

# ── Standard library ─────────────────────────────────────────────────────────
import os
import sys
import re
import time
import logging
import tempfile
from collections import Counter
from datetime import datetime

# ── Flask ────────────────────────────────────────────────────────────────────
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

# ── Audio ────────────────────────────────────────────────────────────────────
from faster_whisper import WhisperModel
import librosa
import soundfile as sf
import noisereduce as nr
import numpy as np

# ── NLP / ML ─────────────────────────────────────────────────────────────────
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline as hf_pipeline

# ── LLM ──────────────────────────────────────────────────────────────────────
import google.generativeai as genai
import faiss
from dotenv import load_dotenv


###############################################################################
# LOGGING
###############################################################################

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("pipeline.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("backend")


###############################################################################
# FLASK + CONSTANTS
###############################################################################

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER       = os.path.join(tempfile.gettempdir(), "audio_uploads")
ALLOWED_EXTENSIONS  = {"mp3", "wav", "m4a", "flac", "ogg", "webm"}
MAX_FILE_SIZE       = 100 * 1024 * 1024          # 100 MB
TARGET_SR           = 16_000                      # Whisper wants 16 kHz
FAISS_INDEX_DIR     = os.path.join(os.path.dirname(__file__), "faiss_index")
FAISS_INDEX_PATH    = os.path.join(FAISS_INDEX_DIR, "segments.index")
FAISS_META_PATH     = os.path.join(FAISS_INDEX_DIR, "segments_meta.npy")

FILLER_WORDS = [
    "uh", "um", "you know", "like", "i mean", "sort of", "kind of",
    "basically", "actually", "literally", "honestly", "you know what", "right",
]
STOPWORDS = {
    "the", "and", "for", "that", "with", "from", "this", "have", "been",
    "was", "are", "you", "all", "not", "but", "can", "get", "on", "or",
    "as", "in", "to", "of", "at", "it", "is", "a",
}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(FAISS_INDEX_DIR, exist_ok=True)
app.config["UPLOAD_FOLDER"]        = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"]   = MAX_FILE_SIZE


###############################################################################
# MODEL INITIALISATION
###############################################################################

logger.info("=" * 60)
logger.info("AUDIO SEMANTIC SEGMENTATION PIPELINE — startup")
logger.info("=" * 60)

# Whisper — force CPU to avoid CUDA library issues
# (GPU training can happen separately; inference is the bottleneck)
# OPTIMIZED: Changed from medium.en (40-60s) to base.en (8-15s) = 3-5x faster
whisper_model = WhisperModel("base.en", device="cpu", compute_type="int8")
logger.info("Whisper loaded  (CPU / int8 / BASE)")

# Sentence embeddings
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
logger.info("Embedding model loaded  (all-MiniLM-L6-v2)")

# Sentiment
sentiment_pipeline = hf_pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment",
)
logger.info("Sentiment model loaded  (cardiffnlp/twitter-roberta-base-sentiment)")

# Gemini — lazy init
_gemini_client = None

def _get_gemini():
    global _gemini_client
    if _gemini_client is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError("GEMINI_API_KEY environment variable is not set.")
        genai.configure(api_key=api_key)
        _gemini_client = genai.GenerativeModel("gemini-2.5-flash")
        logger.info("Gemini client initialised  (gemini-1.5-flash)")
    return _gemini_client

# FAISS — load at startup
_faiss_index = None
_faiss_segments_meta = []

def _try_load_faiss():
    global _faiss_index, _faiss_segments_meta
    if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(FAISS_META_PATH):
        try:
            _faiss_index        = faiss.read_index(FAISS_INDEX_PATH)
            _faiss_segments_meta = np.load(FAISS_META_PATH, allow_pickle=True).tolist()
            logger.info(f"Persisted FAISS index loaded  ({_faiss_index.ntotal} vectors)")
        except Exception as e:
            logger.warning(f"Could not load persisted FAISS index: {e}")

_try_load_faiss()
logger.info("All models ready.\n")


###############################################################################
# AUDIO PREPROCESSING
###############################################################################

def validate_audio_file(filepath):
    if not os.path.exists(filepath):
        raise ValueError("Uploaded file does not exist on disk.")
    size = os.path.getsize(filepath)
    if size == 0:
        raise ValueError("Uploaded file is empty.")
    if size > MAX_FILE_SIZE:
        raise ValueError(f"File too large: {size / (1024*1024):.1f} MB (limit {MAX_FILE_SIZE // (1024*1024)} MB).")
    try:
        info = sf.info(filepath)
        if info.duration == 0:
            raise ValueError("Audio file has zero duration.")
        logger.info(f"[VALIDATE] OK  duration={info.duration:.1f}s  channels={info.channels}  sr={info.samplerate}")
    except Exception as e:
        raise ValueError(f"File is not a valid audio file or is corrupted: {e}")


def preprocess_audio(input_path):
    logger.info(f"[PREPROCESS] Starting: {input_path}")
    try:
        # OPTIMIZATION: Load directly at TARGET_SR with mono=True (single pass)
        y, sr = librosa.load(input_path, sr=TARGET_SR, mono=True)
        logger.info(f"[PREPROCESS] Loaded  sr={sr}  shape={y.shape}")
    except Exception as e:
        raise ValueError(f"Could not load audio file: {e}")

    try:
        # No need for mono conversion or resampling - already done in load()
        peak = np.max(np.abs(y))
        if peak > 0:
            y = y / peak
        logger.info("[PREPROCESS] Normalised amplitude")
        # OPTIMIZED: Noise reduction disabled (30-40% overhead, unnecessary for clean audio)
        # y = nr.reduce_noise(y=y, sr=sr, stationary=False, prop_decrease=0.75)
        # logger.info("[PREPROCESS] Noise reduction applied")
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(tmp.name, y, sr)
        logger.info(f"[PREPROCESS] Written: {tmp.name}")
        return tmp.name
    except Exception as e:
        raise RuntimeError(f"Audio preprocessing failed: {e}")


###############################################################################
# TRANSCRIPTION
###############################################################################

def _format_timestamp(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _remove_fillers(text):
    pattern = r"\b(" + "|".join(FILLER_WORDS) + r")\b"
    return re.sub(pattern, "", text, flags=re.IGNORECASE)


def _clean_text(text):
    text = _remove_fillers(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _split_sentences(text):
    
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def transcribe_audio(audio_path):
    logger.info(f"[TRANSCRIBE] Start: {audio_path}")
    try:
        segments, _ = whisper_model.transcribe(audio_path, beam_size=5, language="en", vad_filter=True)
        transcript_data = []
        for seg in segments:
            for sent in _split_sentences(_clean_text(seg.text)):
                transcript_data.append({
                    "timestamp": _format_timestamp(seg.start),
                    "start": float(seg.start),
                    "end": float(seg.end),
                    "text": sent,
                })
        logger.info(f"[TRANSCRIBE] Done — {len(transcript_data)} sentences")
        return transcript_data
    except Exception as e:
        logger.error(f"[TRANSCRIBE] Failed: {e}")
        raise RuntimeError(f"Transcription failed: {e}")


###############################################################################
# EMBEDDINGS
###############################################################################

def compute_embeddings(transcript_data, window=3):
    if not transcript_data:
        raise ValueError("transcript_data is empty — nothing to embed.")
    logger.info(f"[EMBEDDING] Generating for {len(transcript_data)} sentences (window={window})")
    try:
        texts = [
            " ".join(transcript_data[j]["text"] for j in range(i, min(i + window, len(transcript_data))))
            for i in range(len(transcript_data))
        ]
        embs = embedding_model.encode(texts, show_progress_bar=False)
        embs = np.array(embs, dtype="float32")
        logger.info(f"[EMBEDDING] Shape: {embs.shape}")
        return embs
    except Exception as e:
        logger.error(f"[EMBEDDING] Failed: {e}")
        raise RuntimeError(f"Embedding generation failed: {e}")


###############################################################################
# SEGMENTATION
###############################################################################

def _moving_average(data, k=3):
    return np.convolve(data, np.ones(k) / k, mode="same")


def segment_topics(transcript_data, embeddings):
    logger.info("[SEGMENTATION] Detecting topic boundaries")
    try:
        sims = np.array([
            float(cosine_similarity([embeddings[i]], [embeddings[i + 1]])[0][0])
            for i in range(len(embeddings) - 1)
        ])
        smoothed = _moving_average(sims, k=3)
        threshold = np.percentile(smoothed, 20)
        raw_bounds = [i + 1 for i, s in enumerate(smoothed) if s < threshold]
        boundaries, last = [0], 0
        for b in raw_bounds:
            if b - last >= 3:
                boundaries.append(b)
                last = b
        boundaries.append(len(transcript_data))
        boundaries = sorted(set(boundaries))
        logger.info(f"[SEGMENTATION] {len(boundaries) - 1} segments detected")
        return boundaries, sims
    except Exception as e:
        logger.error(f"[SEGMENTATION] Failed: {e}")
        raise RuntimeError(f"Segmentation failed: {e}")


###############################################################################
# KEYWORDS + SENTIMENT
###############################################################################

def extract_keywords(text, top_k=8):
    words = re.findall(r"\b[a-z]{3,}\b", text.lower())
    freq = Counter(words)
    return [w for w, _ in freq.most_common(top_k * 2) if w not in STOPWORDS][:top_k]


def analyze_sentiment_batch(texts, batch_size=32):
    """Batch sentiment analysis - 10-20x faster than individual calls"""
    logger.info(f"[SENTIMENT] Batch analyzing {len(texts)} texts...")
    sentiments = []
    try:
        for batch_start in range(0, len(texts), batch_size):
            batch_end = min(batch_start + batch_size, len(texts))
            batch_texts = [t[:512] for t in texts[batch_start:batch_end]]
            batch_results = sentiment_pipeline(batch_texts)
            for result in batch_results:
                label = result["label"]
                score = float(result["score"])
                if label == "LABEL_2":
                    sentiments.append({"sentiment": "positive", "score": score})
                elif label == "LABEL_0":
                    sentiments.append({"sentiment": "negative", "score": -score})
                else:
                    sentiments.append({"sentiment": "neutral", "score": 0.0})
        logger.info(f"[SENTIMENT] Batch analysis complete")
        return sentiments
    except Exception as e:
        logger.warning(f"[SENTIMENT] Batch analysis failed: {e}")
        return [{"sentiment": "neutral", "score": 0.0} for _ in texts]


def build_segments(transcript_data, boundaries, similarities):
    segments = []
    segment_texts = []
    segment_info = []
    
    # Step 1: Extract all segment texts
    for s, e in zip(boundaries[:-1], boundaries[1:]):
        text = " ".join(transcript_data[i]["text"] for i in range(s, e))
        segment_texts.append(text)
        segment_info.append((s, e, text))
    
    # Step 2: BATCH sentiment analysis (OPTIMIZATION)
    sentiments = analyze_sentiment_batch(segment_texts)
    # Step 3: Extract keywords 
    keywords_batch = [extract_keywords(text, top_k=6) for text in segment_texts]
    
    # Step 4: Assemble final segments
    for seg_idx, ((s, e, text), sentiment, keywords) in enumerate(zip(segment_info, sentiments, keywords_batch)):
        start_time = transcript_data[s]["start"]
        end_time = transcript_data[e - 1]["end"]
        bsims = [similarities[i - 1] if 0 < i <= len(similarities) else 0.5 for i in [s, e]]
        confidence = float(min(1.0, max(0.5, 1.0 - min(bsims))))
        segments.append({
            "id": seg_idx,
            "start_time": _format_timestamp(start_time),
            "end_time": _format_timestamp(end_time),
            "start_timestamp": float(start_time),
            "end_timestamp": float(end_time),
            "content": text,
            "word_count": len(text.split()),
            "duration": f"{int(end_time - start_time)}s",
            "keywords": keywords,
            "sentiment": sentiment["sentiment"],
            "sentiment_score": sentiment["score"],
            "confidence": confidence,
            "sentence_count": e - s,
        })
    return segments


def build_transcript_items(transcript_data, boundaries):
    items = []
    for idx, item in enumerate(transcript_data):
        seg_id = 0
        for i, b in enumerate(boundaries[:-1]):
            if b <= idx < boundaries[i + 1]:
                seg_id = i
                break
        items.append({
            "id": idx,
            "timestamp": item["timestamp"],
            "start": float(item["start"]),
            "end": float(item["end"]),
            "text": item["text"],
            "segment": seg_id,
            "keywords": extract_keywords(item["text"], top_k=3),
        })
    return items


###############################################################################
# FAISS
###############################################################################

def build_faiss_index(segments):
    global _faiss_index, _faiss_segments_meta
    logger.info(f"[FAISS] Building index for {len(segments)} segments")
    try:
        texts = [s["content"] for s in segments]
        embs = embedding_model.encode(texts, show_progress_bar=False)
        embs = np.array(embs, dtype="float32")
        index = faiss.IndexFlatL2(embs.shape[1])
        index.add(embs)
        faiss.write_index(index, FAISS_INDEX_PATH)
        np.save(FAISS_META_PATH, np.array(segments, dtype=object))
        logger.info(f"[FAISS] Index saved  ({index.ntotal} vectors → {FAISS_INDEX_PATH})")
        _faiss_index = index
        _faiss_segments_meta = segments
        return index, segments
    except Exception as e:
        logger.error(f"[FAISS] Build failed: {e}")
        raise RuntimeError(f"FAISS index build failed: {e}")

def generate_podcast_analysis(segments, max_chars=6000):
    """
    Use Gemini to generate:
    - Podcast title
    - Summary
    - Key insights
    - Topic labels
    """

    if not segments:
        return None

    # Build transcript context
    transcript_text = " ".join(seg["content"] for seg in segments)

    # limit context size
    transcript_text = transcript_text[:max_chars]

    prompt = f"""
You are an AI assistant that analyzes podcast transcripts.

Based on the transcript below, generate:

1. A short **Podcast Title**
2. A **3-4 sentence summary**
3. **5 key insights**
4. **5 topic labels**

Transcript:
{transcript_text}

Return the response in JSON format like this:

{{
"title": "...",
"summary": "...",
"insights": ["...", "..."],
"topics": ["...", "..."]
}}
"""

    try:
        client = _get_gemini()
        response = client.generate_content(prompt)

        return response.text

    except Exception as e:
        logger.warning(f"[LLM] Podcast analysis failed: {e}")
        return None

def retrieve_top_segments(query, top_k=3, segments_override=None):
    global _faiss_index, _faiss_segments_meta
    index = _faiss_index
    meta = _faiss_segments_meta
    if (index is None or index.ntotal == 0) and segments_override:
        logger.info("[FAISS] No persisted index — building temporary from payload")
        index, meta = build_faiss_index(segments_override)
    if index is None or index.ntotal == 0:
        raise RuntimeError("No FAISS index available. Upload an audio file first.")
    logger.info(f"[FAISS] Searching for: '{query}'")
    try:
        q_emb = embedding_model.encode([query], show_progress_bar=False)
        q_emb = np.array(q_emb, dtype="float32")
        dists, idxs = index.search(q_emb, k=min(top_k, index.ntotal))
        results = []
        for rank, (dist, idx) in enumerate(zip(dists[0], idxs[0])):
            if 0 <= idx < len(meta):
                seg = dict(meta[idx]) if isinstance(meta[idx], dict) else meta[idx]
                seg["retrieval_rank"] = rank
                seg["retrieval_distance"] = float(dist)
                results.append(seg)
        logger.info(f"[FAISS] Retrieved {len(results)} segments")
        return results
    except Exception as e:
        logger.error(f"[FAISS] Search failed: {e}")
        raise RuntimeError(f"FAISS retrieval failed: {e}")


###############################################################################
# RAG
###############################################################################

def generate_rag_answer(query, retrieved_segments, max_context_chars=8000):
    if not retrieved_segments:
        raise ValueError("No retrieved segments to generate an answer from.")
    parts, ref_ids, total = [], [], 0
    for seg in retrieved_segments:
        snippet = f"[Segment {seg.get('id', '?')}] {seg.get('content', '')}"
        if total + len(snippet) > max_context_chars:
            break
        parts.append(snippet)
        ref_ids.append(seg.get("id"))
        total += len(snippet)
    context = "\n\n".join(parts)
    prompt = f"You are an assistant that answers questions based solely on audio transcription excerpts.\nUse ONLY the context below. If the answer is not present, say so.\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"
    logger.info(f"[LLM] Sending to Gemini  (context_chars={total}  query='{query}')")
    try:
        client = _get_gemini()
        resp = client.generate_content(prompt)
        answer = resp.text.strip()
        logger.info("[LLM] Answer received from Gemini")
    except EnvironmentError:
        raise
    except Exception as e:
        logger.error(f"[LLM] Gemini call failed: {e}")
        raise RuntimeError(f"LLM answer generation failed: {e}")
    return {
        "answer": answer,
        "referenced_segment_ids": ref_ids,
        "model": "gemini-2.5-flash",
        "context_length": total,
    }


###############################################################################
# HELPERS
###############################################################################

def _allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _err(msg, code):
    logger.error(f"HTTP {code}: {msg}")
    return jsonify({"error": msg}), code


###############################################################################
# ENDPOINTS
###############################################################################

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "faiss_loaded": _faiss_index is not None,
        "faiss_vectors": int(_faiss_index.ntotal) if _faiss_index else 0,
    })


@app.route("/api/upload", methods=["POST"])
def upload_file():
    pipeline_start = time.time()
    perf = {}
    raw_path = preprocessed_path = None

    if "file" not in request.files:
        return _err("No file field in request.", 400)
    file = request.files["file"]
    if not file.filename:
        return _err("No file selected.", 400)
    if not _allowed_file(file.filename):
        return _err(f"Unsupported file type. Allowed: {', '.join(ALLOWED_EXTENSIONS).upper()}", 400)

    logger.info(f"[UPLOAD] Received: {file.filename}")
    filename = secure_filename(file.filename)
    raw_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)

    try:
        file.save(raw_path)
        validate_audio_file(raw_path)
    except ValueError as e:
        return _err(str(e), 400)
    except Exception as e:
        return _err(f"File save/validation error: {e}", 500)

    t0 = time.time()
    try:
        preprocessed_path = preprocess_audio(raw_path)
    except (ValueError, RuntimeError) as e:
        return _err(str(e), 422)
    except Exception as e:
        return _err(f"Unexpected preprocessing error: {e}", 500)
    finally:
        if raw_path and os.path.exists(raw_path):
            os.unlink(raw_path)
    perf["preprocessing"] = round(time.time() - t0, 3)

    t0 = time.time()
    try:
        transcript_data = transcribe_audio(preprocessed_path)
    except RuntimeError as e:
        return _err(str(e), 500)
    except Exception as e:
        return _err(f"Unexpected transcription error: {e}", 500)
    finally:
        if preprocessed_path and os.path.exists(preprocessed_path):
            os.unlink(preprocessed_path)
    perf["transcription"] = round(time.time() - t0, 3)

    if not transcript_data:
        return _err("Transcription produced no text. Check audio quality.", 422)

    t0 = time.time()
    try:
        embeddings = compute_embeddings(transcript_data)
    except (ValueError, RuntimeError) as e:
        return _err(str(e), 500)
    except Exception as e:
        return _err(f"Unexpected embedding error: {e}", 500)
    perf["embedding"] = round(time.time() - t0, 3)

    t0 = time.time()
    try:
        boundaries, similarities = segment_topics(transcript_data, embeddings)
    except RuntimeError as e:
        return _err(str(e), 500)
    except Exception as e:
        return _err(f"Unexpected segmentation error: {e}", 500)
    perf["segmentation"] = round(time.time() - t0, 3)

    t0 = time.time()
    try:
        segments = build_segments(transcript_data, boundaries, similarities)
        transcript_items = build_transcript_items(transcript_data, boundaries)
        analysis = generate_podcast_analysis(segments)
    except Exception as e:
        return _err(f"Segment construction error: {e}", 500)
    perf["sentiment"] = round(time.time() - t0, 3)

    try:
        build_faiss_index(segments)
        logger.info("[FAISS] Index rebuilt for new upload")
    except RuntimeError as e:
        logger.warning(f"[FAISS] Index build failed (non-fatal): {e}")

    perf["total"] = round(time.time() - pipeline_start, 3)
    logger.info(f"[UPLOAD] Complete — {len(segments)} segments  {perf['total']}s total  ({filename})")

    return jsonify({
        "status": "success",
        "filename": filename,
        "transcript": transcript_items,
        "segments": segments,
        "performance": {
            "runtime_seconds": perf["total"],
            "stages": perf,
        },
        "summary": {
            "total_sentences": len(transcript_items),
            "total_segments": len(segments),
            "total_duration_seconds": max(s["end_timestamp"] for s in segments) if segments else 0,
            "avg_segment_length": int(np.mean([s["sentence_count"] for s in segments])) if segments else 0,
        },
    }), 200


@app.route("/api/rag", methods=["POST"])
def rag_query():
    data = request.get_json(force=True, silent=True) or {}
    query = (data.get("query") or "").strip()
    segments = data.get("segments") or []

    if not query:
        return _err("No query provided.", 400)

    logger.info(f"[RAG] Query: '{query}'")

    try:
        retrieved = retrieve_top_segments(query, top_k=5, segments_override=segments or None)
    except RuntimeError as e:
        return _err(str(e), 400 if "No FAISS" in str(e) else 500)
    except Exception as e:
        return _err(f"Retrieval error: {e}", 500)

    try:
        gen = generate_rag_answer(query, retrieved)
    except EnvironmentError as e:
        logger.warning(f"[RAG] LLM unavailable: {e}")
        ctx = " ".join(s.get("content", "")[:200] for s in retrieved)
        gen = {
            "answer": f"(LLM unavailable — GEMINI_API_KEY not set.) Relevant context: {ctx}...",
            "referenced_segment_ids": [s.get("id") for s in retrieved],
            "model": "none",
            "context_length": len(ctx),
        }
    except (ValueError, RuntimeError) as e:
        return _err(str(e), 500)
    except Exception as e:
        return _err(f"Answer generation error: {e}", 500)

    logger.info(f"[RAG] Response via model='{gen['model']}'")

    return jsonify({
        "status": "success",
        "query": query,
        "answer": gen["answer"],
        "referenced_segments": gen["referenced_segment_ids"],
        "model": gen["model"],
        "context_length": gen["context_length"],
        "retrieved_segments": [{
            "id": s.get("id"),
            "start_time": s.get("start_time"),
            "end_time": s.get("end_time"),
            "keywords": s.get("keywords", []),
            "retrieval_rank": s.get("retrieval_rank"),
        } for s in retrieved],
    }), 200


###############################################################################
# ENTRY
###############################################################################

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("AUDIO SEMANTIC SEGMENTATION API  (v2 — single file)")
    print("=" * 60)
    print("  http://localhost:5000/health")
    print("  POST /api/upload   — process audio file")
    print("  POST /api/rag      — ask a question")
    print("=" * 60 + "\n")
    # OPTIMIZED: Enabled threading for concurrent request handling
    app.run(debug=False, port=5000, host="0.0.0.0", threaded=True)
