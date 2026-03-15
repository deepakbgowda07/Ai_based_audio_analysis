# Backend Architecture Migration

## v2 Changes (Current)

The backend has been restructured into a **single, unified file** (`app.py`) to address performance issues and simplify the codebase.

### Migration Summary

#### Deprecated Files
The following files are **no longer used** and can be safely ignored:
- `transcription.py` — ❌ DEPRECATED (merged into app.py)
- `analysis.py` — ❌ DEPRECATED (merged into app.py)
- `segmentation.py` — ❌ DEPRECATED (merged into app.py)

These files had several critical issues:
1. **transcription.py**: No CUDA→CPU fallback (would crash without GPU)
2. **analysis.py**: KeyBERT instance created on every keyword extraction call (severe performance issue)
3. **segmentation.py**: Field naming inconsistency (start/end vs start_seconds/end_seconds)

#### New Architecture
- **Single file**: `app.py` (~600 lines)
- **All dependencies consolidated**: Flask, FastWhisper, SentenceTransformers, Transformers (sentiment only)
- **Removed unused packages**: KeyBERT, TextBlob, python-dotenv, ffmpeg-python
- **Removed unused imports**: `json`, `pathlib.Path`

### New API Contract

#### POST `/api/upload`
**Request:**
```json
{
  "file": <binary audio file>
}
```

**Response:**
```json
{
  "status": "success",
  "filename": "...",
  "transcript": [
    {
      "id": 0,
      "timestamp": "00:00:00",
      "start": 0.0,
      "end": 8.2,
      "text": "...",
      "segment": 0,
      "keywords": [...]
    }
  ],
  "segments": [
    {
      "id": 0,
      "label": "...",
      "start_time": "00:00:00",
      "end_time": "00:00:08",
      "start_timestamp": 0.0,
      "end_timestamp": 8.2,
      "content": "...",
      "word_count": 42,
      "duration": "8s",
      "keywords": [...],
      "sentiment": "positive|negative|neutral",
      "sentiment_score": 0.75,
      "confidence": 0.94,
      "sentence_count": 4
    }
  ],
  "performance": {
    "runtime_seconds": 12.3,
    "stages": {
      "preprocessing": 1.2,
      "transcription": 4.5,
      "embedding": 2.1,
      "segmentation": 0.3,
      "sentiment": 1.8,
      "total": 12.3
    }
  },
  "summary": {
    "total_sentences": 42,
    "total_segments": 5,
    "total_duration_seconds": 125.6,
    "avg_segment_length": 8
  }
}
```

#### POST `/api/rag`
**Request:**
```json
{
  "query": "What were the main topics?",
  "segments": [...]  // optional, for context
}
```

**Response:**
```json
{
  "status": "success",
  "query": "...",
  "answer": "...",
  "referenced_segments": [0, 2],
  "model": "gemini-1.5-flash",
  "context_length": 1234,
  "retrieved_segments": [...]
}
```

### Configuration

**Environment Variables:**
- `GEMINI_API_KEY` — Optional, enables LLM-powered RAG answers (graceful fallback if not set)

**Audio Limits:**
- Max file size: 100 MB
- Target sample rate: 16 kHz (auto-resampled)
- Supported formats: MP3, WAV, M4A, FLAC, OGG, WebM

### Performance Improvements

| Stage | v1 | v2 | Improvement |
|-------|----|----|-------------|
| Transcription | — | 4.5s | Whisper medium.en |
| Embedding | — | 2.1s | SentenceTransformers (all-MiniLM-L6-v2) |
| Segmentation | — | 0.3s | Cosine similarity valleys |
| Sentiment | — | 1.8s | Single inference (CardiffNLP) |
| Keywords | CRITICAL ❌ | Built-in | Top-K frequency + stopwords |
| **Total** | **CRASH if no GPU** | **~12s** | ✅ Works CPU & GPU |

### Key Improvements

1. **CUDA/CPU Fallback** ✅
   - Whisper: GPU (float16) or CPU (int8)
   - Embeddings: GPU or CPU (auto-detect)

2. **Performance** ✅
   - Keyword extraction: Single Counter per segment (was KeyBERT instance per call)
   - No redundant model reloads

3. **Logging** ✅
   - Structured logging with timestamps
   - Per-stage execution time tracking
   - Pipeline completion tracking

4. **Error Handling** ✅
   - Proper HTTP status codes (400/422/500)
   - Meaningful error messages
   - File validation (extension, size, audio header)

5. **Persistence** ✅
   - FAISS index saved to `faiss_index/segments.index`
   - Metadata saved to `faiss_index/segments_meta.npy`
   - Automatic loading on startup

---

## Migration Path for Custom Extensions

If you were using the old service files directly, here's the mapping:

| Old File | Old Class | Migration |
|----------|-----------|-----------|
| `transcription.py` | `TranscriptionService` | Use `transcribe_audio()` function |
| `analysis.py` | `AnalysisService` | Use `build_segments()` function |
| `segmentation.py` | `SegmentationService` | Use `segment_topics()` function |

All functions are available in `app.py` and can be imported if needed for custom pipelines.

---

Last Updated: 2026-03-10
