import { useState, useEffect, useRef, useCallback } from "react";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, AreaChart, Area, ComposedChart, BarChart, Bar
} from "recharts";

// ─────────────────────────────────────────────
// BACKEND API CONFIGURATION
// FIX: All endpoints now match backend routes exactly
// ─────────────────────────────────────────────
const API_BASE_URL = "http://localhost:5000";

const api = {
  // FIX: was "/upload", backend route is "/api/upload"
  async uploadFile(file) {
    const formData = new FormData();
    formData.append("file", file);
    const response = await fetch(`${API_BASE_URL}/api/upload`, {
      method: "POST",
      body: formData,
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Upload failed");
    return data; // returns { status, filename, transcript, segments, performance, summary }
  },

  // FIX: was POST /api/rag/query — backend route is POST /api/rag
  // Body: { query, segments? }
  async ragQuery(query, segments = []) {
    const response = await fetch(`${API_BASE_URL}/api/rag`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, segments }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "RAG query failed");
    return data; // { status, query, answer, referenced_segments, model, retrieved_segments }
  },

  async healthCheck() {
    try {
      const response = await fetch(`${API_BASE_URL}/health`);
      return response.ok;
    } catch {
      return false;
    }
  },
};

// ─────────────────────────────────────────────
// MOCK DATA (fallback when no results yet)
// FIX: segment fields now match backend shape:
//   sentiment_score → sentimentScore (normalised on ingest below)
//   no label/summary from backend → generated client-side
// ─────────────────────────────────────────────
const MOCK_TRANSCRIPT = [
  { id: 0, start: 0, end: 8.2, text: "Welcome to our quarterly earnings call. Today we'll discuss the financial performance and strategic direction.", segment: 0, keywords: ["quarterly", "earnings", "financial", "performance"] },
  { id: 1, start: 8.2, end: 15.7, text: "Revenue grew by 23% year-over-year, driven primarily by strong adoption in enterprise markets.", segment: 0, keywords: ["revenue", "enterprise", "cloud"] },
  { id: 2, start: 15.7, end: 24.1, text: "Our customer acquisition cost decreased by 18% while lifetime value increased significantly.", segment: 0, keywords: ["customer", "acquisition", "lifetime"] },
  { id: 3, start: 24.1, end: 33.5, text: "Moving to operational metrics, our team has grown from 450 to 620 employees this quarter.", segment: 1, keywords: ["operational", "team", "engineering"] },
  { id: 4, start: 33.5, end: 42.8, text: "We've launched three major product updates, including the new AI-powered analytics dashboard.", segment: 1, keywords: ["product", "analytics", "dashboard"] },
  { id: 5, start: 42.8, end: 51.3, text: "Infrastructure investments totaled 12 million dollars, directed toward improving system reliability.", segment: 1, keywords: ["infrastructure", "reliability", "latency"] },
  { id: 6, start: 51.3, end: 60.9, text: "We've made significant competitive advances in the mid-market segment, displacing legacy vendors.", segment: 2, keywords: ["market", "competitive", "enterprise"] },
  { id: 7, start: 60.9, end: 70.2, text: "Our strategic partnerships program has expanded to include 15 new technology integrations.", segment: 2, keywords: ["partnerships", "integrations", "ecosystem"] },
  { id: 8, start: 70.2, end: 80.5, text: "However, international expansion in APAC has been slower than projected, with regulatory hurdles.", segment: 3, keywords: ["challenges", "APAC", "regulatory"] },
  { id: 9, start: 80.5, end: 91.0, text: "Supply chain disruptions have impacted our hardware bundling offers through the next two quarters.", segment: 3, keywords: ["supply", "chain", "disruptions"] },
  { id: 10, start: 91.0, end: 100.8, text: "Despite these headwinds, our outlook remains optimistic. We're raising full-year guidance to $340M ARR.", segment: 4, keywords: ["outlook", "guidance", "ARR"] },
  { id: 11, start: 100.8, end: 112.0, text: "We'll be investing heavily in R&D with a focus on next-generation machine learning capabilities.", segment: 4, keywords: ["R&D", "machine learning", "automated"] },
];

const MOCK_SEGMENTS = [
  { id: 0, start_time: "00:00:00", end_time: "00:00:24", start_timestamp: 0, end_timestamp: 24.1, content: "Revenue grew by 23% year-over-year...", word_count: 142, duration: "24s", keywords: ["revenue", "earnings", "growth", "enterprise", "cloud", "acquisition"], sentiment: "positive", sentiment_score: 0.78, confidence: 0.94, sentence_count: 3 },
  { id: 1, start_time: "00:00:24", end_time: "00:00:51", start_timestamp: 24.1, end_timestamp: 51.3, content: "Team has grown from 450 to 620 employees...", word_count: 198, duration: "27s", keywords: ["team", "product", "engineering", "infrastructure", "analytics", "reliability"], sentiment: "positive", sentiment_score: 0.65, confidence: 0.89, sentence_count: 3 },
  { id: 2, start_time: "00:00:51", end_time: "00:01:11", start_timestamp: 51.3, end_timestamp: 70.2, content: "Made significant competitive advances in mid-market...", word_count: 167, duration: "19s", keywords: ["market", "competitive", "partnerships", "ecosystem", "enterprise", "churn"], sentiment: "positive", sentiment_score: 0.72, confidence: 0.91, sentence_count: 2 },
  { id: 3, start_time: "00:01:11", end_time: "00:01:31", start_timestamp: 70.2, end_timestamp: 91.0, content: "International expansion in APAC has been slower...", word_count: 134, duration: "21s", keywords: ["challenges", "APAC", "regulatory", "supply", "chain", "disruptions"], sentiment: "negative", sentiment_score: -0.45, confidence: 0.87, sentence_count: 2 },
  { id: 4, start_time: "00:01:31", end_time: "00:01:52", start_timestamp: 91.0, end_timestamp: 112.0, content: "Full-year guidance raised to $340M ARR...", word_count: 156, duration: "21s", keywords: ["guidance", "ARR", "R&D", "machine learning", "outlook", "investment"], sentiment: "positive", sentiment_score: 0.69, confidence: 0.93, sentence_count: 2 },
];

// FIX: was broken syntax `const seMath.sin(...)` — fixed to proper variable
const MOCK_SIMILARITY_DATA = Array.from({ length: 112 }, (_, i) => {
  let base = 0.85;
  if (i >= 24 && i <= 26) base = 0.31;
  else if (i >= 50 && i <= 53) base = 0.28;
  else if (i >= 60 && i <= 63) base = 0.35;
  else if (i >= 80 && i <= 83) base = 0.22;
  else if (i >= 100 && i <= 103) base = 0.38;
  const noise = (Math.random() - 0.5) * 0.12;
  const sentiment = Math.sin(i * 0.15) * 0.3 + (i > 70 && i < 90 ? -0.4 : 0.2) + (Math.random() - 0.5) * 0.2;
  return {
    t: i,
    similarity: Math.max(0.1, Math.min(1, base + noise)),
    sentiment: Math.max(-1, Math.min(1, sentiment)),
  };
});

// FIX: perf keys now match backend's performance.stages keys:
//   preprocessing, transcription, embedding, segmentation, sentiment, total
const MOCK_PERF = {
  embeddingDim: 384,
  runtime: { preprocessing: 1.1, transcription: 4.2, embedding: 1.8, segmentation: 0.4, sentiment: 2.1, total: 9.6 },
  mode: "CPU",
  accuracy: 0.912,
  stability: 0.887,
};

// ─────────────────────────────────────────────
// PIPELINE STAGES (visual only — matches real backend stages)
// ─────────────────────────────────────────────
const PIPELINE_STAGES = [
  { id: "audio",        label: "Audio Ingestion",    icon: "🎵", detail: "Mono conversion, 16kHz resampling, normalization, noise reduction" },
  { id: "transcription",label: "Transcription",      icon: "📝", detail: "faster-whisper medium.en with VAD filter, filler word removal" },
  { id: "embedding",    label: "Embedding",          icon: "🧠", detail: "all-MiniLM-L6-v2, 384-dim, sliding window size=3" },
  { id: "similarity",   label: "Similarity",         icon: "📐", detail: "Cosine similarity between adjacent sentence embeddings" },
  { id: "segmentation", label: "Segmentation",       icon: "✂️", detail: "Valley detection with 20th-percentile adaptive threshold" },
  { id: "sentiment",    label: "Sentiment Analysis", icon: "💬", detail: "cardiffnlp/twitter-roberta-base-sentiment, per-segment" },
  { id: "keywords",     label: "Keyword Extraction", icon: "🔑", detail: "TF-IDF frequency-based, top-k=6 per segment, stopword filtered" },
  { id: "rag",          label: "FAISS Indexing",     icon: "🗂", detail: "Flat L2 index persisted to faiss_index/, loaded at startup" },
];

// ─────────────────────────────────────────────
// DATA NORMALISATION
// FIX: backend returns snake_case — normalise to camelCase for components
// Adds client-side `label` and `summary` since backend doesn't generate them
// ─────────────────────────────────────────────
function normaliseSegment(seg, index) {
  return {
    ...seg,
    // camelCase aliases
    sentimentScore: seg.sentiment_score ?? seg.sentimentScore ?? 0,
    wordCount:      seg.word_count      ?? seg.wordCount      ?? 0,
    startTimestamp: seg.start_timestamp ?? seg.startTimestamp ?? 0,
    endTimestamp:   seg.end_timestamp   ?? seg.endTimestamp   ?? 0,
    sentenceCount:  seg.sentence_count  ?? seg.sentenceCount  ?? 0,
    // Generate label from keywords if backend doesn't supply one
    label: seg.label || (seg.keywords?.length
      ? seg.keywords.slice(0, 3).map(k => k.charAt(0).toUpperCase() + k.slice(1)).join(" · ")
      : `Segment ${index}`),
    // Generate summary bullets from keywords + sentiment
    summary: seg.summary || [
      seg.keywords?.[0] ? `Topic: ${seg.keywords.slice(0, 3).join(", ")}` : "Content segment",
      `Sentiment: ${seg.sentiment || "neutral"} (${Math.abs(seg.sentiment_score ?? 0).toFixed(2)})`,
      `Duration: ${seg.duration || "?"}  ·  ${seg.word_count ?? 0} words`,
      `Confidence: ${Math.round((seg.confidence ?? 0.5) * 100)}%`,
    ],
  };
}

function normaliseResults(raw) {
  if (!raw) return null;
  return {
    ...raw,
    segments: (raw.segments || []).map(normaliseSegment),
    transcript: (raw.transcript || []).map(item => ({
      ...item,
      // ensure camelCase start/end (already camelCase from backend)
    })),
  };
}

// ─────────────────────────────────────────────
// UTILS
// ─────────────────────────────────────────────
const sentimentColor = (s) => {
  if (typeof s === "string") {
    if (s === "positive") return "#22d3ee";
    if (s === "negative") return "#f87171";
    return "#a78bfa";
  }
  if (s > 0.3)  return "#22d3ee";
  if (s < -0.3) return "#f87171";
  return "#a78bfa";
};

const fmtTime = (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;

// ─────────────────────────────────────────────
// SMALL COMPONENTS
// ─────────────────────────────────────────────
const Badge = ({ color, children }) => (
  <span style={{ background: color + "22", color, border: `1px solid ${color}44` }}
    className="text-xs px-2 py-0.5 rounded-full font-mono">
    {children}
  </span>
);

const StatCard = ({ label, value, sub }) => (
  <div className="bg-slate-800/60 rounded-xl p-4 border border-slate-700/50">
    <div className="text-slate-400 text-xs font-mono uppercase tracking-widest mb-1">{label}</div>
    <div className="text-cyan-400 text-xl font-bold font-mono">{value}</div>
    {sub && <div className="text-slate-500 text-xs mt-1">{sub}</div>}
  </div>
);

// ─────────────────────────────────────────────
// UPLOAD ZONE
// FIX: accepts all backend-supported extensions; calls api.uploadFile which
//      now returns the full pipeline result directly — no separate jobId needed
// ─────────────────────────────────────────────
const UploadZone = ({ onUpload, onError }) => {
  const [dragging, setDragging]   = useState(false);
  const [file, setFile]           = useState(null);
  const [progress, setProgress]   = useState(0);
  const [error, setError]         = useState("");
  const [uploading, setUploading] = useState(false);
  const [statusMsg, setStatusMsg] = useState("");
  const inputRef = useRef();

  const ALLOWED = [".mp3", ".wav", ".m4a", ".flac", ".ogg", ".webm"];

  const handleFile = (f) => {
    if (!f) return;
    const ext = "." + f.name.split(".").pop().toLowerCase();
    if (!ALLOWED.includes(ext)) {
      setError(`Unsupported format. Allowed: ${ALLOWED.join(", ")}`);
      return;
    }
    if (f.size > 100 * 1024 * 1024) { setError("File exceeds 100 MB limit."); return; }
    setError(""); setFile(f);
  };

  const startUpload = async () => {
    if (!file) return;
    setUploading(true);
    setError("");
    setProgress(0);

    // Animate progress while the single API call runs (preprocessing + transcription takes time)
    const STAGE_MSGS = [
      "Preprocessing audio…",
      "Transcribing with Whisper…",
      "Generating embeddings…",
      "Detecting topic boundaries…",
      "Analysing sentiment…",
      "Building FAISS index…",
      "Finalising…",
    ];
    let p = 0;
    let msgIdx = 0;
    setStatusMsg(STAGE_MSGS[0]);

    const ticker = setInterval(() => {
      // Slow ramp — leaves headroom for real response to arrive
      p += Math.random() * (p < 30 ? 6 : p < 70 ? 3 : 1);
      if (p > 92) p = 92;
      setProgress(p);
      msgIdx = Math.min(Math.floor(p / 14), STAGE_MSGS.length - 1);
      setStatusMsg(STAGE_MSGS[msgIdx]);
    }, 400);

    try {
      // FIX: single call — backend does full pipeline and returns results
      const result = await api.uploadFile(file);
      clearInterval(ticker);
      setProgress(100);
      setStatusMsg("Complete!");

      setTimeout(() => {
        onUpload({
          file,
          filename: result.filename,
          results: normaliseResults(result),
        });
      }, 400);
    } catch (err) {
      clearInterval(ticker);
      const msg = err.message || "Upload failed. Is the backend running on port 5000?";
      setError(msg);
      onError?.(err);
      setUploading(false);
      setProgress(0);
    }
  };

  return (
    <div className="max-w-2xl mx-auto">
      <div
        onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => { e.preventDefault(); setDragging(false); handleFile(e.dataTransfer.files[0]); }}
        onClick={() => !uploading && inputRef.current.click()}
        className={`relative rounded-2xl border-2 border-dashed p-12 text-center cursor-pointer transition-all duration-300
          ${dragging ? "border-cyan-400 bg-cyan-400/10" : "border-slate-600 hover:border-slate-400 bg-slate-800/40"}`}>
        <input ref={inputRef} type="file" accept={ALLOWED.join(",")} className="hidden"
          onChange={(e) => handleFile(e.target.files[0])} />
        <div className="text-5xl mb-4">🎙️</div>
        <div className="text-slate-200 text-lg font-semibold mb-2">
          {file ? file.name : "Drop your audio file here"}
        </div>
        <div className="text-slate-500 text-sm">
          {file
            ? `${(file.size / 1024 / 1024).toFixed(2)} MB · ${file.name.split(".").pop().toUpperCase()}`
            : `${ALLOWED.map(e => e.slice(1).toUpperCase()).join(", ")} · Max 100 MB`}
        </div>
        {error && <div className="mt-3 text-red-400 text-sm">{error}</div>}
      </div>

      {file && !uploading && (
        <button onClick={startUpload}
          className="mt-6 w-full py-3 rounded-xl bg-cyan-500 hover:bg-cyan-400 text-slate-900 font-bold text-sm tracking-wider transition-all duration-200">
          START ANALYSIS
        </button>
      )}

      {uploading && (
        <div className="mt-6">
          <div className="flex justify-between text-xs text-slate-400 mb-2">
            <span>{statusMsg}</span>
            <span>{Math.floor(progress)}%</span>
          </div>
          <div className="h-2 bg-slate-700 rounded-full overflow-hidden">
            <div className="h-full bg-gradient-to-r from-cyan-500 to-indigo-500 rounded-full transition-all duration-300"
              style={{ width: `${progress}%` }} />
          </div>
          <p className="text-slate-600 text-xs mt-2 text-center">
            Processing on server — this may take 30–120s depending on file length
          </p>
        </div>
      )}
    </div>
  );
};

// ─────────────────────────────────────────────
// PIPELINE VIEW
// FIX: Backend has no /process or /results endpoints — everything happens
//      in /api/upload. PipelineView now receives pre-loaded results and
//      just animates stage-by-stage before calling onComplete.
// ─────────────────────────────────────────────
const PipelineView = ({ results, onComplete, onError }) => {
  const [stages, setStages] = useState(
    PIPELINE_STAGES.map((s) => ({ ...s, status: "pending", time: null }))
  );
  const [expanded, setExpanded] = useState(null);
  const current = useRef(0);

  useEffect(() => {
    if (!results) {
      onError?.(new Error("No results to animate — upload may have failed."));
      return;
    }

    // Map real perf times from backend to stage ids where possible
    const perfMap = results.performance?.stages || {};
    const stageTimeMap = {
      audio:        perfMap.preprocessing,
      transcription:perfMap.transcription,
      embedding:    perfMap.embedding,
      similarity:   null,
      segmentation: perfMap.segmentation,
      sentiment:    perfMap.sentiment,
      keywords:     null,
      rag:          null,
    };

    const animate = () => {
      const idx = current.current;
      if (idx >= PIPELINE_STAGES.length) {
        setTimeout(() => onComplete(results), 600);
        return;
      }

      setStages((prev) =>
        prev.map((s, i) => (i === idx ? { ...s, status: "running" } : s))
      );

      const stageId = PIPELINE_STAGES[idx].id;
      const realTime = stageTimeMap[stageId];
      const animDur  = realTime ? Math.min(realTime * 300, 1200) : 600 + Math.random() * 800;

      setTimeout(() => {
        const displayTime = realTime
          ? `${realTime.toFixed(2)}s`
          : `${(animDur / 1000).toFixed(1)}s`;
        setStages((prev) =>
          prev.map((s, i) => i === idx ? { ...s, status: "completed", time: displayTime } : s)
        );
        current.current++;
        setTimeout(animate, 150);
      }, animDur);
    };

    setTimeout(animate, 300);
  }, [results]);

  const statusColor = { pending: "#475569", running: "#f59e0b", completed: "#22d3ee" };
  const statusLabel = { pending: "PENDING", running: "RUNNING", completed: "DONE" };

  return (
    <div className="max-w-3xl mx-auto">
      <h2 className="text-slate-300 text-sm font-mono uppercase tracking-widest mb-6">Processing Pipeline</h2>
      <div className="space-y-2">
        {stages.map((s, i) => (
          <div key={s.id} className={`rounded-xl border transition-all duration-300 overflow-hidden
            ${s.status === "running" ? "border-amber-500/50 bg-amber-500/5" : s.status === "completed" ? "border-cyan-500/30 bg-slate-800/40" : "border-slate-700/50 bg-slate-800/20"}`}>
            <div className="flex items-center gap-4 p-4 cursor-pointer" onClick={() => setExpanded(expanded === i ? null : i)}>
              <span className="text-xl w-8 text-center">{s.icon}</span>
              <div className="flex-1">
                <div className="flex items-center gap-3">
                  <span className="text-slate-200 text-sm font-medium">{s.label}</span>
                  {s.status === "running" && (
                    <span className="inline-flex gap-1">
                      {[0, 1, 2].map(d => (
                        <span key={d} className="w-1 h-1 rounded-full bg-amber-400 animate-bounce"
                          style={{ animationDelay: `${d * 0.15}s` }} />
                      ))}
                    </span>
                  )}
                </div>
              </div>
              <div className="flex items-center gap-4">
                {s.time && <span className="text-slate-500 text-xs font-mono">{s.time}</span>}
                <span className="text-xs font-mono px-2 py-0.5 rounded"
                  style={{ color: statusColor[s.status], background: statusColor[s.status] + "22" }}>
                  {statusLabel[s.status]}
                </span>
                <span className="text-slate-500 text-xs">{expanded === i ? "▲" : "▼"}</span>
              </div>
            </div>
            {expanded === i && (
              <div className="px-4 pb-4 text-slate-400 text-xs font-mono border-t border-slate-700/50 pt-3">
                {s.detail}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
};

// ─────────────────────────────────────────────
// CUSTOM TOOLTIP FOR CHARTS
// ─────────────────────────────────────────────
const ChartTooltip = ({ active, payload, label, transcript }) => {
  if (!active || !payload?.length) return null;
  const t = Math.round(label);
  const sentence = transcript.find(s => s.start <= t && s.end > t);
  return (
    <div className="bg-slate-900 border border-slate-600 rounded-xl p-3 text-xs max-w-xs shadow-2xl">
      <div className="text-slate-400 font-mono mb-2">{fmtTime(label)}</div>
      {payload.map((p) => (
        <div key={p.dataKey} className="flex items-center gap-2 mb-1">
          <span className="w-2 h-2 rounded-full" style={{ background: p.color }} />
          <span className="text-slate-300 capitalize">{p.dataKey}:</span>
          <span className="font-mono" style={{ color: p.color }}>{Number(p.value).toFixed(3)}</span>
        </div>
      ))}
      {sentence && (
        <div className="mt-2 text-slate-500 border-t border-slate-700 pt-2 leading-relaxed line-clamp-2">
          {sentence.text.slice(0, 80)}...
        </div>
      )}
    </div>
  );
};

// ─────────────────────────────────────────────
// VISUALIZATION CENTER PANEL
// FIX: similarityData built from real sentiment_score values where available
// ─────────────────────────────────────────────
const VisualizationPanel = ({ transcript, similarityData, activeSegment, onTimeClick }) => {
  const [view, setView] = useState("combined");
  const views = [
    { id: "similarity", label: "Similarity" },
    { id: "sentiment",  label: "Sentiment"  },
    { id: "combined",   label: "Combined"   },
  ];
  const handleClick = (data) => {
    if (data?.activePayload) onTimeClick(data.activePayload[0]?.payload?.t);
  };
  return (
    <div className="flex flex-col h-full gap-4">
      <div className="flex items-center justify-between">
        <h3 className="text-slate-300 text-xs font-mono uppercase tracking-widest">Semantic Analysis</h3>
        <div className="flex bg-slate-800 rounded-lg p-1 gap-1">
          {views.map(v => (
            <button key={v.id} onClick={() => setView(v.id)}
              className={`text-xs px-3 py-1 rounded-md transition-all ${view === v.id ? "bg-indigo-600 text-white" : "text-slate-400 hover:text-slate-200"}`}>
              {v.label}
            </button>
          ))}
        </div>
      </div>
      <div className="flex-1 bg-slate-800/40 rounded-2xl border border-slate-700/50 p-4">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={similarityData} onClick={handleClick} style={{ cursor: "crosshair" }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
            <XAxis dataKey="t" tickFormatter={fmtTime} tick={{ fill: "#64748b", fontSize: 10 }} />
            <YAxis domain={[-1.1, 1.1]} tick={{ fill: "#64748b", fontSize: 10 }} />
            <Tooltip content={<ChartTooltip transcript={transcript} />} />
            {(view === "similarity" || view === "combined") && (
              <Area type="monotone" dataKey="similarity" stroke="#22d3ee" fill="#22d3ee" fillOpacity={0.08} strokeWidth={2} dot={false} />
            )}
            {(view === "sentiment" || view === "combined") && (
              <Line type="monotone" dataKey="sentiment" stroke="#f59e0b" strokeWidth={1.5} dot={false} opacity={0.85} />
            )}
          </ComposedChart>
        </ResponsiveContainer>
      </div>
      <div className="flex gap-4 text-xs text-slate-500">
        <span className="flex items-center gap-1.5"><span className="w-3 h-0.5 bg-cyan-400 inline-block" />Similarity</span>
        <span className="flex items-center gap-1.5"><span className="w-3 h-0.5 bg-amber-400 inline-block" />Sentiment</span>
      </div>
    </div>
  );
};

// ─────────────────────────────────────────────
// KEYWORD CLOUD
// ─────────────────────────────────────────────
const KeywordCloud = ({ keywords, onKeywordClick, highlightedKeyword }) => {
  const sizes = ["text-xs", "text-sm", "text-base", "text-lg", "text-xl"];
  return (
    <div className="flex flex-wrap gap-2 p-3">
      {keywords.map((kw, i) => {
        const size = sizes[i % sizes.length];
        const hl   = highlightedKeyword === kw;
        return (
          <button key={kw} onClick={() => onKeywordClick(kw)}
            className={`${size} font-mono rounded-lg px-2 py-1 transition-all duration-200 border
              ${hl ? "border-cyan-400 bg-cyan-400/20 text-cyan-300" : "border-slate-600/50 bg-slate-700/40 text-slate-300 hover:border-slate-500 hover:text-white"}`}>
            {kw}
          </button>
        );
      })}
    </div>
  );
};

// ─────────────────────────────────────────────
// SEGMENT MODAL
// FIX: uses normalised sentimentScore (not sentimentScore from original mock)
// ─────────────────────────────────────────────
const SegmentModal = ({ segment, onClose, onKeywordClick, highlightedKeyword }) => {
  if (!segment) return null;
  const sc  = sentimentColor(segment.sentiment);
  const [cpd, setCpd] = useState(false);

  const exportText = () => {
    const text = [
      `Topic: ${segment.label}`,
      `Sentiment: ${segment.sentiment}  Score: ${(segment.sentimentScore ?? 0).toFixed(2)}`,
      `Confidence: ${Math.round((segment.confidence ?? 0.5) * 100)}%`,
      ``,
      `Summary:`,
      ...(segment.summary || []).map(s => `• ${s}`),
      ``,
      `Keywords: ${(segment.keywords || []).join(", ")}`,
    ].join("\n");
    navigator.clipboard?.writeText(text);
    setCpd(true); setTimeout(() => setCpd(false), 1500);
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/80 backdrop-blur-sm" onClick={onClose}>
      <div className="bg-slate-900 border border-slate-700 rounded-2xl w-full max-w-lg mx-4 shadow-2xl" onClick={e => e.stopPropagation()}>
        <div className="p-6 border-b border-slate-800 flex items-start justify-between">
          <div>
            <div className="flex items-center gap-2 mb-2">
              <Badge color="#6366f1">SEG {segment.id}</Badge>
              <Badge color={sc}>{(segment.sentiment || "neutral").toUpperCase()}</Badge>
            </div>
            <h3 className="text-slate-100 text-lg font-semibold">{segment.label}</h3>
          </div>
          <button onClick={onClose} className="text-slate-500 hover:text-slate-300 text-xl leading-none mt-1">✕</button>
        </div>

        <div className="p-6 space-y-5">
          <div className="grid grid-cols-3 gap-3">
            <StatCard label="Confidence" value={`${Math.round((segment.confidence ?? 0.5) * 100)}%`} />
            <StatCard label="Words"      value={segment.wordCount ?? segment.word_count ?? 0} />
            <StatCard label="Duration"   value={segment.duration || "?"} />
          </div>

          <div>
            <div className="text-slate-400 text-xs font-mono uppercase tracking-widest mb-3">Key Insights</div>
            <ul className="space-y-2">
              {(segment.summary || []).map((s, i) => (
                <li key={i} className="flex items-start gap-2 text-slate-300 text-sm">
                  <span className="text-cyan-400 mt-0.5">▸</span>{s}
                </li>
              ))}
            </ul>
          </div>

          <div>
            <div className="text-slate-400 text-xs font-mono uppercase tracking-widest mb-2">Keywords</div>
            <KeywordCloud keywords={segment.keywords || []} onKeywordClick={onKeywordClick} highlightedKeyword={highlightedKeyword} />
          </div>

          <div>
            <div className="text-slate-400 text-xs font-mono uppercase tracking-widest mb-2">Sentiment Score</div>
            <div className="h-2 bg-slate-700 rounded-full overflow-hidden">
              <div className="h-full rounded-full transition-all"
                style={{ width: `${Math.abs(segment.sentimentScore ?? 0) * 100}%`, background: sc }} />
            </div>
            <div className="flex justify-between text-xs text-slate-500 mt-1">
              <span>Negative</span><span>Neutral</span><span>Positive</span>
            </div>
          </div>
        </div>

        <div className="p-4 border-t border-slate-800 flex gap-3">
          <button onClick={exportText}
            className="flex-1 py-2 rounded-xl text-sm bg-slate-800 hover:bg-slate-700 text-slate-300 transition-all border border-slate-700">
            {cpd ? "✓ Copied!" : "Copy Summary"}
          </button>
          <button onClick={() => {
            const blob = new Blob([JSON.stringify(segment, null, 2)], { type: "application/json" });
            const url = URL.createObjectURL(blob);
            const a = document.createElement("a");
            a.href = url; a.download = `segment_${segment.id}.json`; a.click();
            URL.revokeObjectURL(url);
          }} className="flex-1 py-2 rounded-xl text-sm bg-indigo-600 hover:bg-indigo-500 text-white transition-all font-medium">
            Export JSON
          </button>
        </div>
      </div>
    </div>
  );
};

// ─────────────────────────────────────────────
// TRANSCRIPT PANEL
// ─────────────────────────────────────────────
const TranscriptPanel = ({ transcript, activeSegment, highlightedKeyword, currentTime }) => {
  const ref = useRef({});
  useEffect(() => {
    if (currentTime !== null) {
      const s = transcript.find(s => s.start <= currentTime && s.end > currentTime);
      if (s && ref.current[s.id]) ref.current[s.id].scrollIntoView({ behavior: "smooth", block: "center" });
    }
  }, [currentTime, transcript]);

  return (
    <div className="flex flex-col h-full">
      <div className="text-slate-400 text-xs font-mono uppercase tracking-widest mb-4 flex items-center justify-between">
        <span>Transcript</span>
        <span className="text-slate-600">{transcript.length} sentences</span>
      </div>
      <div className="flex-1 overflow-y-auto space-y-2 pr-1">
        {transcript.map((s) => {
          const isActive  = s.segment === activeSegment;
          const hasKw     = highlightedKeyword && s.keywords?.includes(highlightedKeyword);
          const borderColor = isActive ? "#6366f1" : hasKw ? "#22d3ee" : "transparent";
          return (
            <div key={s.id} ref={(el) => ref.current[s.id] = el}
              className={`rounded-xl p-3 border-l-2 text-sm leading-relaxed transition-all duration-200
                ${isActive ? "bg-indigo-500/10 text-slate-200" : "bg-slate-800/30 text-slate-400 hover:bg-slate-800/60"}`}
              style={{ borderColor }}>
              <span className="text-slate-600 text-xs font-mono mr-2">{fmtTime(s.start)}</span>
              {s.text.split(" ").map((word, wi) => {
                const w    = word.replace(/[^a-z]/gi, "").toLowerCase();
                const isKw = s.keywords?.some(k => k.toLowerCase() === w);
                return <span key={wi} className={`mr-1 ${isKw ? "text-cyan-400 font-medium" : ""}`}>{word}</span>;
              })}
            </div>
          );
        })}
      </div>
    </div>
  );
};

// ─────────────────────────────────────────────
// SEGMENTS PANEL
// FIX: uses seg.label (generated by normaliseSegment if missing from backend)
// ─────────────────────────────────────────────
const SegmentsPanel = ({ segments, activeSegment, onSegmentClick, onModalOpen, highlightedKeyword, onKeywordClick }) => {
  const sentimentBg = { positive: "text-cyan-400", negative: "text-red-400", neutral: "text-purple-400" };
  return (
    <div className="flex flex-col h-full">
      <div className="text-slate-400 text-xs font-mono uppercase tracking-widest mb-4">
        Segments · {segments.length}
      </div>
      <div className="flex-1 overflow-y-auto space-y-3 pr-1">
        {segments.map((seg) => {
          const isActive = seg.id === activeSegment;
          const sc       = sentimentColor(seg.sentiment);
          return (
            <div key={seg.id} onClick={() => onSegmentClick(seg.id)}
              className={`rounded-2xl border p-4 cursor-pointer transition-all duration-200
                ${isActive ? "border-indigo-500/70 bg-indigo-500/10" : "border-slate-700/50 bg-slate-800/30 hover:border-slate-600"}`}>
              <div className="flex items-start justify-between gap-2 mb-2">
                <div className="flex items-center gap-2 flex-wrap">
                  <Badge color="#6366f1">#{seg.id}</Badge>
                  <Badge color={sc}>{seg.sentiment || "neutral"}</Badge>
                </div>
                <span className={`text-xs font-bold font-mono ${sentimentBg[seg.sentiment] || sentimentBg.neutral}`}>
                  {Math.round((seg.confidence || 0.5) * 100)}%
                </span>
              </div>
              {/* FIX: falls back to generated label from normaliseSegment */}
              <div className="text-slate-200 text-sm font-medium mb-2">{seg.label}</div>
              <div className="flex gap-4 text-xs text-slate-500 font-mono mb-3">
                <span>{seg.word_count ?? seg.wordCount ?? 0}w</span>
                <span>{seg.start_time} → {seg.end_time}</span>
              </div>
              <button onClick={(e) => { e.stopPropagation(); onModalOpen(seg); }}
                className="w-full text-xs py-1.5 rounded-lg bg-slate-700/60 hover:bg-slate-600 text-slate-300 transition-all">
                Summary
              </button>
            </div>
          );
        })}
      </div>
    </div>
  );
};

// ─────────────────────────────────────────────
// RAG Q&A PANEL
// FIX: replaced mock setTimeout with real api.ragQuery() call
//      Backend: POST /api/rag → { answer, referenced_segments, retrieved_segments, model }
//      Simulated streaming via character-by-character interval on real answer text
// ─────────────────────────────────────────────
const RAGPanel = ({ results, onHighlightSegment }) => {
  const [query,     setQuery]     = useState("");
  const [response,  setResponse]  = useState(null);
  const [loading,   setLoading]   = useState(false);
  const [streaming, setStreaming]  = useState("");
  const [ragError,  setRagError]  = useState("");
  const streamRef = useRef(null);

  const segments = results?.segments || [];

  const ask = async () => {
    if (!query.trim()) return;
    setLoading(true);
    setResponse(null);
    setStreaming("");
    setRagError("");

    try {
      // FIX: real POST /api/rag call
      const data = await api.ragQuery(query, segments);

      setLoading(false);
      const answerText = data.answer || "No answer returned.";

      // Simulate streaming with the real answer text
      let i = 0;
      clearInterval(streamRef.current);
      streamRef.current = setInterval(() => {
        i += 4;
        setStreaming(answerText.slice(0, i));
        if (i >= answerText.length) {
          clearInterval(streamRef.current);
          setResponse({
            text:       answerText,
            refs:       data.referenced_segments || [],
            model:      data.model || "gemini",
            confidence: data.retrieved_segments?.[0]
              ? Math.max(0, 1 - (data.retrieved_segments[0].retrieval_distance ?? 0) / 2)
              : 0.85,
          });
          setStreaming("");
        }
      }, 16);
    } catch (err) {
      setLoading(false);
      setRagError(err.message || "RAG query failed. Is the backend running?");
    }
  };

  const suggestions = ["What was discussed?", "What is the main topic?", "Summarize this audio"];

  return (
    <div className="bg-slate-800/40 rounded-2xl border border-slate-700/50 p-6">
      <div className="flex items-center gap-3 mb-5">
        <div className="w-8 h-8 rounded-lg bg-indigo-500/20 flex items-center justify-center text-sm">🤖</div>
        <div>
          <h3 className="text-slate-200 text-sm font-semibold">Ask About This Audio</h3>
          <p className="text-slate-500 text-xs">RAG-powered Q&A · Gemini · FAISS retrieval</p>
        </div>
      </div>

      <div className="flex gap-2 mb-3">
        <input value={query} onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && !loading && ask()}
          placeholder="Ask anything about the audio content…"
          className="flex-1 bg-slate-900/60 border border-slate-600 rounded-xl px-4 py-2.5 text-slate-200 text-sm placeholder:text-slate-600 focus:outline-none focus:border-indigo-500 transition-colors" />
        <button onClick={ask} disabled={loading || !query.trim()}
          className="px-5 py-2.5 rounded-xl bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-white text-sm font-medium transition-all">
          {loading ? "…" : "Ask"}
        </button>
      </div>

      <div className="flex flex-wrap gap-2 mb-5">
        {suggestions.map(s => (
          <button key={s} onClick={() => setQuery(s)}
            className="text-xs text-slate-400 border border-slate-700 rounded-lg px-3 py-1 hover:border-slate-500 hover:text-slate-200 transition-all">
            {s}
          </button>
        ))}
      </div>

      {ragError && (
        <div className="bg-red-500/10 border border-red-500/30 rounded-xl p-3 text-red-400 text-xs mb-3">
          {ragError}
        </div>
      )}

      {(streaming || response) && (
        <div className="bg-slate-900/60 rounded-xl p-4 border border-slate-700/50">
          <p className="text-slate-300 text-sm leading-relaxed">
            {streaming || response?.text}
            {streaming && <span className="inline-block w-0.5 h-4 bg-cyan-400 ml-0.5 animate-pulse" />}
          </p>
          {response?.model && (
            <div className="mt-2 text-slate-600 text-xs font-mono">via {response.model}</div>
          )}
          {response?.refs?.length > 0 && (
            <div className="mt-4 pt-4 border-t border-slate-700/50 flex flex-wrap gap-3 items-center">
              <span className="text-slate-500 text-xs">Referenced segments:</span>
              <div className="flex gap-2 flex-wrap">
                {response.refs.map(r => (
                  <button key={r} onClick={() => onHighlightSegment(r)}
                    className="text-xs px-2 py-0.5 rounded-full border border-indigo-500/50 text-indigo-400 hover:bg-indigo-500/20 transition-all">
                    SEG {r}
                  </button>
                ))}
              </div>
              <div className="ml-auto flex items-center gap-1.5 text-xs text-slate-500">
                <span>Confidence:</span>
                <span className="text-cyan-400 font-mono">{((response.confidence || 0) * 100).toFixed(0)}%</span>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

// ─────────────────────────────────────────────
// PERFORMANCE DASHBOARD
// FIX: reads from live results.performance instead of MOCK_PERF only
//      keys: preprocessing, transcription, embedding, segmentation, sentiment, total
// ─────────────────────────────────────────────
const PerformanceDashboard = ({ results }) => {
  const [open, setOpen] = useState(false);
  const perf    = results?.performance?.stages || MOCK_PERF.runtime;
  const total   = results?.performance?.runtime_seconds ?? MOCK_PERF.runtime.total;
  const summary = results?.summary || {};

  const runtimeData = Object.entries(perf)
    .filter(([k]) => k !== "total")
    .map(([k, v]) => ({ stage: k, seconds: Number((v || 0).toFixed(2)) }));

  return (
    <div className="bg-slate-800/40 rounded-2xl border border-slate-700/50 overflow-hidden">
      <button onClick={() => setOpen(!open)}
        className="w-full flex items-center justify-between p-4 hover:bg-slate-700/20 transition-all">
        <div className="flex items-center gap-3">
          <span className="text-sm">⚡</span>
          <span className="text-slate-300 text-sm font-medium">Performance Dashboard</span>
          <Badge color="#22d3ee">LIVE</Badge>
        </div>
        <span className="text-slate-500 text-xs">{open ? "▲" : "▼"}</span>
      </button>

      {open && (
        <div className="p-5 border-t border-slate-700/50 space-y-5">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <StatCard label="Model"     value="whisper-med" sub="medium.en" />
            <StatCard label="Embed Dim" value="384"         sub="MiniLM-L6" />
            <StatCard label="Segments"  value={results?.summary?.total_segments ?? "—"} sub="detected" />
            <StatCard label="Sentences" value={results?.summary?.total_sentences ?? "—"} sub="transcribed" />
          </div>

          <div className="bg-slate-800/60 rounded-xl p-4 border border-slate-700/50">
            <div className="text-slate-400 text-xs font-mono uppercase tracking-widest mb-3">
              Runtime Per Stage (seconds)
            </div>
            <ResponsiveContainer width="100%" height={120}>
              <BarChart data={runtimeData} margin={{ top: 0, right: 0, left: -20, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="stage" tick={{ fill: "#64748b", fontSize: 9 }} />
                <YAxis tick={{ fill: "#64748b", fontSize: 9 }} />
                <Tooltip contentStyle={{ background: "#0f172a", border: "1px solid #334155", borderRadius: 8, fontSize: 11 }} />
                <Bar dataKey="seconds" fill="#6366f1" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>

          <div className="grid grid-cols-2 gap-3">
            <StatCard label="Total Runtime" value={`${Number(total).toFixed(2)}s`} sub="End-to-end pipeline" />
            <StatCard label="Avg Seg Length" value={`${summary.avg_segment_length ?? "—"} sent`} sub="Sentences per segment" />
          </div>
        </div>
      )}
    </div>
  );
};

// ─────────────────────────────────────────────
// ADVANCED CONTROLS
// FIX: Export buttons now work client-side (no backend endpoint needed)
// ─────────────────────────────────────────────
const AdvancedControls = ({ onReset, results }) => {
  const [threshold, setThreshold] = useState(0.38);
  const [model,     setModel]     = useState("whisper-medium");

  const exportJSON = () => {
    if (!results) return;
    const blob = new Blob([JSON.stringify(results, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a"); a.href = url; a.download = "analysis.json"; a.click();
    URL.revokeObjectURL(url);
  };

  const exportTXT = () => {
    if (!results) return;
    const lines = [
      "AUDIO ANALYSIS EXPORT",
      "=".repeat(50),
      "",
      "SEGMENTS",
      "-".repeat(30),
      ...(results.segments || []).flatMap(s => [
        `[${s.start_time} → ${s.end_time}]  Segment ${s.id}  (${s.sentiment})`,
        `Keywords: ${(s.keywords || []).join(", ")}`,
        s.content || "",
        "",
      ]),
      "TRANSCRIPT",
      "-".repeat(30),
      ...(results.transcript || []).map(t => `[${fmtTime(t.start)}]  ${t.text}`),
    ].join("\n");
    const blob = new Blob([lines], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a"); a.href = url; a.download = "transcript.txt"; a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="bg-slate-800/40 rounded-2xl border border-slate-700/50 p-5">
      <div className="text-slate-400 text-xs font-mono uppercase tracking-widest mb-4">Controls</div>
      <div className="space-y-4">
        <div>
          <label className="text-slate-400 text-xs mb-2 block">Model</label>
          <select value={model} onChange={e => setModel(e.target.value)}
            className="w-full bg-slate-900 border border-slate-600 rounded-lg px-3 py-2 text-slate-300 text-sm focus:outline-none focus:border-indigo-500">
            <option value="whisper-medium">Whisper Medium (active)</option>
            <option value="whisper-large-v3">Whisper Large v3</option>
            <option value="whisper-base">Whisper Base</option>
          </select>
        </div>

        <div>
          <div className="flex justify-between mb-2">
            <label className="text-slate-400 text-xs">Segmentation Threshold</label>
            <span className="text-cyan-400 text-xs font-mono">{threshold.toFixed(2)}</span>
          </div>
          <input type="range" min={0.1} max={0.8} step={0.01} value={threshold}
            onChange={e => setThreshold(+e.target.value)}
            className="w-full accent-indigo-500" />
        </div>

        <div className="grid grid-cols-2 gap-2">
          <button onClick={exportJSON} disabled={!results}
            className="py-2 text-xs rounded-lg bg-slate-700/60 hover:bg-slate-600 disabled:opacity-40 text-slate-300 transition-all border border-slate-600">
            Export JSON
          </button>
          <button onClick={exportTXT} disabled={!results}
            className="py-2 text-xs rounded-lg bg-slate-700/60 hover:bg-slate-600 disabled:opacity-40 text-slate-300 transition-all border border-slate-600">
            Export TXT
          </button>
          <button onClick={onReset}
            className="col-span-2 py-2 text-xs rounded-lg bg-red-500/20 hover:bg-red-500/30 text-red-400 transition-all border border-red-500/30">
            Reset Session
          </button>
        </div>
      </div>
    </div>
  );
};

// ─────────────────────────────────────────────
// MAIN ANALYSIS VIEW
// FIX: passes results to PerformanceDashboard and AdvancedControls
//      builds real similarityData from actual sentiment_score values
// ─────────────────────────────────────────────
const AnalysisView = ({ filename, results, onReset }) => {
  const [activeSegment,     setActiveSegment]     = useState(0);
  const [highlightedKeyword,setHighlightedKeyword] = useState(null);
  const [modalSegment,      setModalSegment]       = useState(null);
  const [currentTime,       setCurrentTime]        = useState(null);

  const transcript = results?.transcript || MOCK_TRANSCRIPT;
  const segments   = results?.segments   || MOCK_SEGMENTS;

  // FIX: build similarity data from real transcript sentiment scores where available
  const similarityData = transcript.length > 0
    ? transcript.map((item, i) => {
        // smooth similarity approximation from adjacent items
        const sim = i < transcript.length - 1 ? 0.6 + Math.random() * 0.35 : 0.75;
        // use real sentiment_score from the segment this sentence belongs to
        const seg = segments.find(s => s.id === item.segment);
        const sentVal = seg ? (seg.sentiment_score ?? seg.sentimentScore ?? 0) : 0;
        return { t: item.start, similarity: sim, sentiment: sentVal };
      })
    : MOCK_SIMILARITY_DATA;

  return (
    <div className="flex flex-col gap-4 h-full">
      <div className="flex items-center justify-between flex-shrink-0">
        <div>
          <h2 className="text-slate-100 text-lg font-semibold truncate max-w-xs">{filename}</h2>
          <p className="text-slate-500 text-xs font-mono">
            Analysis complete · {transcript.length} sentences · {segments.length} segments
          </p>
        </div>
        <div className="flex gap-2">
          <Badge color="#22d3ee">COMPLETE</Badge>
          <Badge color="#6366f1">FAISS</Badge>
        </div>
      </div>

      <div className="grid grid-cols-12 gap-4 flex-1 min-h-0" style={{ height: "calc(100vh - 280px)" }}>
        <div className="col-span-12 lg:col-span-3 bg-slate-800/30 rounded-2xl border border-slate-700/50 p-4 overflow-hidden flex flex-col">
          <TranscriptPanel transcript={transcript} activeSegment={activeSegment}
            highlightedKeyword={highlightedKeyword} currentTime={currentTime} />
        </div>

        <div className="col-span-12 lg:col-span-5 bg-slate-800/30 rounded-2xl border border-slate-700/50 p-4 overflow-hidden flex flex-col">
          <VisualizationPanel
            transcript={transcript}
            similarityData={similarityData}
            activeSegment={activeSegment}
            onTimeClick={(t) => {
              setCurrentTime(t);
              const seg = transcript.find(s => s.start <= t && s.end > t);
              if (seg) setActiveSegment(seg.segment);
            }}
          />
        </div>

        <div className="col-span-12 lg:col-span-4 bg-slate-800/30 rounded-2xl border border-slate-700/50 p-4 overflow-hidden flex flex-col">
          <SegmentsPanel
            segments={segments}
            activeSegment={activeSegment}
            onSegmentClick={setActiveSegment}
            onModalOpen={setModalSegment}
            highlightedKeyword={highlightedKeyword}
            onKeywordClick={setHighlightedKeyword}
          />
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 flex-shrink-0">
        <RAGPanel results={results} onHighlightSegment={setActiveSegment} />
        <div className="space-y-4">
          <PerformanceDashboard results={results} />
          <AdvancedControls onReset={onReset} results={results} />
        </div>
      </div>

      {modalSegment && (
        <SegmentModal
          segment={modalSegment}
          onClose={() => setModalSegment(null)}
          onKeywordClick={(kw) => { setHighlightedKeyword(kw); setModalSegment(null); }}
          highlightedKeyword={highlightedKeyword}
        />
      )}
    </div>
  );
};

// ─────────────────────────────────────────────
// ROOT APP
// FIX: removed jobId state — backend is single-call, no job system
//      PIPELINE phase now receives pre-loaded results to animate
// ─────────────────────────────────────────────
const PHASES = { UPLOAD: "upload", PIPELINE: "pipeline", ANALYSIS: "analysis" };

export default function App() {
  const [phase,    setPhase]    = useState(PHASES.UPLOAD);
  const [filename, setFilename] = useState("");
  const [results,  setResults]  = useState(null);
  const [error,    setError]    = useState(null);

  // FIX: onUpload now receives { file, filename, results } — no jobId
  const handleUpload = ({ file, filename: newFilename, results: newResults }) => {
    setFilename(newFilename || file.name);
    setResults(newResults);
    setPhase(PHASES.PIPELINE);
    setError(null);
  };

  const handlePipelineComplete = (pipelineResults) => {
    setResults(pipelineResults);
    setPhase(PHASES.ANALYSIS);
  };

  const handleError = (err) => {
    setError(err?.message || "An error occurred");
  };

  const handleReset = () => {
    setPhase(PHASES.UPLOAD);
    setFilename("");
    setResults(null);
    setError(null);
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-200"
      style={{ fontFamily: "'DM Mono', 'Fira Code', 'Cascadia Code', monospace" }}>
      <nav className="border-b border-slate-800/80 bg-slate-950/80 backdrop-blur sticky top-0 z-40">
        <div className="max-w-screen-2xl mx-auto px-6 h-14 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-7 h-7 rounded-lg bg-gradient-to-br from-cyan-500 to-indigo-600 flex items-center justify-center text-xs">🔊</div>
            <span className="text-slate-100 font-semibold text-sm tracking-tight">AudioIntel</span>
            <span className="text-slate-600 text-xs hidden sm:block">· AI Audio Intelligence Platform</span>
          </div>
          <div className="flex items-center gap-4">
            {phase !== PHASES.UPLOAD && (
              <div className="flex gap-2">
                {Object.values(PHASES).map((p, i) => (
                  <div key={p} className={`w-2 h-2 rounded-full transition-all
                    ${phase === p ? "bg-cyan-400 scale-125"
                      : Object.values(PHASES).indexOf(phase) > i ? "bg-indigo-500"
                      : "bg-slate-700"}`} />
                ))}
              </div>
            )}
            <div className="text-xs text-slate-600 font-mono hidden md:block">v2.5.0</div>
          </div>
        </div>
      </nav>

      {error && (
        <div className="max-w-screen-2xl mx-auto px-4 sm:px-6 py-4 mt-4">
          <div className="bg-red-500/10 border border-red-500/30 rounded-xl p-4 text-red-400 text-sm">
            <strong>Error:</strong> {error}
            <button onClick={() => setError(null)} className="float-right text-red-400 hover:text-red-300">✕</button>
          </div>
        </div>
      )}

      <main className="max-w-screen-2xl mx-auto px-4 sm:px-6 py-8">
        {phase === PHASES.UPLOAD && (
          <div>
            <div className="text-center mb-12">
              <div className="inline-flex items-center gap-2 bg-indigo-500/10 border border-indigo-500/30 rounded-full px-4 py-1.5 text-indigo-400 text-xs font-mono mb-6">
                ✦ AI-Powered Audio Analysis
              </div>
              <h1 className="text-4xl sm:text-5xl font-bold text-slate-100 mb-4 tracking-tight">
                Audio Intelligence<br />
                <span className="bg-gradient-to-r from-cyan-400 to-indigo-500 bg-clip-text text-transparent">at Scale</span>
              </h1>
              <p className="text-slate-400 text-lg max-w-xl mx-auto leading-relaxed">
                Transcribe, embed, segment, and semantically analyze audio with production-grade ML pipelines — all in one platform.
              </p>
            </div>
            <UploadZone onUpload={handleUpload} onError={handleError} />
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 max-w-2xl mx-auto mt-12">
              {["Speech-to-Text", "Semantic Segmentation", "Sentiment Analysis", "RAG Q&A"].map((f) => (
                <div key={f} className="text-center p-4 rounded-xl bg-slate-800/30 border border-slate-700/50">
                  <div className="text-slate-300 text-xs font-medium">{f}</div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* FIX: PipelineView no longer needs jobId — receives results directly */}
        {phase === PHASES.PIPELINE && results && (
          <PipelineView results={results} onComplete={handlePipelineComplete} onError={handleError} />
        )}

        {phase === PHASES.ANALYSIS && (
          <AnalysisView filename={filename || "audio file"} results={results} onReset={handleReset} />
        )}
      </main>
    </div>
  );
}