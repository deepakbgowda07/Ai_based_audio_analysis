import { useState, useEffect, useRef } from "react";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ReferenceLine, AreaChart, Area, ComposedChart, Bar, BarChart
} from "recharts";
import SegmentList from "./SegmentList";
import SentimentChart from "./SentimentChart";
import SimilarityChart from "./SimilarityChart";
import KeywordExplorer from "./KeywordExplorer";
import TimelineBar from "./TimelineBar";
import TranscriptView from "./TranscriptView";
import SemanticSearch from "./SemanticSearch";
import SegmentTimeline from "./SegmentTimeline";
import SegmentDetails from "./SegmentDetails";

// ─────────────────────────────────────────────
// API CLIENT
// ─────────────────────────────────────────────
const API_BASE_URL = "http://localhost:5000";

const api = {
  uploadFile: async (file) => {
    const formData = new FormData();
    formData.append("file", file);
    const resp = await fetch(`${API_BASE_URL}/api/upload`, {
      method: "POST",
      body: formData,
    });
    if (!resp.ok) {
      const errorData = await resp.json().catch(() => ({}));
      const errorMsg = errorData.error || `Upload failed: HTTP ${resp.status}`;
      throw new Error(errorMsg);
    }
    return resp.json();
  },
  ragQuery: async (query, segments = []) => {
    const resp = await fetch(`${API_BASE_URL}/api/rag`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, segments }),
    });
    if (!resp.ok) throw new Error(`RAG query failed: ${resp.status}`);
    return resp.json();
  },
  health: async () => {
    const resp = await fetch(`${API_BASE_URL}/health`);
    return resp.json();
  },
};

// ─────────────────────────────────────────────
// DATA NORMALIZATION (snake_case → camelCase)
// ─────────────────────────────────────────────
const normalizeSegment = (seg) => ({
  id: seg.id ?? 0,
  label: seg.label ?? seg.content?.slice(0, 50) ?? "Untitled",
  startTime: seg.start_time ?? seg.start ?? "00:00:00",
  endTime: seg.end_time ?? seg.end ?? "00:00:00",
  startTimestamp: seg.start_timestamp ?? seg.start ?? 0,
  endTimestamp: seg.end_timestamp ?? seg.end ?? 0,
  content: seg.content ?? "",
  wordCount: seg.word_count ?? 0,
  sentenceCount: seg.sentence_count ?? 1,
  duration: seg.duration ?? "0s",
  keywords: seg.keywords ?? [],
  sentiment: seg.sentiment ?? "neutral",
  sentimentScore: seg.sentiment_score ?? 0,
  confidence: seg.confidence ?? 0.5,
  summary: [
    seg.content?.slice(0, 100) ?? "",
  ].filter(Boolean),
});

const normalizeTranscript = (item, segments = []) => {
  const seg = segments.find(s => s.id === item.segment);
  return {
    id: item.id ?? 0,
    timestamp: item.timestamp ?? "00:00:00",
    start: item.start ?? 0,
    end: item.end ?? 0,
    text: item.text ?? "",
    segment: item.segment ?? 0,
    keywords: item.keywords ?? [],
  };
};

// ─────────────────────────────────────────────
// MOCK DATA (Fallback)
// ─────────────────────────────────────────────
const MOCK_TRANSCRIPT = [
  { id: 0, start: 0, end: 8.2, text: "Welcome to our quarterly earnings call. Today we'll discuss the financial performance and strategic direction of the company for the past quarter.", segment: 0, keywords: ["quarterly", "earnings", "financial", "performance"] },
  { id: 1, start: 8.2, end: 15.7, text: "Revenue grew by 23% year-over-year, driven primarily by strong adoption in enterprise markets and expansion of our cloud services division.", segment: 0, keywords: ["revenue", "enterprise", "cloud"] },
  { id: 2, start: 15.7, end: 24.1, text: "Our customer acquisition cost decreased by 18% while lifetime value increased significantly, reflecting the maturity of our go-to-market strategy.", segment: 0, keywords: ["customer", "acquisition", "lifetime"] },
  { id: 3, start: 24.1, end: 33.5, text: "Moving to operational metrics, our team has grown from 450 to 620 employees this quarter, with particular focus on engineering and product talent.", segment: 1, keywords: ["operational", "team", "engineering", "product"] },
  { id: 4, start: 33.5, end: 42.8, text: "We've successfully launched three major product updates, including the new AI-powered analytics dashboard which has seen overwhelming positive reception.", segment: 1, keywords: ["product", "AI", "analytics", "dashboard"] },
  { id: 5, start: 42.8, end: 51.3, text: "Infrastructure investments totaled 12 million dollars, with 60% directed toward improving system reliability and reducing latency by 40%.", segment: 1, keywords: ["infrastructure", "reliability", "latency"] },
  { id: 6, start: 51.3, end: 60.9, text: "Looking at our market positioning, we've made significant competitive advances in the mid-market segment, displacing legacy vendors in over 200 enterprise accounts.", segment: 2, keywords: ["market", "competitive", "enterprise", "segment"] },
  { id: 7, start: 60.9, end: 70.2, text: "Our strategic partnerships program has expanded to include 15 new technology integrations, broadening our ecosystem and reducing customer churn by 8%.", segment: 2, keywords: ["partnerships", "integrations", "ecosystem", "churn"] },
  { id: 8, start: 70.2, end: 80.5, text: "However, we must address some challenges. International expansion in APAC has been slower than projected, with regulatory hurdles in three key markets.", segment: 3, keywords: ["challenges", "international", "APAC", "regulatory"] },
  { id: 9, start: 80.5, end: 91.0, text: "Supply chain disruptions have impacted our hardware bundling offers, and we anticipate this to continue through the next two quarters.", segment: 3, keywords: ["supply", "chain", "hardware", "disruptions"] },
  { id: 10, start: 91.0, end: 100.8, text: "Despite these headwinds, our outlook remains optimistic. We're raising full-year guidance to 340 million in ARR, up from our previous 310 million estimate.", segment: 4, keywords: ["outlook", "guidance", "ARR", "optimistic"] },
  { id: 11, start: 100.8, end: 112.0, text: "We'll be investing heavily in R&D over the next 18 months, with a focus on next-generation machine learning capabilities and automated workflow tools.", segment: 4, keywords: ["R&D", "machine learning", "automated", "capabilities"] },
];

const MOCK_SEGMENTS = [
  { id: 0, label: "Financial Performance Overview", confidence: 0.94, sentiment: "positive", sentimentScore: 0.78, wordCount: 142, duration: "0:24", keywords: ["revenue", "earnings", "growth", "enterprise", "cloud", "acquisition"], summary: ["Revenue grew 23% YoY", "Customer acquisition cost down 18%", "LTV significantly increased", "Strong enterprise market adoption"] },
  { id: 1, label: "Operational & Team Metrics", confidence: 0.89, sentiment: "positive", sentimentScore: 0.65, wordCount: 198, duration: "0:27", keywords: ["team", "product", "engineering", "infrastructure", "analytics", "reliability"], summary: ["Team grew from 450 to 620 employees", "3 major product launches", "AI analytics dashboard well received", "$12M infrastructure investment"] },
  { id: 2, label: "Market Position & Partnerships", confidence: 0.91, sentiment: "positive", sentimentScore: 0.72, wordCount: 167, duration: "0:19", keywords: ["market", "competitive", "partnerships", "ecosystem", "enterprise", "churn"], summary: ["Strong mid-market advances", "Displaced legacy vendors in 200+ accounts", "15 new technology integrations", "Churn reduced by 8%"] },
  { id: 3, label: "Challenges & Headwinds", confidence: 0.87, sentiment: "negative", sentimentScore: -0.45, wordCount: 134, duration: "0:21", keywords: ["challenges", "APAC", "regulatory", "supply", "chain", "disruptions"], summary: ["APAC expansion below projections", "Regulatory hurdles in 3 key markets", "Hardware supply chain disruptions", "Issues expected to persist 2 quarters"] },
  { id: 4, label: "Forward Guidance & Strategy", confidence: 0.93, sentiment: "positive", sentimentScore: 0.69, wordCount: 156, duration: "0:21", keywords: ["guidance", "ARR", "R&D", "machine learning", "outlook", "investment"], summary: ["FY guidance raised to $340M ARR", "Previous estimate was $310M", "18-month R&D investment plan", "Focus on ML and automated workflows"] },
];

const MOCK_SIMILARITY_DATA = Array.from({ length: 112 }, (_, i) => {
  let base = 0.85;
  if (i >= 24 && i <= 26) base = 0.31;
  else if (i >= 50 && i <= 53) base = 0.28;
  else if (i >= 60 && i <= 63) base = 0.35;
  else if (i >= 80 && i <= 83) base = 0.22;
  else if (i >= 100 && i <= 103) base = 0.38;
  const noise = (Math.random() - 0.5) * 0.12;
  const sentiment = Math.sin(i * 0.15) * 0.3 + (i > 70 && i < 90 ? -0.4 : 0.2) + (Math.random() - 0.5) * 0.2;
  return { t: i, similarity: Math.max(0.1, Math.min(1, base + noise)), sentiment: Math.max(-1, Math.min(1, sentiment)) };
});

const SEGMENT_BOUNDARIES = [24, 51, 60, 80, 100];

const MOCK_PERF = {
  model: "whisper-medium.en",
  embeddingDim: 384,
  runtime: { preprocessing: 1.2, transcription: 4.2, embedding: 1.8, segmentation: 0.4, sentiment: 2.1 },
  mode: "GPU",
  memory: "3.2 GB",
  accuracy: 0.912,
  stability: 0.887,
};

// ─────────────────────────────────────────────
// PIPELINE STAGES
// ─────────────────────────────────────────────
const PIPELINE_STAGES = [
  { id: "audio", label: "Audio Ingestion", icon: "🎵", detail: "PCM decode, 16kHz mono normalization, VAD filtering" },
  { id: "transcription", label: "Transcription", icon: "📝", detail: "Whisper-large-v3 with word-level timestamps" },
  { id: "embedding", label: "Embedding", icon: "🧠", detail: "sentence-transformers/all-mpnet-base-v2, 768-dim vectors" },
  { id: "similarity", label: "Similarity", icon: "📐", detail: "Cosine similarity between adjacent sentences" },
  { id: "segmentation", label: "Segmentation", icon: "✂️", detail: "Valley-peak detection with adaptive threshold θ=0.38" },
  { id: "sentiment", label: "Sentiment Analysis", icon: "💬", detail: "cardiffnlp/twitter-roberta-base-sentiment, per-sentence" },
  { id: "keywords", label: "Keyword Extraction", icon: "🔑", detail: "KeyBERT with MMR diversity, top-k=8 per segment" },
  { id: "rag", label: "RAG Indexing", icon: "🗂", detail: "FAISS index with ChromaDB persistence, cosine retrieval" },
];

// ─────────────────────────────────────────────
// UTILS
// ─────────────────────────────────────────────
const sentimentColor = (s) => {
  if (typeof s === "string") {
    if (s === "positive") return "#22d3ee";
    if (s === "negative") return "#f87171";
    return "#a78bfa";
  }
  if (s > 0.3) return "#22d3ee";
  if (s < -0.3) return "#f87171";
  return "#a78bfa";
};

const fmtTime = (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;

// Export transcript as TXT file
const exportTranscriptAsText = (transcript = [], segments = [], filename = "") => {
  let textContent = "============================================================\n";
  textContent += "PODCAST TRANSCRIPT\n";
  textContent += "============================================================\n\n";

  // Group transcript by segments
  const groupedBySegment = {};
  transcript.forEach(item => {
    const segId = item.segment || 0;
    if (!groupedBySegment[segId]) groupedBySegment[segId] = [];
    groupedBySegment[segId].push(item);
  });

  // Create segment map
  const segmentMap = {};
  segments.forEach(seg => {
    segmentMap[seg.id] = seg;
  });

  // Build transcript text
  Object.entries(groupedBySegment).forEach(([segId, items]) => {
    const segment = segmentMap[parseInt(segId)];
    const startTime = fmtTime(segment?.startTimestamp || segment?.startTime || 0);
    const endTime = fmtTime(segment?.endTimestamp || segment?.endTime || 0);

    textContent += `Segment ${segId}\n`;
    textContent += `Time: ${startTime} → ${endTime}\n`;
    textContent += `Sentiment: ${segment?.sentiment || "neutral"}\n\n`;

    // Add transcript sentences
    items.forEach(item => {
      textContent += `${item.text}\n`;
    });

    // Add keywords
    if (segment?.keywords && segment.keywords.length > 0) {
      textContent += `\nKeywords: ${segment.keywords.join(", ")}\n`;
    }

    textContent += "\n" + "------------------------------------------------------------\n\n";
  });

  // Trigger download
  const blob = new Blob([textContent], { type: "text/plain" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  const timestamp = new Date().toISOString().split("T")[0];
  link.href = url;
  link.download = `podcast_transcript_${timestamp}.txt`;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
};

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
// ─────────────────────────────────────────────
const UploadZone = ({ onUpload, selectedModel = "medium.en" }) => {
  const [dragging, setDragging] = useState(false);
  const [file, setFile] = useState(null);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState("");
  const [uploading, setUploading] = useState(false);
  const inputRef = useRef();

  const handleFile = (f) => {
    if (!f) return;
    const validExts = ['.mp3', '.wav', '.m4a', '.flac', '.ogg', '.webm'];
    const ext = '.' + f.name.split('.').pop().toLowerCase();
    if (!validExts.includes(ext)) {
      setError(`Only ${validExts.join(', ')} files are supported.`);
      return;
    }
    if (f.size > 100 * 1024 * 1024) { setError("File exceeds 100MB limit."); return; }
    setError(""); setFile(f);
  };

  const startUpload = async () => {
    if (!file) return;
    setUploading(true);
    setError("");
    setProgress(0);
    
    try {
      // Simulate initial progress
      let p = 10;
      const progressInterval = setInterval(() => {
        if (p < 90) {
          p += Math.random() * 20;
          setProgress(Math.min(p, 90));
        }
      }, 200);

      // Call backend API with model parameter
      const formData = new FormData();
      formData.append("file", file);
      formData.append("model", selectedModel);

      const resp = await fetch(`${API_BASE_URL}/api/upload`, {
        method: "POST",
        body: formData,
      });

      if (!resp.ok) {
        const errorData = await resp.json().catch(() => ({}));
        const errorMsg = errorData.error || `Upload failed: HTTP ${resp.status}`;
        throw new Error(errorMsg);
      }

      const result = await resp.json();
      clearInterval(progressInterval);
      setProgress(100);

      // Pass result to parent with normalized data
      const segments = (result.segments || []).map(normalizeSegment);
      const transcript = (result.transcript || []).map(item => normalizeTranscript(item, segments));
      
      setTimeout(() => {
        onUpload({
          filename: file.name,
          transcript,
          segments,
          performance: result.performance || {},
          summary: result.summary || {},
        });
      }, 300);
    } catch (err) {
      setError(`Upload failed: ${err.message}`);
      setUploading(false);
      setProgress(0);
    }
  };

  return (
    <div className="max-w-2xl mx-auto">
      <div onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => { e.preventDefault(); setDragging(false); handleFile(e.dataTransfer.files[0]); }}
        onClick={() => inputRef.current.click()}
        className={`relative rounded-2xl border-2 border-dashed p-12 text-center cursor-pointer transition-all duration-300
          ${dragging ? "border-cyan-400 bg-cyan-400/10" : "border-slate-600 hover:border-slate-400 bg-slate-800/40"}`}>
        <input ref={inputRef} type="file" accept=".mp3,.wav,.m4a,.flac,.ogg,.webm" className="hidden" onChange={(e) => handleFile(e.target.files[0])} />
        <div className="text-5xl mb-4">🎙️</div>
        <div className="text-slate-200 text-lg font-semibold mb-2">
          {file ? file.name : "Drop your audio file here"}
        </div>
        <div className="text-slate-500 text-sm">
          {file ? `${(file.size / 1024 / 1024).toFixed(2)} MB · MP3/WAV/M4A/FLAC/OGG/WebM` : "MP3, WAV, M4A, FLAC, OGG, WebM · Max 100MB"}
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
            <span>Uploading & Processing...</span><span>{Math.floor(progress)}%</span>
          </div>
          <div className="h-2 bg-slate-700 rounded-full overflow-hidden">
            <div className="h-full bg-gradient-to-r from-cyan-500 to-indigo-500 rounded-full transition-all duration-200"
              style={{ width: `${progress}%` }} />
          </div>
        </div>
      )}
    </div>
  );
};

// ─────────────────────────────────────────────
// PIPELINE VIEW
// ─────────────────────────────────────────────
const PipelineView = ({ onComplete }) => {
  const [stages, setStages] = useState(PIPELINE_STAGES.map((s) => ({ ...s, status: "pending", time: null })));
  const [expanded, setExpanded] = useState(null);
  const current = useRef(0);

  useEffect(() => {
    const run = () => {
      if (current.current >= PIPELINE_STAGES.length) { setTimeout(onComplete, 600); return; }
      const idx = current.current;
      setStages((prev) => prev.map((s, i) => i === idx ? { ...s, status: "running" } : s));
      const dur = 800 + Math.random() * 1200;
      setTimeout(() => {
        const t = (dur / 1000).toFixed(1) + "s";
        setStages((prev) => prev.map((s, i) => i === idx ? { ...s, status: "completed", time: t } : s));
        current.current++;
        setTimeout(run, 200);
      }, dur);
    };
    setTimeout(run, 400);
  }, []);

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
                      {[0, 1, 2].map(d => <span key={d} className="w-1 h-1 rounded-full bg-amber-400 animate-bounce" style={{ animationDelay: `${d * 0.15}s` }} />)}
                    </span>
                  )}
                </div>
              </div>
              <div className="flex items-center gap-4">
                {s.time && <span className="text-slate-500 text-xs font-mono">{s.time}</span>}
                <span className="text-xs font-mono px-2 py-0.5 rounded" style={{ color: statusColor[s.status], background: statusColor[s.status] + "22" }}>
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
const ChartTooltip = ({ active, payload, label, transcript = MOCK_TRANSCRIPT }) => {
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
      {sentence && <div className="mt-2 text-slate-500 border-t border-slate-700 pt-2 leading-relaxed line-clamp-2">{sentence.text.slice(0, 80)}...</div>}
    </div>
  );
};

// ─────────────────────────────────────────────
// VISUALIZATION CENTER PANEL
// ─────────────────────────────────────────────
const VisualizationPanel = ({ activeSegment, onTimeClick, similarityData = MOCK_SIMILARITY_DATA, boundaries = SEGMENT_BOUNDARIES, transcript = MOCK_TRANSCRIPT }) => {
  const [view, setView] = useState("combined");

  const views = [
    { id: "similarity", label: "Similarity" },
    { id: "sentiment", label: "Sentiment" },
    { id: "combined", label: "Combined" },
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
            {boundaries.map(b => (
              <ReferenceLine key={b} x={b} stroke="#6366f1" strokeDasharray="4 4" strokeOpacity={0.7} label={{ value: "▼", fill: "#6366f1", fontSize: 10 }} />
            ))}
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
        <span className="flex items-center gap-1.5"><span className="w-3 h-0.5 bg-indigo-400 inline-block border-dashed" style={{ borderTop: "2px dashed" }} />Segment boundary</span>
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
        const hl = highlightedKeyword === kw;
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
// ─────────────────────────────────────────────
const SegmentModal = ({ segment, onClose, onKeywordClick, highlightedKeyword }) => {
  if (!segment) return null;
  const sc = sentimentColor(segment.sentiment);
  const [cpd, setCpd] = useState(false);

  const exportText = () => {
    const text = `Topic: ${segment.label}\nSentiment: ${segment.sentiment}\nConfidence: ${(segment.confidence * 100).toFixed(0)}%\n\nSummary:\n${segment.summary.map(s => `• ${s}`).join("\n")}\n\nKeywords: ${segment.keywords.join(", ")}`;
    navigator.clipboard?.writeText(text);
    setCpd(true); setTimeout(() => setCpd(false), 1500);
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/80 backdrop-blur-sm" onClick={onClose}>
      <div className="bg-slate-900 border border-slate-700 rounded-2xl w-full max-w-lg mx-4 shadow-2xl" onClick={(e) => e.stopPropagation()}>
        <div className="p-6 border-b border-slate-800 flex items-start justify-between">
          <div>
            <div className="flex items-center gap-2 mb-2">
              <Badge color="#6366f1">SEG {segment.id}</Badge>
              <Badge color={sc}>{segment.sentiment.toUpperCase()}</Badge>
            </div>
            <h3 className="text-slate-100 text-lg font-semibold">{segment.label}</h3>
          </div>
          <button onClick={onClose} className="text-slate-500 hover:text-slate-300 text-xl leading-none mt-1">✕</button>
        </div>

        <div className="p-6 space-y-5">
          <div className="grid grid-cols-3 gap-3">
            <StatCard label="Confidence" value={`${(segment.confidence * 100).toFixed(0)}%`} />
            <StatCard label="Words" value={segment.wordCount} />
            <StatCard label="Duration" value={segment.duration} />
          </div>

          <div>
            <div className="text-slate-400 text-xs font-mono uppercase tracking-widest mb-3">Key Insights</div>
            <ul className="space-y-2">
              {segment.summary.map((s, i) => (
                <li key={i} className="flex items-start gap-2 text-slate-300 text-sm">
                  <span className="text-cyan-400 mt-0.5">▸</span>{s}
                </li>
              ))}
            </ul>
          </div>

          <div>
            <div className="text-slate-400 text-xs font-mono uppercase tracking-widest mb-2">Keywords</div>
            <KeywordCloud keywords={segment.keywords} onKeywordClick={onKeywordClick} highlightedKeyword={highlightedKeyword} />
          </div>

          <div>
            <div className="text-slate-400 text-xs font-mono uppercase tracking-widest mb-2">Sentiment Score</div>
            <div className="h-2 bg-slate-700 rounded-full overflow-hidden">
              <div className="h-full rounded-full transition-all" style={{ width: `${Math.abs(segment.sentimentScore) * 100}%`, background: sc }} />
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
          <button className="flex-1 py-2 rounded-xl text-sm bg-indigo-600 hover:bg-indigo-500 text-white transition-all font-medium">
            Export JSON {/* TODO: connect to backend export API */}
          </button>
        </div>
      </div>
    </div>
  );
};

// ─────────────────────────────────────────────
// TRANSCRIPT PANEL
// ─────────────────────────────────────────────
const TranscriptPanel = ({ activeSegment, highlightedKeyword, currentTime, transcript = MOCK_TRANSCRIPT }) => {
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
      <div className="flex-1 overflow-y-auto space-y-2 pr-1 scrollbar-thin scrollbar-track-slate-800 scrollbar-thumb-slate-600">
        {transcript.map((s) => {
          const isActive = s.segment === activeSegment;
          const hasKw = highlightedKeyword && s.keywords?.includes(highlightedKeyword);
          const borderColor = isActive ? "#6366f1" : hasKw ? "#22d3ee" : "transparent";
          return (
            <div key={s.id} ref={(el) => ref.current[s.id] = el}
              className={`rounded-xl p-3 border-l-2 text-sm leading-relaxed transition-all duration-200
                ${isActive ? "bg-indigo-500/10 text-slate-200" : "bg-slate-800/30 text-slate-400 hover:bg-slate-800/60"}`}
              style={{ borderColor }}>
              <span className="text-slate-600 text-xs font-mono mr-2">{fmtTime(s.start)}</span>
              {s.text.split(" ").map((word, wi) => {
                const w = word.replace(/[^a-z]/gi, "").toLowerCase();
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
// ─────────────────────────────────────────────
const SegmentsPanel = ({ activeSegment, onSegmentClick, onModalOpen, highlightedKeyword, onKeywordClick, segments = MOCK_SEGMENTS }) => {
  const sentimentBg = { positive: "text-cyan-400", negative: "text-red-400", neutral: "text-purple-400" };

  return (
    <div className="flex flex-col h-full">
      <div className="text-slate-400 text-xs font-mono uppercase tracking-widest mb-4">
        Segments · {segments.length}
      </div>
      <div className="flex-1 overflow-y-auto space-y-3 pr-1">
        {segments.map((seg) => {
          const isActive = seg.id === activeSegment;
          const sc = sentimentColor(seg.sentiment);
          return (
            <div key={seg.id}
              onClick={() => onSegmentClick(seg.id)}
              className={`rounded-2xl border p-4 cursor-pointer transition-all duration-200
                ${isActive ? "border-indigo-500/70 bg-indigo-500/10" : "border-slate-700/50 bg-slate-800/30 hover:border-slate-600"}`}>
              <div className="flex items-start justify-between gap-2 mb-2">
                <div className="flex items-center gap-2 flex-wrap">
                  <Badge color="#6366f1">#{seg.id}</Badge>
                  <Badge color={sc}>{seg.sentiment}</Badge>
                </div>
                <span className={`text-xs font-bold font-mono ${sentimentBg[seg.sentiment] || sentimentBg["neutral"]}`}>
                  {(seg.confidence * 100).toFixed(0)}%
                </span>
              </div>
              <div className="text-slate-200 text-sm font-medium mb-2">{seg.label}</div>
              <div className="flex gap-4 text-xs text-slate-500 font-mono mb-3">
                <span>{seg.wordCount}w</span>
                <span>{seg.duration}</span>
              </div>
              <div className="flex gap-2 mt-3">
                <button onClick={(e) => { e.stopPropagation(); onModalOpen(seg); }}
                  className="flex-1 text-xs py-1.5 rounded-lg bg-slate-700/60 hover:bg-slate-600 text-slate-300 transition-all">
                  Summary
                </button>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};

// ─────────────────────────────────────────────
// RAG Q&A PANEL
// ─────────────────────────────────────────────
const RAGPanel = ({ onHighlightSegment, segments = [] }) => {
  const [query, setQuery] = useState("");
  const [response, setResponse] = useState(null);
  const [loading, setLoading] = useState(false);
  const [streaming, setStreaming] = useState("");
  const streamRef = useRef(null);

  const ask = async () => {
    if (!query.trim()) return;
    setLoading(true);
    setResponse(null);
    setStreaming("");

    try {
      // Denormalize segments back to snake_case for backend
      const denormSegments = segments.map(seg => ({
        id: seg.id,
        content: seg.content,
        start_timestamp: seg.startTimestamp,
        end_timestamp: seg.endTimestamp,
        sentiment: seg.sentiment,
        keywords: seg.keywords,
      }));
      
      const result = await api.ragQuery(query, denormSegments);
      
      setLoading(false);
      
      // Stream in the answer
      let i = 0;
      const text = result.answer || "";
      clearInterval(streamRef.current);
      streamRef.current = setInterval(() => {
        i += 3;
        setStreaming(text.slice(0, i));
        if (i >= text.length) {
          clearInterval(streamRef.current);
          setResponse(result);
          setStreaming("");
        }
      }, 18);
    } catch (err) {
      setLoading(false);
      setResponse({
        answer: `Error: ${err.message}. Make sure the backend is running on http://localhost:5000`,
        referenced_segments: [],
        model: "error",
        context_length: 0,
      });
    }
  };

  const suggestions = ["What were the main topics?", "What is the sentiment?", "Summarize the content"];

  return (
    <div className="bg-slate-800/40 rounded-2xl border border-slate-700/50 p-6">
      <div className="flex items-center gap-3 mb-5">
        <div className="w-8 h-8 rounded-lg bg-indigo-500/20 flex items-center justify-center text-sm">🤖</div>
        <div>
          <h3 className="text-slate-200 text-sm font-semibold">Ask About This Audio</h3>
          <p className="text-slate-500 text-xs">RAG-powered Q&A with segment references</p>
        </div>
      </div>

      <div className="flex gap-2 mb-3">
        <input value={query} onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && ask()}
          placeholder="Ask anything about the audio content..."
          className="flex-1 bg-slate-900/60 border border-slate-600 rounded-xl px-4 py-2.5 text-slate-200 text-sm placeholder:text-slate-600 focus:outline-none focus:border-indigo-500 transition-colors" />
        <button onClick={ask} disabled={loading}
          className="px-5 py-2.5 rounded-xl bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-white text-sm font-medium transition-all">
          {loading ? "..." : "Ask"}
        </button>
      </div>

      <div className="flex flex-wrap gap-2 mb-5">
        {suggestions.map(s => (
          <button key={s} onClick={() => { setQuery(s); }}
            className="text-xs text-slate-400 border border-slate-700 rounded-lg px-3 py-1 hover:border-slate-500 hover:text-slate-200 transition-all">
            {s}
          </button>
        ))}
      </div>

      {(streaming || response) && (
        <div className="bg-slate-900/60 rounded-xl p-4 border border-slate-700/50">
          <p className="text-slate-300 text-sm leading-relaxed">
            {streaming || response?.answer}
            {streaming && <span className="inline-block w-0.5 h-4 bg-cyan-400 ml-0.5 animate-pulse" />}
          </p>
          {response && response.referenced_segments && (
            <div className="mt-4 pt-4 border-t border-slate-700/50 flex flex-wrap gap-3 items-center">
              <div className="flex gap-2 flex-wrap">
                <span className="text-slate-500 text-xs">Referenced segments:</span>
                {(response.referenced_segments || []).map(r => (
                  <button key={r} onClick={() => onHighlightSegment?.(r)}
                    className="text-xs px-2 py-0.5 rounded-full border border-indigo-500/50 text-indigo-400 hover:bg-indigo-500/20 transition-all">
                    SEG {r}
                  </button>
                ))}
              </div>
              {response.model && response.model !== "error" && (
                <div className="ml-auto flex items-center gap-1.5 text-xs text-slate-500">
                  <span>Model:</span>
                  <span className="text-cyan-400 font-mono">{response.model}</span>
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
};

// ─────────────────────────────────────────────
// PERFORMANCE DASHBOARD
// ─────────────────────────────────────────────
const PerformanceDashboard = ({ performance = MOCK_PERF }) => {
  const [open, setOpen] = useState(false);
  
  const stages = performance?.stages || MOCK_PERF.runtime;
  const runtimeData = Object.entries(stages).map(([k, v]) => ({ stage: k, ms: v * 1000 }));
  const totalRuntime = performance?.runtime_seconds || Object.values(stages).reduce((a, b) => a + b, 0);

  return (
    <div className="bg-slate-800/40 rounded-2xl border border-slate-700/50 overflow-hidden">
      <button onClick={() => setOpen(!open)}
        className="w-full flex items-center justify-between p-4 hover:bg-slate-700/20 transition-all">
        <div className="flex items-center gap-3">
          <span className="text-sm">⚡</span>
          <span className="text-slate-300 text-sm font-medium">Performance Dashboard</span>
          <Badge color="#22d3ee">{performance?.mode || "GPU"}</Badge>
        </div>
        <span className="text-slate-500 text-xs">{open ? "▲" : "▼"}</span>
      </button>

      {open && (
        <div className="p-5 border-t border-slate-700/50 space-y-5">
          <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
            <StatCard label="Total Time" value={`${totalRuntime.toFixed(1)}s`} sub="End-to-end" />
            <StatCard label="Stages" value={Object.keys(stages).length} sub="Pipeline steps" />
            <StatCard label="Runtime" value={`${totalRuntime.toFixed(1)}s`} sub="Execution time" />
          </div>

          {runtimeData.length > 0 && (
            <div className="bg-slate-800/60 rounded-xl p-4 border border-slate-700/50">
              <div className="text-slate-400 text-xs font-mono uppercase tracking-widest mb-3">Runtime Per Stage (milliseconds)</div>
              <ResponsiveContainer width="100%" height={120}>
                <BarChart data={runtimeData} margin={{ top: 0, right: 0, left: -20, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                  <XAxis dataKey="stage" tick={{ fill: "#64748b", fontSize: 9 }} />
                  <YAxis tick={{ fill: "#64748b", fontSize: 9 }} />
                  <Tooltip contentStyle={{ background: "#0f172a", border: "1px solid #334155", borderRadius: 8, fontSize: 11 }} />
                  <Bar dataKey="ms" fill="#6366f1" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

// ─────────────────────────────────────────────
// ADVANCED CONTROLS
// ─────────────────────────────────────────────
const AdvancedControls = ({ onReset, transcript = [], segments = [], theme, onThemeToggle, onModelChange, selectedModel = "medium.en" }) => {
  const [threshold, setThreshold] = useState(0.38);

  const models = [
    { value: "tiny.en", label: "Tiny (Fast)" },
    { value: "base.en", label: "Base (Balanced)" },
    { value: "small.en", label: "Small (Good)" },
    { value: "medium.en", label: "Medium (Better)" },
    { value: "large-v3", label: "Large v3 (Best)" },
  ];

  const handleModelChange = (e) => {
    onModelChange?.(e.target.value);
  };

  const handleExportTxt = () => {
    exportTranscriptAsText(transcript, segments);
  };

  return (
    <div className="bg-slate-800/40 rounded-2xl border border-slate-700/50 p-5">
      <div className="text-slate-400 text-xs font-mono uppercase tracking-widest mb-4">Controls</div>
      <div className="space-y-4">
        <div>
          <label className="text-slate-400 text-xs mb-2 block">Transcription Model</label>
          <select value={selectedModel} onChange={handleModelChange}
            className="w-full bg-slate-900 border border-slate-600 rounded-lg px-3 py-2 text-slate-300 text-sm focus:outline-none focus:border-indigo-500">
            {models.map(m => (
              <option key={m.value} value={m.value}>{m.label}</option>
            ))}
          </select>
          <div className="text-xs text-slate-500 mt-1">Selected: {selectedModel}</div>
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
          <button className="py-2 text-xs rounded-lg bg-slate-700/60 hover:bg-slate-600 text-slate-300 transition-all border border-slate-600">
            Re-run Segmentation {/* TODO: POST /api/segment/rerun */}
          </button>
          <button className="py-2 text-xs rounded-lg bg-slate-700/60 hover:bg-slate-600 text-slate-300 transition-all border border-slate-600">
            Export JSON {/* TODO: GET /api/export?format=json */}
          </button>
          <button onClick={handleExportTxt} className="py-2 text-xs rounded-lg bg-indigo-600/40 hover:bg-indigo-600/60 text-indigo-300 transition-all border border-indigo-500/30">
            Export TXT ✓
          </button>
          <button onClick={onReset}
            className="py-2 text-xs rounded-lg bg-red-500/20 hover:bg-red-500/30 text-red-400 transition-all border border-red-500/30">
            Reset Session
          </button>
        </div>
      </div>
    </div>
  );
};

// ─────────────────────────────────────────────
// MAIN ANALYSIS VIEW (REDESIGNED 2-COLUMN LAYOUT)
// ─────────────────────────────────────────────
const AnalysisView = ({ filename, transcript = [], segments = [], performance = {}, onReset, selectedModel = "medium.en", onModelChange }) => {
  const [activeSegment, setActiveSegment] = useState(0);
  const [highlightedKeyword, setHighlightedKeyword] = useState(null);
  const [modalSegment, setModalSegment] = useState(null);
  const [currentTime, setCurrentTime] = useState(null);

  // Use real data if available, fallback to mock
  const displayTranscript = transcript.length > 0 ? transcript : MOCK_TRANSCRIPT;
  const displaySegments = segments.length > 0 ? segments : MOCK_SEGMENTS;

  return (
    <div className="flex flex-col gap-6 h-full">
      {/* Dashboard header */}
      <div className="flex-shrink-0">
        <h1 className="text-4xl font-bold text-slate-100 mb-2">
          Podcast Semantic Segmentation Dashboard
        </h1>
        <div className="flex items-center justify-between">
          <div>
            <p className="text-slate-400 text-sm">
              <span className="font-semibold text-slate-300">{filename}</span>
              {" · "}
              <span>{displayTranscript.length} sentences</span>
              {" · "}
              <span>{displaySegments.length} segments</span>
              {performance?.runtime_seconds && ` · ${performance.runtime_seconds.toFixed(1)}s`}
            </p>
          </div>
          <div className="flex gap-2">
            <Badge color="#06b6d4">LIVE</Badge>
            <Badge color="#6366f1">ANALYSIS</Badge>
          </div>
        </div>
      </div>

      {/* Segment Timeline - PRIMARY NAVIGATION */}
      <div className="flex-shrink-0">
        <SegmentTimeline
          segments={displaySegments}
          activeSegment={activeSegment}
          onSegmentClick={setActiveSegment}
        />
      </div>

      {/* Main 2-column grid layout: 40% left, 60% right */}
      <div className="grid grid-cols-12 gap-6 flex-1 min-h-0">
        {/* LEFT PANEL (40%) - Segment Details & Keywords */}
        <div className="col-span-12 lg:col-span-5 flex flex-col gap-4 min-h-0">
          {/* Segment Details Panel - ONLY shows active segment */}
          <div className="flex-1 min-h-0 overflow-hidden">
            <SegmentDetails
              segments={displaySegments}
              activeSegment={activeSegment}
              onSegmentChange={setActiveSegment}
              onKeywordClick={setHighlightedKeyword}
              highlightedKeyword={highlightedKeyword}
            />
          </div>

          {/* Keywords Explorer */}
          <div className="bg-slate-800/30 rounded-2xl border border-slate-700/50 p-5">
            <KeywordExplorer
              segments={displaySegments}
              onKeywordClick={setHighlightedKeyword}
              highlightedKeyword={highlightedKeyword}
            />
          </div>
        </div>

        {/* RIGHT PANEL (60%) - Analytics Charts */}
        <div className="col-span-12 lg:col-span-7 flex flex-col gap-4 min-h-0">
          {/* Charts row */}
          <div className="grid grid-cols-2 gap-4">
            <div className="bg-slate-800/30 rounded-2xl border border-slate-700/50 p-5">
              <SentimentChart segments={displaySegments} />
            </div>
            <div className="bg-slate-800/30 rounded-2xl border border-slate-700/50 p-5">
              <SimilarityChart transcript={displayTranscript} segments={displaySegments} />
            </div>
          </div>

          {/* Semantic search */}
          <div className="bg-slate-800/30 rounded-2xl border border-slate-700/50 p-5">
            <SemanticSearch segments={displaySegments} onHighlightSegment={setActiveSegment} />
          </div>
        </div>
      </div>

      {/* Full-width transcript view below */}
      <div className="flex-1 min-h-0 bg-slate-800/30 rounded-2xl border border-slate-700/50 p-5">
        <TranscriptView
          transcript={displayTranscript}
          segments={displaySegments}
          activeSegment={activeSegment}
          highlightedKeyword={highlightedKeyword}
          currentTime={currentTime}
        />
      </div>

      {/* Bottom controls */}
      <div className="flex-shrink-0 grid grid-cols-1 lg:grid-cols-2 gap-4">
        <PerformanceDashboard performance={performance} />
        <AdvancedControls 
          onReset={onReset} 
          selectedModel={selectedModel}
          onModelChange={onModelChange}
          transcript={displayTranscript}
          segments={displaySegments}
        />
      </div>

      {modalSegment && (
        <SegmentModal
          segment={modalSegment}
          onClose={() => setModalSegment(null)}
          onKeywordClick={(kw) => {
            setHighlightedKeyword(kw);
            setModalSegment(null);
          }}
          highlightedKeyword={highlightedKeyword}
        />
      )}
    </div>
  );
};

// ─────────────────────────────────────────────
// ROOT APP
// ─────────────────────────────────────────────
const PHASES = { UPLOAD: "upload", PIPELINE: "pipeline", ANALYSIS: "analysis" };

export default function App() {
  const [phase, setPhase] = useState(PHASES.UPLOAD);
  const [filename, setFilename] = useState("");
  const [analysisData, setAnalysisData] = useState({});
  const [selectedModel, setSelectedModel] = useState("medium.en");

  const handleUpload = (data) => {
    setFilename(data.filename);
    setAnalysisData(data);
    setPhase(PHASES.PIPELINE);
  };

  const handleReset = () => {
    setPhase(PHASES.UPLOAD);
    setFilename("");
    setAnalysisData({});
  };

  const handleModelChange = (newModel) => {
    setSelectedModel(newModel);
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-200 flex flex-col" style={{ fontFamily: "'DM Mono', 'Fira Code', 'Cascadia Code', monospace" }}>
      {/* Top nav */}
      <nav className="border-b border-slate-800/80 bg-slate-950/80 backdrop-blur sticky top-0 z-40 flex-shrink-0">
        <div className="max-w-screen-2xl w-full mx-auto px-6 h-14 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-7 h-7 rounded-lg bg-gradient-to-br from-cyan-500 to-indigo-600 flex items-center justify-center text-xs">🔊</div>
            <span className="text-slate-100 font-semibold text-sm tracking-tight">AudioIntel</span>
            <span className="text-slate-600 text-xs hidden sm:block">· AI Audio Intelligence Platform</span>
          </div>
          <div className="flex items-center gap-4">
            {phase !== PHASES.UPLOAD && (
              <div className="flex gap-2">
                {[PHASES.UPLOAD, PHASES.PIPELINE, PHASES.ANALYSIS].map((p, i) => (
                  <div key={p} className={`w-2 h-2 rounded-full transition-all ${phase === p ? "bg-cyan-400 scale-125" : Object.values(PHASES).indexOf(phase) > i ? "bg-indigo-500" : "bg-slate-700"}`} />
                ))}
              </div>
            )}
            <div className="text-xs text-slate-600 font-mono hidden md:block">v2.5.0</div>
          </div>
        </div>
      </nav>

      {/* Main content */}
      <main className={`flex-1 flex flex-col overflow-hidden ${phase === PHASES.ANALYSIS ? "" : "max-w-screen-2xl mx-auto px-4 sm:px-6 py-8"}`}>
        {phase === PHASES.UPLOAD && (
          <div className="max-w-screen-2xl mx-auto w-full px-4 sm:px-6 py-8">
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
            <UploadZone onUpload={handleUpload} selectedModel={selectedModel} />
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 max-w-2xl mx-auto mt-12">
              {["Speech-to-Text", "Semantic Segmentation", "Sentiment Analysis", "RAG Q&A"].map((f) => (
                <div key={f} className="text-center p-4 rounded-xl bg-slate-800/30 border border-slate-700/50">
                  <div className="text-slate-300 text-xs font-medium">{f}</div>
                </div>
              ))}
            </div>
          </div>
        )}

        {phase === PHASES.PIPELINE && (
          <div className="max-w-screen-2xl mx-auto w-full px-4 sm:px-6 py-8">
            <PipelineView onComplete={() => setPhase(PHASES.ANALYSIS)} />
          </div>
        )}

        {phase === PHASES.ANALYSIS && (
          <div className="flex-1 flex flex-col overflow-hidden px-4 sm:px-6 py-6">
            <div className="max-w-screen-2xl w-full mx-auto h-full flex flex-col gap-6 overflow-hidden">
              <AnalysisView 
                filename={filename} 
                transcript={analysisData?.transcript || []}
                segments={analysisData?.segments || []}
                performance={analysisData?.performance || {}}
                onReset={handleReset}
                selectedModel={selectedModel}
                onModelChange={handleModelChange}
              />
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
