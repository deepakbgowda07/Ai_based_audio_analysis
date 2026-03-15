import React, { useState } from "react";

const sentimentColor = (s) => {
  if (typeof s === "string") {
    if (s === "positive") return "#10b981";
    if (s === "negative") return "#ef4444";
    return "#6b7280";
  }
  if (s > 0.3) return "#10b981";
  if (s < -0.3) return "#ef4444";
  return "#6b7280";
};

const fmtTime = (s) => {
  const seconds = typeof s === "number" ? s : parseFloat(s);
  const m = Math.floor(seconds / 60);
  const sec = Math.floor(seconds % 60);
  return `${m}:${String(sec).padStart(2, "0")}`;
};

export default function SegmentList({ segments = [], activeSegment, onSegmentClick, onSegmentModal }) {
  const [expandedId, setExpandedId] = useState(null);

  return (
    <div className="flex flex-col h-full">
      <div className="text-slate-400 text-xs font-mono uppercase tracking-widest mb-4 flex items-center justify-between">
        <span>Episode Segments</span>
        <span className="text-slate-600">{segments.length} topics</span>
      </div>
      <div className="flex-1 overflow-y-auto space-y-3 pr-2">
        {segments.map((seg) => {
          const isActive = seg.id === activeSegment;
          const sc = sentimentColor(seg.sentiment);
          const startTime = fmtTime(seg.startTimestamp || seg.startTime || 0);
          const endTime = fmtTime(seg.endTimestamp || seg.endTime || 0);

          return (
            <div
              key={seg.id}
              onClick={() => onSegmentClick(seg.id)}
              className={`rounded-xl border p-4 cursor-pointer transition-all duration-200
                ${isActive 
                  ? "border-indigo-500/70 bg-indigo-500/10 shadow-lg shadow-indigo-500/20" 
                  : "border-slate-700/50 bg-slate-800/30 hover:border-slate-600/70 hover:bg-slate-800/50"
              }`}
            >
              {/* Segment header */}
              <div className="flex items-start justify-between gap-2 mb-2">
                <div className="flex items-center gap-2 flex-wrap">
                  <span className="inline-block px-2.5 py-0.5 rounded-full bg-indigo-500/20 border border-indigo-500/40 text-indigo-400 text-xs font-bold">
                    #{seg.id}
                  </span>
                  <span 
                    className="inline-block px-2.5 py-0.5 rounded-full border text-xs font-semibold text-white"
                    style={{ background: sc + "22", color: sc, borderColor: sc + "44" }}
                  >
                    {seg.sentiment?.charAt(0).toUpperCase() + seg.sentiment?.slice(1)}
                  </span>
                </div>
                <span className="text-xs font-bold text-cyan-400 font-mono">
                  {(seg.confidence * 100).toFixed(0)}%
                </span>
              </div>

              {/* Segment title */}
              <div className="text-slate-200 text-sm font-semibold mb-2 truncate">
                {seg.label || `Segment ${seg.id}`}
              </div>

              {/* Time range */}
              <div className="text-xs text-slate-400 font-mono mb-3">
                {startTime} → {endTime}
              </div>

              {/* Keywords preview */}
              <div className="mb-3">
                <div className="flex gap-1.5 flex-wrap">
                  {(seg.keywords || []).slice(0, 4).map((kw, i) => (
                    <span key={i} className="inline-block px-2 py-0.5 rounded-md bg-slate-700/50 text-slate-300 text-xs truncate max-w-[100px]">
                      {kw}
                    </span>
                  ))}
                  {(seg.keywords || []).length > 4 && (
                    <span className="inline-block px-2 py-0.5 text-slate-500 text-xs">
                      +{(seg.keywords || []).length - 4}
                    </span>
                  )}
                </div>
              </div>

              {/* Transcript preview */}
              <div className="mb-3">
                <p className="text-xs text-slate-400 leading-relaxed line-clamp-2">
                  {seg.content || seg.summary?.[0] || "No content available"}
                </p>
              </div>

              {/* Action buttons */}
              <div className="flex gap-2">
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    setExpandedId(expandedId === seg.id ? null : seg.id);
                  }}
                  className="flex-1 text-xs py-1.5 rounded-lg bg-slate-700/60 hover:bg-slate-600 text-slate-300 transition-all border border-slate-600"
                >
                  {expandedId === seg.id ? "Hide" : "Details"}
                </button>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    onSegmentModal?.(seg);
                  }}
                  className="flex-1 text-xs py-1.5 rounded-lg bg-indigo-600/40 hover:bg-indigo-600/60 text-indigo-300 transition-all border border-indigo-500/30"
                >
                  Full
                </button>
              </div>

              {/* Expanded details */}
              {expandedId === seg.id && (
                <div className="mt-3 pt-3 border-t border-slate-700/50 space-y-2">
                  {seg.summary && seg.summary.length > 0 && (
                    <div>
                      <div className="text-xs text-slate-500 font-mono mb-1.5">Summary:</div>
                      <ul className="space-y-1">
                        {seg.summary.map((s, i) => (
                          <li key={i} className="text-xs text-slate-400 flex gap-2">
                            <span className="text-cyan-400 flex-shrink-0">•</span>
                            <span>{s}</span>
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                  {seg.sentimentScore !== undefined && (
                    <div>
                      <div className="text-xs text-slate-500 font-mono mb-1.5">Sentiment Score</div>
                      <div className="h-1.5 bg-slate-700 rounded-full overflow-hidden">
                        <div
                          className="h-full rounded-full transition-all"
                          style={{
                            width: `${Math.abs(seg.sentimentScore) * 100}%`,
                            background: sentimentColor(seg.sentiment),
                          }}
                        />
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
