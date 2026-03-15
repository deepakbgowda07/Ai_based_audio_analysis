import React, { useRef, useEffect, useMemo } from "react";

const fmtTime = (s) => {
  const seconds = typeof s === "number" ? s : parseFloat(s);
  const m = Math.floor(seconds / 60);
  const sec = Math.floor(seconds % 60);
  return `${m}:${String(sec).padStart(2, "0")}`;
};

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

export default function TranscriptView({ 
  transcript = [], 
  segments = [], 
  activeSegment, 
  highlightedKeyword, 
  currentTime 
}) {
  const ref = useRef({});
  const groupedBySegment = useMemo(() => {
    const grouped = {};
    transcript.forEach(item => {
      const segId = item.segment || 0;
      if (!grouped[segId]) grouped[segId] = [];
      grouped[segId].push(item);
    });
    return grouped;
  }, [transcript]);

  useEffect(() => {
    if (currentTime !== null && ref.current[currentTime]) {
      ref.current[currentTime].scrollIntoView({ behavior: "smooth", block: "center" });
    }
  }, [currentTime]);

  const segmentMap = useMemo(() => {
    const map = {};
    segments.forEach(seg => {
      map[seg.id] = seg;
    });
    return map;
  }, [segments]);

  if (transcript.length === 0) {
    return (
      <div className="flex items-center justify-center h-96 text-slate-500">
        <div className="text-center">
          <div className="text-4xl mb-2">🎙️</div>
          <p className="text-sm">No transcript data available</p>
        </div>
      </div>
    );
  }

  return (
    <div className="w-full">
      <h3 className="text-slate-300 text-xs font-mono uppercase tracking-widest mb-3">
        Segmented Transcript
      </h3>
      <div className="bg-slate-800/40 border border-slate-700/50 rounded-xl overflow-hidden flex flex-col h-full">
        <div className="overflow-y-auto flex-1 p-4 space-y-6">
          {Object.entries(groupedBySegment).map(([segId, items]) => {
            const segment = segmentMap[parseInt(segId)];
            const segColor = segment ? sentimentColor(segment.sentiment) : "#6b7280";
            const isActive = parseInt(segId) === activeSegment;

            return (
              <div
                key={segId}
                className={`rounded-xl p-4 border-l-4 transition-all ${
                  isActive
                    ? "bg-indigo-500/10 border-indigo-500"
                    : "bg-slate-800/20 border-slate-700/50"
                }`}
                style={{ borderLeftColor: segColor }}
              >
                {/* Topic header */}
                <div className="mb-4 flex items-start justify-between">
                  <div>
                    <div className="text-slate-300 font-semibold text-sm mb-1">
                      TOPIC {segment?.id || segId}
                    </div>
                    <div className="text-slate-500 text-xs font-mono">
                      {segment?.label || `Segment ${segId}`}
                    </div>
                  </div>
                  <div className="flex gap-1.5">
                    <span
                      className="inline-block px-2 py-0.5 rounded text-xs font-semibold text-white"
                      style={{ background: segColor + "22", color: segColor }}
                    >
                      {segment?.sentiment || "neutral"}
                    </span>
                    {segment?.confidence && (
                      <span className="inline-block px-2 py-0.5 rounded text-xs font-semibold text-cyan-400 bg-cyan-500/10">
                        {(segment.confidence * 100).toFixed(0)}%
                      </span>
                    )}
                  </div>
                </div>

                {/* Time range */}
                {segment && (
                  <div className="text-xs text-slate-400 font-mono mb-3">
                    Time Range: {fmtTime(segment.startTimestamp || 0)} → {fmtTime(segment.endTimestamp || 0)}
                  </div>
                )}

                {/* Transcript items */}
                <div className="space-y-2 mb-3">
                  {items.map((item, idx) => {
                    const hasKw = highlightedKeyword && item.keywords?.includes(highlightedKeyword);
                    return (
                      <div
                        key={item.id}
                        ref={(el) => {
                          if (el) ref.current[item.id] = el;
                        }}
                        className={`text-sm leading-relaxed p-2 rounded transition-all ${
                          hasKw ? "bg-cyan-500/10 text-slate-100" : "text-slate-400 hover:text-slate-300"
                        }`}
                        style={{ borderLeft: hasKw ? "2px solid #06b6d4" : "1px solid transparent" }}
                      >
                        <span className="text-slate-600 text-xs font-mono mr-2">
                          {fmtTime(item.start || item.timestamp || 0)}
                        </span>
                        <span>
                          {(item.text || "").split(" ").map((word, wi) => {
                            const cleanWord = word.replace(/[^a-z]/gi, "").toLowerCase();
                            const isKw = item.keywords?.some(
                              (k) => k.toLowerCase() === cleanWord
                            );
                            return (
                              <span
                                key={wi}
                                className={`mr-1 ${isKw ? "text-cyan-400 font-semibold" : ""}`}
                              >
                                {word}
                              </span>
                            );
                          })}
                        </span>
                      </div>
                    );
                  })}
                </div>

                {/* Segment metadata */}
                {segment && (
                  <div className="pt-3 border-t border-slate-700/30 space-y-2">
                    {segment.keywords && segment.keywords.length > 0 && (
                      <div>
                        <div className="text-xs text-slate-500 font-mono mb-1.5">Keywords:</div>
                        <div className="flex flex-wrap gap-1.5">
                          {segment.keywords.map((kw, i) => (
                            <span
                              key={i}
                              className={`inline-block px-2 py-0.5 rounded text-xs font-mono ${
                                kw === highlightedKeyword
                                  ? "bg-yellow-500/20 text-yellow-300 border border-yellow-500/40"
                                  : "bg-slate-700/30 text-slate-400"
                              }`}
                            >
                              {kw}
                            </span>
                          ))}
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
    </div>
  );
}
