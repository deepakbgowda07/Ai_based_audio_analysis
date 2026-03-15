import React, { useRef, useEffect } from "react";

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

export default function SegmentDetails({ 
  segments = [], 
  activeSegment = 0, 
  onSegmentChange,
  onKeywordClick,
  highlightedKeyword 
}) {
  const containerRef = useRef(null);
  const segment = segments[activeSegment];
  const totalSegments = segments.length;

  // Scroll to top when segment changes
  useEffect(() => {
    if (containerRef.current) {
      containerRef.current.scrollTop = 0;
    }
  }, [activeSegment]);

  if (!segment) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="text-slate-500 text-center">
          <div className="text-3xl mb-2">📭</div>
          <p className="text-sm">No segment selected</p>
        </div>
      </div>
    );
  }

  const sc = sentimentColor(segment.sentiment);
  const startTime = fmtTime(segment.startTimestamp || segment.startTime || 0);
  const endTime = fmtTime(segment.endTimestamp || segment.endTime || 0);
  const canGoPrev = activeSegment > 0;
  const canGoNext = activeSegment < totalSegments - 1;

  const goToPrev = () => {
    if (canGoPrev) onSegmentChange?.(activeSegment - 1);
  };

  const goToNext = () => {
    if (canGoNext) onSegmentChange?.(activeSegment + 1);
  };

  return (
    <div className="flex flex-col h-full bg-slate-800/30 rounded-2xl border border-slate-700/50 overflow-hidden">
      {/* Header with segment info */}
      <div className="flex-shrink-0 border-b border-slate-700/50 p-4 bg-slate-800/40">
        <div className="flex items-start justify-between gap-3 mb-3">
          <div>
            <div className="inline-block px-2.5 py-0.5 rounded-full bg-indigo-500/20 border border-indigo-500/40 text-indigo-400 text-xs font-bold mb-2">
              SEGMENT #{segment.id}
            </div>
            <h2 className="text-slate-100 text-xl font-bold">{segment.label}</h2>
          </div>
          <div
            className="flex-shrink-0 px-3 py-1.5 rounded-lg text-white text-sm font-bold"
            style={{ background: sc + "22", color: sc }}
          >
            {segment.sentiment?.toUpperCase() || "NEUTRAL"}
          </div>
        </div>

        {/* Time range and confidence */}
        <div className="flex items-center gap-4">
          <div className="text-xs text-slate-400 font-mono">
            <span className="text-slate-500">Time:</span> {startTime} → {endTime}
          </div>
          {segment.confidence !== undefined && (
            <div className="text-xs text-slate-400 font-mono">
              <span className="text-slate-500">Confidence:</span> {(segment.confidence * 100).toFixed(0)}%
            </div>
          )}
        </div>
      </div>

      {/* Scrollable content */}
      <div ref={containerRef} className="flex-1 overflow-y-auto p-5 space-y-5">
        {/* Segment stats */}
        <div className="grid grid-cols-3 gap-3">
          {segment.wordCount !== undefined && (
            <div className="bg-slate-700/30 rounded-lg p-3 border border-slate-700/50">
              <div className="text-slate-500 text-xs font-mono mb-1">Words</div>
              <div className="text-slate-200 font-bold text-lg">{segment.wordCount}</div>
            </div>
          )}
          {segment.duration && (
            <div className="bg-slate-700/30 rounded-lg p-3 border border-slate-700/50">
              <div className="text-slate-500 text-xs font-mono mb-1">Duration</div>
              <div className="text-slate-200 font-bold text-lg">{segment.duration}</div>
            </div>
          )}
          {segment.sentenceCount !== undefined && (
            <div className="bg-slate-700/30 rounded-lg p-3 border border-slate-700/50">
              <div className="text-slate-500 text-xs font-mono mb-1">Sentences</div>
              <div className="text-slate-200 font-bold text-lg">{segment.sentenceCount}</div>
            </div>
          )}
        </div>

        {/* Summary */}
        {segment.summary && segment.summary.length > 0 && (
          <div>
            <div className="text-slate-400 text-xs font-mono uppercase tracking-widest mb-3">Summary</div>
            <ul className="space-y-2">
              {segment.summary.map((point, idx) => (
                <li key={idx} className="flex gap-2 text-sm text-slate-300">
                  <span className="text-cyan-400 flex-shrink-0 mt-0.5">▸</span>
                  <span>{point}</span>
                </li>
              ))}
            </ul>
          </div>
        )}

        {/* Keywords */}
        {segment.keywords && segment.keywords.length > 0 && (
          <div>
            <div className="text-slate-400 text-xs font-mono uppercase tracking-widest mb-3">Keywords</div>
            <div className="flex flex-wrap gap-2">
              {segment.keywords.map((kw, idx) => {
                const isHighlighted = kw === highlightedKeyword;
                return (
                  <button
                    key={idx}
                    onClick={() => onKeywordClick?.(kw)}
                    className={`
                      px-3 py-1.5 rounded-lg text-xs font-mono transition-all border
                      ${isHighlighted
                        ? "border-yellow-500 bg-yellow-500/20 text-yellow-300 ring-2 ring-yellow-500/40"
                        : "border-slate-600/50 bg-slate-700/30 text-slate-300 hover:bg-slate-700/50 hover:border-slate-500"
                      }
                    `}
                  >
                    {kw}
                  </button>
                );
              })}
            </div>
          </div>
        )}

        {/* Sentiment score bar */}
        {segment.sentimentScore !== undefined && (
          <div>
            <div className="text-slate-400 text-xs font-mono uppercase tracking-widest mb-3">Sentiment Score</div>
            <div className="bg-slate-700/30 rounded-lg p-4 border border-slate-700/50">
              <div className="flex items-center justify-between mb-2">
                <span className="text-slate-400 text-xs">Negative</span>
                <span className="text-slate-400 text-xs">Neutral</span>
                <span className="text-slate-400 text-xs">Positive</span>
              </div>
              <div className="h-2.5 bg-slate-800 rounded-full overflow-hidden">
                <div
                  className="h-full rounded-full transition-all"
                  style={{
                    width: `${((segment.sentimentScore + 1) / 2) * 100}%`,
                    background: sc,
                  }}
                />
              </div>
              <div className="text-xs text-slate-500 text-center mt-2 font-mono">
                {segment.sentimentScore.toFixed(2)}
              </div>
            </div>
          </div>
        )}

        {/* Content preview */}
        {segment.content && (
          <div>
            <div className="text-slate-400 text-xs font-mono uppercase tracking-widest mb-3">Content Preview</div>
            <div className="bg-slate-700/20 rounded-lg p-4 border border-slate-700/50">
              <p className="text-sm text-slate-300 leading-relaxed line-clamp-6">
                {segment.content}
              </p>
            </div>
          </div>
        )}
      </div>

      {/* Navigation footer */}
      <div className="flex-shrink-0 border-t border-slate-700/50 p-4 bg-slate-800/40 flex items-center justify-between gap-3">
        <button
          onClick={goToPrev}
          disabled={!canGoPrev}
          className={`
            flex-1 py-2.5 rounded-lg font-medium text-sm transition-all border
            ${canGoPrev
              ? "bg-slate-700/60 hover:bg-slate-600 text-slate-200 border-slate-600"
              : "bg-slate-800/40 text-slate-500 border-slate-700/50 cursor-not-allowed"
            }
          `}
        >
          ← Previous
        </button>

        {/* Progress indicator */}
        <div className="text-xs text-slate-500 font-mono whitespace-nowrap">
          {activeSegment + 1} / {totalSegments}
        </div>

        <button
          onClick={goToNext}
          disabled={!canGoNext}
          className={`
            flex-1 py-2.5 rounded-lg font-medium text-sm transition-all border
            ${canGoNext
              ? "bg-slate-700/60 hover:bg-slate-600 text-slate-200 border-slate-600"
              : "bg-slate-800/40 text-slate-500 border-slate-700/50 cursor-not-allowed"
            }
          `}
        >
          Next →
        </button>
      </div>
    </div>
  );
}
