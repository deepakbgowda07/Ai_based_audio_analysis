import React, { useRef, useEffect } from "react";

const fmtTime = (s) => {
  const seconds = typeof s === "number" ? s : parseFloat(s);
  const m = Math.floor(seconds / 60);
  const sec = Math.floor(seconds % 60);
  return `${m}:${String(sec).padStart(2, "0")}`;
};

const sentimentColor = (sentiment) => {
  if (typeof sentiment === "string") {
    if (sentiment === "positive") return "#10b981";
    if (sentiment === "negative") return "#ef4444";
    return "#3b82f6";
  }
  if (sentiment > 0.3) return "#10b981";
  if (sentiment < -0.3) return "#ef4444";
  return "#3b82f6";
};

export default function SegmentTimeline({ segments = [], activeSegment = 0, onSegmentClick }) {
  const scrollRef = useRef(null);
  const activeRef = useRef(null);

  // Auto-scroll to active segment
  useEffect(() => {
    if (activeRef.current && scrollRef.current) {
      activeRef.current.scrollIntoView({ behavior: "smooth", block: "nearest", inline: "center" });
    }
  }, [activeSegment]);

  if (segments.length === 0) {
    return (
      <div className="w-full bg-slate-800/40 border border-slate-700/50 rounded-xl p-6">
        <div className="text-slate-500 text-sm text-center py-8">
          No segments to display
        </div>
      </div>
    );
  }

  // Calculate total duration for time scale
  const totalDuration = Math.max(
    ...segments.map((seg) => seg.endTimestamp || seg.endTime || 0),
    1
  );

  return (
    <div className="w-full">
      <h3 className="text-slate-300 text-xs font-mono uppercase tracking-widest mb-4">
        Episode Timeline
      </h3>

      <div className="bg-slate-800/40 border border-slate-700/50 rounded-xl p-4">
        {/* Timeline scroll container - FIXED DIMENSIONS */}
        <div
          ref={scrollRef}
          className="overflow-x-auto overflow-y-hidden -mx-4 px-4 scroll-smooth"
          style={{
            scrollBehavior: "smooth",
            WebkitOverflowScrolling: "touch",
          }}
        >
          {/* Custom scrollbar styling */}
          <style>{`
            #timeline-scroll::-webkit-scrollbar {
              height: 10px;
            }
            #timeline-scroll::-webkit-scrollbar-track {
              background: rgba(30, 41, 59, 0.8);
              border-radius: 5px;
              margin: 0 4px;
            }
            #timeline-scroll::-webkit-scrollbar-thumb {
              background: linear-gradient(to right, rgba(6, 182, 212, 0.8), rgba(99, 102, 241, 0.8));
              border-radius: 5px;
              border: 2px solid rgba(30, 41, 59, 0.8);
            }
            #timeline-scroll::-webkit-scrollbar-thumb:hover {
              background: linear-gradient(to right, rgba(6, 182, 212, 1), rgba(99, 102, 241, 1));
              border: 2px solid rgba(15, 23, 42, 1);
              box-shadow: 0 0 6px rgba(6, 182, 212, 0.5);
            }
            #timeline-scroll {
              scroll-behavior: smooth;
            }
          `}</style>

          <div
            id="timeline-scroll"
            className="flex gap-3 pb-2"
            style={{
              display: "flex",
              gap: "12px",
              paddingBottom: "8px",
            }}
          >
            {/* Segment blocks - UNIFORM SIZE */}
            {segments.map((segment) => {
              const isActive = segment.id === activeSegment;
              const color = sentimentColor(segment.sentiment);
              const startTime = fmtTime(segment.startTimestamp || segment.startTime || 0);
              const endTime = fmtTime(segment.endTimestamp || segment.endTime || 0);
              const preview = segment.label?.slice(0, 25) || `Segment ${segment.id}`;

              return (
                <button
                  ref={isActive ? activeRef : null}
                  key={segment.id}
                  onClick={() => onSegmentClick(segment.id)}
                  className={`
                    relative rounded-lg transition-all duration-200 group cursor-pointer
                    flex flex-col p-2.5 text-left
                    ${isActive
                      ? "border-2 shadow-lg shadow-indigo-500/50 ring-2 ring-indigo-400/50 bg-slate-700/60"
                      : "border border-slate-600/40 bg-slate-800/50 hover:bg-slate-700/50 hover:border-slate-500/60"
                    }
                  `}
                  style={{
                    minWidth: "180px",
                    maxWidth: "180px",
                    width: "180px",
                    height: "110px",
                    flexShrink: 0,
                    borderColor: isActive ? color : undefined,
                  }}
                  title={`${segment.label} (${startTime} - ${endTime})`}
                >
                  {/* Segment number */}
                  <div className="text-xs font-bold text-slate-300 font-mono mb-1">
                    SEG {segment.id}
                  </div>

                  {/* Time range */}
                  <div className="text-xs text-slate-500 font-mono mb-1.5 line-clamp-1">
                    {startTime} → {endTime}
                  </div>

                  {/* Preview text - TRUNCATED */}
                  <div className="text-xs text-slate-400 leading-tight mb-2 line-clamp-2 flex-1">
                    "{preview}"
                  </div>

                  {/* Sentiment label */}
                  <div
                    className="text-xs font-semibold text-white px-2 py-0.5 rounded self-start"
                    style={{
                      background: color + "33",
                      color: color,
                    }}
                  >
                    {segment.sentiment || "neutral"}
                  </div>

                  {/* Sentiment color strip at bottom */}
                  <div
                    className="absolute bottom-0 left-0 right-0 h-1 rounded-b-lg"
                    style={{
                      background: color,
                    }}
                  />

                  {/* Hover tooltip */}
                  <div className="absolute -top-24 left-1/2 transform -translate-x-1/2 px-3 py-2 bg-slate-950 border border-slate-600 rounded-lg text-xs text-slate-200 whitespace-nowrap opacity-0 group-hover:opacity-100 pointer-events-none transition-opacity z-50 shadow-xl">
                    <div className="font-mono font-semibold mb-1">{segment.label}</div>
                    <div className="text-slate-400 text-xs mb-0.5">
                      {startTime} → {endTime}
                    </div>
                    <div
                      className="text-xs font-semibold"
                      style={{ color }}
                    >
                      {(segment.confidence * 100).toFixed(0)}% confidence
                    </div>
                  </div>
                </button>
              );
            })}
          </div>
        </div>

        {/* Timeline scale below */}
        <div className="mt-4 px-2 flex justify-between text-xs text-slate-500 font-mono">
          <span>0:00</span>
          <span>{fmtTime(totalDuration / 3)}</span>
          <span>{fmtTime((totalDuration * 2) / 3)}</span>
          <span>{fmtTime(totalDuration)}</span>
        </div>

        {/* Active segment info card */}
        {segments[activeSegment] && (
          <div className="mt-4 bg-slate-900/60 rounded-lg p-4 border border-slate-700/50">
            <div className="flex items-start justify-between gap-4">
              <div className="flex-1 min-w-0">
                <div className="text-sm font-semibold text-slate-200 mb-1">
                  {segments[activeSegment].label}
                </div>
                <div className="text-xs text-slate-500 font-mono mb-2">
                  {fmtTime(segments[activeSegment].startTimestamp || segments[activeSegment].startTime || 0)} - {fmtTime(segments[activeSegment].endTimestamp || segments[activeSegment].endTime || 0)}
                </div>
                {segments[activeSegment].content && (
                  <p className="text-xs text-slate-400 line-clamp-2">
                    {segments[activeSegment].content}
                  </p>
                )}
              </div>
              <div
                className="flex-shrink-0 px-3 py-1.5 rounded text-xs font-semibold text-white"
                style={{
                  background: sentimentColor(segments[activeSegment].sentiment) + "33",
                  color: sentimentColor(segments[activeSegment].sentiment),
                  border: `1px solid ${sentimentColor(segments[activeSegment].sentiment)}66`,
                }}
              >
                {segments[activeSegment].sentiment?.toUpperCase() || "NEUTRAL"}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
