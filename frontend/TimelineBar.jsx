import React, { useMemo } from "react";

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

export default function TimelineBar({ segments = [], activeSegment, onSegmentClick }) {
  const totalDuration = useMemo(() => {
    if (segments.length === 0) return 0;
    const lastSeg = segments[segments.length - 1];
    return lastSeg.endTimestamp || lastSeg.endTime || 0;
  }, [segments]);

  if (segments.length === 0) {
    return (
      <div className="w-full h-16 flex items-center justify-center bg-slate-800/40 rounded-lg border border-slate-700/50">
        <div className="text-slate-500 text-sm">No segments to display</div>
      </div>
    );
  }

  return (
    <div className="w-full">
      <h3 className="text-slate-300 text-xs font-mono uppercase tracking-widest mb-3">
        Segment Timeline
      </h3>
      <div className="bg-slate-800/40 border border-slate-700/50 rounded-xl p-4">
        {/* Timeline bar */}
        <div className="relative h-12 bg-slate-900/50 rounded-lg overflow-hidden border border-slate-700/30">
          {segments.map((seg, idx) => {
            const startPercent = totalDuration > 0 ? ((seg.startTimestamp || 0) / totalDuration) * 100 : 0;
            const endPercent = totalDuration > 0 ? ((seg.endTimestamp || 0) / totalDuration) * 100 : 0;
            const width = endPercent - startPercent;
            const isActive = seg.id === activeSegment;
            const color = sentimentColor(seg.sentiment);

            return (
              <button
                key={seg.id}
                onClick={() => onSegmentClick(seg.id)}
                className="absolute h-full group transition-all hover:opacity-100"
                style={{
                  left: `${startPercent}%`,
                  width: `${Math.max(width, 2)}%`,
                  background: color,
                  opacity: isActive ? 1 : 0.6,
                }}
                title={`${seg.label} (${fmtTime(seg.startTimestamp || 0)} - ${fmtTime(seg.endTimestamp || 0)})`}
              >
                {/* Tooltip on hover */}
                <div className="absolute bottom-full left-1/2 transform -translate-x-1/2 mb-2 px-2 py-1 bg-slate-900 border border-slate-600 rounded text-xs text-slate-300 whitespace-nowrap opacity-0 group-hover:opacity-100 pointer-events-none transition-opacity z-10 shadow-lg">
                  {seg.label?.slice(0, 20)}...
                </div>

                {/* Segment border for active */}
                {isActive && (
                  <div className="absolute inset-0 border-2 border-white/80 pointer-events-none" />
                )}
              </button>
            );
          })}
        </div>

        {/* Time markers */}
        <div className="flex justify-between text-xs text-slate-500 font-mono mt-2">
          <span>0:00</span>
          <span>{fmtTime(totalDuration / 2)}</span>
          <span>{fmtTime(totalDuration)}</span>
        </div>

        {/* Segment labels below timeline */}
        <div className="mt-4 space-y-2">
          {segments.map((seg) => {
            const isActive = seg.id === activeSegment;
            const color = sentimentColor(seg.sentiment);
            return (
              <div
                key={seg.id}
                onClick={() => onSegmentClick(seg.id)}
                className={`flex items-center gap-2 px-3 py-1.5 rounded-lg cursor-pointer transition-all ${
                  isActive
                    ? "bg-indigo-500/20 border border-indigo-500/40"
                    : "bg-slate-700/30 border border-slate-700/40 hover:bg-slate-700/50"
                }`}
              >
                <div
                  className="w-2 h-2 rounded-full flex-shrink-0"
                  style={{ background: color }}
                />
                <span className="text-xs text-slate-300 flex-1 truncate">
                  {seg.label?.slice(0, 30) || `Segment ${seg.id}`}
                </span>
                <span className="text-xs text-slate-500 font-mono whitespace-nowrap">
                  {fmtTime(seg.startTimestamp || 0)} - {fmtTime(seg.endTimestamp || 0)}
                </span>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
