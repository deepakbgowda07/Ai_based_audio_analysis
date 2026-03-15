import React from "react";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend
} from "recharts";

const fmtTime = (s) => {
  const seconds = typeof s === "number" ? s : parseFloat(s);
  const m = Math.floor(seconds / 60);
  const sec = Math.floor(seconds % 60);
  return `${m}:${String(sec).padStart(2, "0")}`;
};

const SentimentTooltip = ({ active, payload }) => {
  if (!active || !payload?.length) return null;
  const data = payload[0]?.payload;
  return (
    <div className="bg-slate-900 border border-slate-600 rounded-lg p-2.5 text-xs shadow-lg">
      <div className="text-slate-400 font-mono mb-1">{fmtTime(data?.timestamp || 0)}</div>
      <div className="text-slate-300">
        <span className="text-indigo-400 font-semibold">Sentiment: </span>
        {(data?.sentiment || 0).toFixed(2)}
      </div>
      {data?.label && <div className="text-slate-500 text-xs mt-1">{data.label}</div>}
    </div>
  );
};

export default function SentimentChart({ segments = [] }) {
  // Convert segments to chart data
  const chartData = segments.map((seg, idx) => ({
    name: `Seg ${seg.id}`,
    timestamp: seg.startTimestamp || 0,
    sentiment: seg.sentimentScore || 0,
    label: seg.label || `Segment ${seg.id}`,
  }));

  if (chartData.length === 0) {
    return (
      <div className="w-full h-64 flex items-center justify-center bg-slate-800/40 rounded-lg border border-slate-700/50">
        <div className="text-slate-500 text-sm">No segment data available</div>
      </div>
    );
  }

  return (
    <div className="w-full">
      <h3 className="text-slate-300 text-xs font-mono uppercase tracking-widest mb-3">
        Sentiment Trend Across Segments
      </h3>
      <div className="bg-slate-800/40 border border-slate-700/50 rounded-xl p-3">
        <ResponsiveContainer width="100%" height={250}>
          <LineChart data={chartData} margin={{ top: 5, right: 20, left: -20, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
            <XAxis 
              dataKey="name" 
              tick={{ fill: "#64748b", fontSize: 10 }}
            />
            <YAxis 
              domain={[-1, 1]}
              tick={{ fill: "#64748b", fontSize: 10 }}
              label={{ value: "Sentiment Score", angle: -90, position: "insideLeft", style: { fill: "#94a3b8", fontSize: 10 } }}
            />
            <Tooltip content={<SentimentTooltip />} />
            <Legend wrapperStyle={{ paddingTop: "10px" }} />
            <Line
              type="monotone"
              dataKey="sentiment"
              stroke="#10b981"
              strokeWidth={2.5}
              dot={{ fill: "#10b981", r: 4 }}
              activeDot={{ r: 6 }}
              name="Sentiment"
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
